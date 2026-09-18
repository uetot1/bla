"""Make a raw VCM evaluation folder loadable by AnnotatedVideoDataset.

Some SFU label exports carry boxes that stick out past the frame border, which
``load_ground_truth`` rejects outright. The fix is to clip those boxes to the
frame and leave everything else untouched, exactly as the original Class-C
evaluation notebook did inline.

The clipped copy must be byte-identical whenever it is rebuilt, because
``evaluation_id`` fingerprints label text and frame paths: the anchor run and
the candidate runs only compare if both rebuilt the same dataset. So the output
layout, the file order and the number formatting here are fixed, and frames are
copied rather than linked (``_relative_name`` resolves symlinks and would then
record absolute mount paths that differ between sessions).

A dataset whose labels are already valid is returned untouched, so a clean class
keeps the ``evaluation_id`` its existing result files were measured under.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

COORDINATE_FORMAT = "{:.8f}"


def _iter_label_rows(text):
    for line_number, line in enumerate(text.splitlines(), 1):
        if line.strip():
            yield line_number, line.split()


def _clip_row(fields):
    """Return (class_id, cx, cy, w, h) clipped to the frame, or None if unchanged."""
    class_id = float(fields[0])
    center_x, center_y, width, height = (float(value) for value in fields[1:])
    if all(0.0 <= value <= 1.0 for value in (center_x, center_y, width, height)):
        return None
    left = max(0.0, center_x - width / 2.0)
    top = max(0.0, center_y - height / 2.0)
    right = min(1.0, center_x + width / 2.0)
    bottom = min(1.0, center_y + height / 2.0)
    if right <= left or bottom <= top:
        raise ValueError("box lies entirely outside the frame")
    return (
        int(class_id),
        (left + right) / 2.0,
        (top + bottom) / 2.0,
        right - left,
        bottom - top,
    )


def inspect_labels(data_dir, manifest_path):
    """Count rows that need clipping, without writing anything."""
    description = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    out_of_range = 0
    for entry in description["sequences"]:
        labels_dir = Path(data_dir) / entry["labels_dir"]
        for label_path in sorted(labels_dir.glob("*.txt")):
            text = label_path.read_text(encoding="utf-8-sig")
            for line_number, fields in _iter_label_rows(text):
                if len(fields) != 5:
                    raise ValueError(
                        f"{label_path}:{line_number} must contain "
                        "class_id x_center y_center width height"
                    )
                try:
                    if _clip_row(fields) is not None:
                        out_of_range += 1
                except ValueError as error:
                    raise ValueError(f"{label_path}:{line_number} {error}") from error
    return out_of_range


def prepare(data_dir, manifest_path, output_root, copy_frames=True, verbose=True):
    """Return (data_dir, manifest_path) ready for AnnotatedVideoDataset.

    Returns the inputs unchanged when every label is already valid.
    """
    data_dir = Path(data_dir)
    manifest_path = Path(manifest_path)
    out_of_range = inspect_labels(data_dir, manifest_path)
    if out_of_range == 0:
        if verbose:
            print(f"{data_dir}: nhan da hop le, giu nguyen bo goc")
        return data_dir, manifest_path

    output_root = Path(output_root)
    description = json.loads(manifest_path.read_text(encoding="utf-8"))
    prepared_entries = []
    clipped_rows = 0
    for entry in description["sequences"]:
        name = entry["name"]
        labels_out = output_root / "labels" / name
        frames_out = output_root / "frames" / name
        if labels_out.exists():
            shutil.rmtree(labels_out)
        labels_out.mkdir(parents=True)
        for label_path in sorted((data_dir / entry["labels_dir"]).glob("*.txt")):
            rows = []
            for _, fields in _iter_label_rows(label_path.read_text(encoding="utf-8-sig")):
                clipped = _clip_row(fields)
                if clipped is None:
                    class_id, *values = fields
                    rows.append(" ".join([str(int(float(class_id)))]
                                         + [COORDINATE_FORMAT.format(float(v)) for v in values]))
                else:
                    clipped_rows += 1
                    class_id, *values = clipped
                    rows.append(" ".join([str(class_id)]
                                         + [COORDINATE_FORMAT.format(v) for v in values]))
            (labels_out / label_path.name).write_text(
                "\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")

        frames_in = data_dir / entry["frames_dir"]
        if copy_frames:
            source_frames = sorted(path for path in frames_in.iterdir() if path.is_file())
            if not frames_out.is_dir() or len(list(frames_out.iterdir())) != len(source_frames):
                if frames_out.exists():
                    shutil.rmtree(frames_out)
                frames_out.mkdir(parents=True)
                for frame_path in source_frames:
                    shutil.copy2(frame_path, frames_out / frame_path.name)
            frames_dir = f"frames/{name}"
        else:
            frames_dir = str(frames_in)
        prepared_entries.append({
            "name": name,
            "frames_dir": frames_dir,
            "labels_dir": f"labels/{name}",
            "fps": entry["fps"],
        })

    prepared_manifest = output_root / "manifest.json"
    prepared_manifest.write_text(
        json.dumps({"sequences": prepared_entries}, indent=2) + "\n", encoding="utf-8")
    if verbose:
        print(f"{data_dir}: da cat {clipped_rows} box vuot bien -> {output_root}")
    return output_root, prepared_manifest
