"""Full-resolution annotated video dataset for machine-task evaluation."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms import functional as transforms


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}


def _natural_key(path: Path) -> list[tuple[int, int | str]]:
    return [
        (0, int(part)) if part.isdigit() else (1, part.lower())
        for part in re.split(r"(\d+)", path.name)
    ]


@dataclass(frozen=True)
class VideoSequence:
    name: str
    frame_paths: tuple[Path, ...]
    label_paths: tuple[Path, ...]
    fps: float
    width: int
    height: int

    @property
    def frame_count(self) -> int:
        return len(self.frame_paths)


class AnnotatedVideoDataset:
    """Read full-resolution frame sequences and YOLO-format ground truth.

    Manifest format::

        {
          "sequences": [
            {
              "name": "Kimono",
              "frames_dir": "frames/Kimono",
              "labels_dir": "labels/Kimono",
              "fps": 24
            }
          ]
        }

    A label file must exist for every frame. Each row uses normalized YOLO
    bounding-box format: ``class_id x_center y_center width height``.
    Empty label files represent frames without annotated objects.
    """

    def __init__(self, root_dir: str | Path, manifest: str | Path):
        self.root_dir = Path(root_dir)
        if not self.root_dir.is_dir():
            raise FileNotFoundError(f"Evaluation root not found: {self.root_dir}")

        manifest_path = Path(manifest)
        if not manifest_path.is_file():
            manifest_path = self.root_dir / manifest_path
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Evaluation manifest not found: {manifest}")

        description = json.loads(manifest_path.read_text(encoding="utf-8"))
        entries = description.get("sequences")
        if not isinstance(entries, list) or not entries:
            raise ValueError("Evaluation manifest must contain a non-empty 'sequences' list")
        self.sequences = tuple(self._load_sequence(entry) for entry in entries)
        names = [sequence.name for sequence in self.sequences]
        if len(names) != len(set(names)):
            raise ValueError("Sequence names in the evaluation manifest must be unique")

    def _load_sequence(self, entry: dict) -> VideoSequence:
        for key in ("name", "frames_dir", "labels_dir", "fps"):
            if key not in entry:
                raise ValueError(f"Sequence entry is missing required field '{key}'")

        frames_dir = self.root_dir / entry["frames_dir"]
        labels_dir = self.root_dir / entry["labels_dir"]
        if not frames_dir.is_dir():
            raise FileNotFoundError(f"Frame directory not found: {frames_dir}")
        if not labels_dir.is_dir():
            raise FileNotFoundError(f"Label directory not found: {labels_dir}")

        frame_paths = tuple(
            sorted(
                (
                    path
                    for path in frames_dir.iterdir()
                    if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
                ),
                key=_natural_key,
            )
        )
        if len(frame_paths) < 2:
            raise ValueError(f"{frames_dir} must contain a seed and at least one P-frame")

        label_paths = tuple(labels_dir / f"{path.stem}.txt" for path in frame_paths)
        missing = [path for path in label_paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                f"Missing {len(missing)} label files; first missing label: {missing[0]}. "
                "Use an empty .txt file for a frame without objects."
            )

        fps = float(entry["fps"])
        if fps <= 0:
            raise ValueError(f"fps must be positive for sequence {entry['name']}")

        with Image.open(frame_paths[0]) as image:
            width, height = image.size
        for path in frame_paths[1:]:
            with Image.open(path) as image:
                if image.size != (width, height):
                    raise ValueError(
                        f"All frames in {frames_dir} must have resolution {width}x{height}; "
                        f"{path.name} has {image.width}x{image.height}"
                    )

        return VideoSequence(
            name=str(entry["name"]),
            frame_paths=frame_paths,
            label_paths=label_paths,
            fps=fps,
            width=width,
            height=height,
        )

    def __iter__(self):
        return iter(self.sequences)

    def __len__(self) -> int:
        return len(self.sequences)

    @staticmethod
    def load_frame(path: Path) -> torch.Tensor:
        with Image.open(path) as image:
            return transforms.to_tensor(image.convert("RGB"))

    @staticmethod
    def load_ground_truth(
        path: Path,
        width: int,
        height: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        boxes = []
        classes = []
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            fields = line.split()
            if len(fields) != 5:
                raise ValueError(
                    f"{path}:{line_number} must contain "
                    "class_id x_center y_center width height"
                )
            class_id, x_center, y_center, box_width, box_height = map(float, fields)
            if class_id < 0 or int(class_id) != class_id:
                raise ValueError(f"{path}:{line_number} has an invalid class id")
            if not all(0.0 <= value <= 1.0 for value in (x_center, y_center, box_width, box_height)):
                raise ValueError(f"{path}:{line_number} coordinates must be normalized to [0, 1]")
            if box_width <= 0 or box_height <= 0:
                raise ValueError(f"{path}:{line_number} box width and height must be positive")

            left = (x_center - box_width / 2.0) * width
            top = (y_center - box_height / 2.0) * height
            right = (x_center + box_width / 2.0) * width
            bottom = (y_center + box_height / 2.0) * height
            boxes.append(
                [
                    max(0.0, left),
                    max(0.0, top),
                    min(float(width), right),
                    min(float(height), bottom),
                ]
            )
            classes.append(int(class_id))

        if not boxes:
            return torch.empty((0, 4), dtype=torch.float32), torch.empty((0,), dtype=torch.int64)
        return torch.tensor(boxes, dtype=torch.float32), torch.tensor(classes, dtype=torch.int64)
