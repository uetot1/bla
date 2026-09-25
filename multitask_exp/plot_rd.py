"""Rate-distortion figures and BD-rate tables for the thesis, from existing results files.

Every curve is drawn from a schema-v7 results JSON (neural codecs from evaluate_vcm.py or
evaluate_mask_proxy.py, standard codecs from evaluate_hevc.py or evaluate_traditional.py),
so the figure shows exactly the points the BD-rate numbers were computed from. Dominated
points are kept as hollow markers but left out of the line, matching how BD-rate treats
them.

BD-rate is computed with the project's own functions against one anchor per panel, and a
file whose evaluation fingerprint differs from the anchor's is refused rather than plotted,
because a curve from a different test set would look comparable and not be.
"""

import argparse
import json
from pathlib import Path

import numpy as np

STYLE = {
    # method key: (label, colour, marker, line style)
    "avc": ("AVC (x264)", "#9A8F7A", "v", (0, (4, 2))),
    "hevc": ("HEVC (x265)", "#6B7280", "s", (0, (4, 2))),
    "dcvcrt": ("DCVC-RT gốc", "#C4512E", "^", "-"),
    "paper": ("DCVC-RT-VCM (bài)", "#2F6F8F", "o", "-"),
    "multitask": ("DCVC-RT-VCM đa tác vụ", "#1B2433", "D", "-"),
}


def load(path):
    text = Path(path).read_text(encoding="utf-8-sig").lstrip()
    return json.JSONDecoder().raw_decode(text)[0]


def curve(data, metric):
    rates = np.asarray([p["actual_bpp"] for p in data["points"]], dtype=float)
    quality = np.asarray([p[metric] for p in data["points"]], dtype=float)
    order = np.argsort(rates)
    rates, quality = rates[order], quality[order]
    kept, best = [], -np.inf
    for index, value in enumerate(quality):
        if value > best:
            kept.append(index)
            best = value
    mask = np.zeros(len(rates), dtype=bool)
    mask[kept] = True
    return rates, quality, mask


def bd_rate(anchor, candidate, metric):
    from dcvc_rt.src.utils.bd_rate import compute_bd_rate, pareto_front

    try:
        a_rate, a_q = pareto_front([p["actual_bpp"] for p in anchor["points"]],
                                   [p[metric] for p in anchor["points"]])
        c_rate, c_q = pareto_front([p["actual_bpp"] for p in candidate["points"]],
                                   [p[metric] for p in candidate["points"]])
        return compute_bd_rate(a_rate, a_q, c_rate, c_q)
    except ValueError:
        return None


# Detector settings that change the score. weights_id is left out on purpose: the same
# yolov5s weights are recorded as "torch-hub:..." by one script and as a sha256 by another.
DETECTOR_FIELDS = ("model", "input_size", "confidence_threshold", "nms_iou_threshold",
                   "max_detections")


def incompatibility(anchor, data):
    """Why `data` cannot share a panel with `anchor`, or None when it can."""
    if anchor.get("evaluation_id") and data.get("evaluation_id") not in (
            None, anchor["evaluation_id"]):
        return "evaluation_id khac moc -> khong cung tap test"
    task_a = anchor.get("task", "object_detection")
    task_d = data.get("task", "object_detection")
    if task_a != task_d:
        return f"task {task_d} khac moc ({task_a})"
    config_a, config_d = anchor.get("detector_config"), data.get("detector_config")
    if isinstance(config_a, dict) and isinstance(config_d, dict):
        differing = [f for f in DETECTOR_FIELDS if config_a.get(f) != config_d.get(f)]
        if differing:
            return "detector_config khac moc o " + ", ".join(differing)
    return None


def draw_panel(axis, series, metric, title, anchor_key):
    """series: {method_key: results dict}. Returns {method_key: BD-rate vs anchor}."""
    anchor = series.get(anchor_key)
    table = {}
    for key, data in series.items():
        reason = incompatibility(anchor, data) if anchor else None
        if reason:
            print(f"  [{title}] bo qua {key}: {reason}")
            continue
        label, colour, marker, style = STYLE[key]
        rates, quality, kept = curve(data, metric)
        axis.plot(rates[kept], quality[kept], color=colour, linestyle=style, linewidth=2.2,
                  marker=marker, markersize=7, label=label, zorder=3)
        if (~kept).any():
            axis.plot(rates[~kept], quality[~kept], linestyle="none", marker=marker,
                      markersize=7, markerfacecolor="none", markeredgecolor=colour, zorder=2)
        table[key] = 0.0 if key == anchor_key else (bd_rate(anchor, data, metric)
                                                     if anchor else None)
    axis.set_xscale("log")
    axis.set_xlabel("bpp (bitstream thật)")
    axis.set_ylabel("mAP@0.5" if metric == "map50" else "mAP@0.5:0.95")
    axis.set_title(title)
    axis.grid(True, which="both", color="#DDD8CB", linewidth=0.8)
    axis.legend(fontsize=9, frameon=False, loc="lower right")
    return table


def figure(panels, output, metric="map50", anchor_key="hevc", columns=2):
    """panels: list of (title, {method_key: path}). Writes a PNG, returns the BD tables."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = int(np.ceil(len(panels) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(6.2 * columns, 4.6 * rows), squeeze=False)
    tables = {}
    for axis, (title, paths) in zip(axes.flat, panels):
        series = {key: load(path) for key, path in paths.items() if path and Path(path).is_file()}
        tables[title] = draw_panel(axis, series, metric, title, anchor_key)
    for axis in list(axes.flat)[len(panels):]:
        axis.set_visible(False)
    fig.tight_layout()
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    plt.close(fig)
    return tables


def markdown_table(tables, anchor_key="hevc"):
    methods = [key for key in STYLE if any(key in t for t in tables.values())]
    header = "| | " + " | ".join(STYLE[k][0] for k in methods) + " |"
    lines = [header, "|---|" + "---|" * len(methods)]
    for title, table in tables.items():
        cells = []
        for key in methods:
            value = table.get(key)
            cells.append("—" if value is None else ("mốc" if key == anchor_key
                                                    else f"{value:+.1f} %"))
        lines.append(f"| {title} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def self_check(args):
    """Synthetic curves: a codec at 60% of the anchor's rate must read about -40%."""
    import tempfile

    rates = np.array([0.02, 0.04, 0.08, 0.16, 0.32, 0.64])
    quality = np.array([0.20, 0.30, 0.40, 0.47, 0.52, 0.55])

    def write(path, scale, fingerprint="sha256:x"):
        Path(path).write_text(json.dumps({
            "evaluation_id": fingerprint,
            "points": [{"actual_bpp": float(r * scale), "map50": float(q), "map5095": float(q) * .7}
                       for r, q in zip(rates, quality)]}))

    with tempfile.TemporaryDirectory() as folder:
        folder = Path(folder)
        write(folder / "hevc.json", 1.0)
        write(folder / "paper.json", 0.6)
        write(folder / "other.json", 0.5, fingerprint="sha256:other")
        tables = figure([("kiem tra", {"hevc": folder / "hevc.json",
                                       "paper": folder / "paper.json",
                                       "dcvcrt": folder / "other.json"})],
                        folder / "out.png")
        value = tables["kiem tra"]["paper"]
        assert abs(value - (-40.0)) < 0.5, value
        assert "dcvcrt" not in tables["kiem tra"], "a foreign test set must be refused"
        base = {"evaluation_id": "sha256:x", "detector_config": {"model": "yolov5s",
                                                                 "input_size": 640}}
        assert incompatibility(base, dict(base)) is None
        assert incompatibility(base, {**base, "task": "instance_segmentation"})
        assert incompatibility(base, {**base, "detector_config": {"model": "yolov5s",
                                                                  "input_size": 1280}})
        assert incompatibility(base, {**base, "detector_config": {
            **base["detector_config"], "weights_id": "sha256:other"}}) is None
        assert (folder / "out.png").stat().st_size > 10_000
        print(markdown_table(tables))
    print(f"plot_rd self-check passed: codec at 0.6x rate reads {value:+.1f} %, "
          "a curve from another test set, task or detector setting is refused, figure written")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--self_check", action="store_true")
    arguments = parser.parse_args()
    if arguments.self_check:
        self_check(arguments)
