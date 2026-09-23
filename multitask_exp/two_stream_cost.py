"""RQ3: does ONE shared bitstream cost less than TWO task-specific ones?

The Gray-Wyner argument says coding once for two requirements never costs more than
coding twice, and the gap is the value of the shared part. This turns that into a number
from results files already produced:

    two streams  = rate the paper codec needs for detection quality Q_d
                 + rate the segmentation-only codec needs for mask quality Q_s
    one  stream  = rate the multi-task codec needs to deliver BOTH Q_d and Q_s

Q_d and Q_s are read off the multi-task run itself at each of its rate points, so the
comparison is "what would it have cost to buy this exact pair of qualities separately".

Rates are interpolated the way BD-rate does it: PCHIP through (quality, log10 rate) on the
Pareto front, and a target outside a curve's measured range is skipped rather than
extrapolated.

Caveat worth stating in any write-up: the box axis is evaluated with --codec-precision
fp16 and the mask axis in fp32, so one checkpoint's two files do not carry bit-identical
bitrates. Re-run one axis to match before quoting a final figure.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.interpolate import PchipInterpolator

from dcvc_rt.src.utils.bd_rate import pareto_front


def curve(path, metric="map50"):
    """(quality -> log10 bpp) interpolator plus the measured quality range."""
    points = json.loads(Path(path).read_text(encoding="utf-8"))["points"]
    rate, quality = pareto_front([p["actual_bpp"] for p in points],
                                 [p[metric] for p in points])
    order = np.argsort(quality)
    quality, log_rate = quality[order], np.log10(rate[order])
    if np.any(np.diff(quality) <= 0):
        raise ValueError(f"{path}: quality must be strictly increasing after pareto_front")
    return PchipInterpolator(quality, log_rate), float(quality.min()), float(quality.max())


def rate_for(interpolator, low, high, target):
    return None if not (low <= target <= high) else float(10 ** interpolator(target))


def compare(box_paper, box_multi, mask_seg_only, mask_multi, multi_name="multi-task"):
    det, det_low, det_high = curve(box_paper)
    seg, seg_low, seg_high = curve(mask_seg_only)
    multi_box = json.loads(Path(box_multi).read_text(encoding="utf-8"))["points"]
    multi_mask = {p["base_qp"]: p for p in
                  json.loads(Path(mask_multi).read_text(encoding="utf-8"))["points"]}

    rows = []
    for point in multi_box:
        mask_point = multi_mask.get(point["base_qp"])
        if mask_point is None:
            continue
        target_det, target_seg = point["map50"], mask_point["map50"]
        detection_rate = rate_for(det, det_low, det_high, target_det)
        segmentation_rate = rate_for(seg, seg_low, seg_high, target_seg)
        if detection_rate is None or segmentation_rate is None:
            rows.append({"base_qp": point["base_qp"], "skipped":
                         "detection quality outside the paper curve" if detection_rate is None
                         else "mask quality outside the segmentation-only curve"})
            continue
        two = detection_rate + segmentation_rate
        one = point["actual_bpp"]
        rows.append({"base_qp": point["base_qp"], "target_box_map50": target_det,
                     "target_mask_map50": target_seg, "one_stream_bpp": one,
                     "two_stream_bpp": two, "detection_stream_bpp": detection_rate,
                     "segmentation_stream_bpp": segmentation_rate,
                     "saving_percent": 100.0 * (one - two) / two})
    usable = [r for r in rows if "saving_percent" in r]
    summary = {"rate_points_compared": len(usable),
               "mean_saving_percent": sum(r["saving_percent"] for r in usable) / len(usable)
               if usable else float("nan"),
               "multi_task_run": multi_name, "points": rows}
    return summary


def report(summary):
    print(f"{'qp':>4} {'box mAP':>9} {'mask mAP':>9} {'1 stream':>10} {'2 streams':>10} "
          f"{'det':>9} {'seg':>9} {'saving':>9}")
    for row in summary["points"]:
        if "skipped" in row:
            print(f"{row['base_qp']:>4}   bo qua: {row['skipped']}")
            continue
        print(f"{row['base_qp']:>4} {row['target_box_map50']:>9.4f} {row['target_mask_map50']:>9.4f} "
              f"{row['one_stream_bpp']:>10.5f} {row['two_stream_bpp']:>10.5f} "
              f"{row['detection_stream_bpp']:>9.5f} {row['segmentation_stream_bpp']:>9.5f} "
              f"{row['saving_percent']:>8.1f}%")
    print(f"\nTrung binh tren {summary['rate_points_compared']} diem toc do: "
          f"{summary['mean_saving_percent']:+.1f} % "
          "(am = mot bitstream chung RE HON hai bitstream rieng)")


def self_check():
    """Two identical tasks served by identical codecs must show a 50% saving."""
    import tempfile

    def fake(path, qualities, rates):
        Path(path).write_text(json.dumps({
            "schema_version": 7,
            "points": [{"base_qp": index, "actual_bpp": rate, "map50": quality,
                        "map5095": quality * 0.8}
                       for index, (rate, quality) in enumerate(zip(rates, qualities))]}))

    rates = [0.02, 0.05, 0.1, 0.2, 0.4, 0.8]
    qualities = [0.10, 0.20, 0.30, 0.38, 0.44, 0.48]
    with tempfile.TemporaryDirectory() as folder:
        paths = [Path(folder) / f"{name}.json" for name in ("bp", "bm", "ms", "mm")]
        for path in paths:
            fake(path, qualities, rates)
        summary = compare(*paths)
    assert summary["rate_points_compared"] == len(rates), summary
    assert math.isclose(summary["mean_saving_percent"], -50.0, abs_tol=1e-6), summary
    print("two_stream_cost self-check passed: identical curves give exactly -50 % "
          "(one stream costs half of two)")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--box-paper", help="box results of the detection-only baseline (paper)")
    parser.add_argument("--box-multi", help="box results of the multi-task run")
    parser.add_argument("--mask-seg-only", help="mask results of the segmentation-only run")
    parser.add_argument("--mask-multi", help="mask results of the multi-task run")
    parser.add_argument("--multi-name", default="multi-task")
    parser.add_argument("--output", help="write the summary JSON here")
    parser.add_argument("--self_check", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.self_check:
        self_check()
    else:
        for required in ("box_paper", "box_multi", "mask_seg_only", "mask_multi"):
            if getattr(arguments, required) is None:
                raise SystemExit(f"--{required.replace('_', '-')} is required")
        result = compare(arguments.box_paper, arguments.box_multi, arguments.mask_seg_only,
                         arguments.mask_multi, arguments.multi_name)
        report(result)
        if arguments.output:
            Path(arguments.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
            print("saved", arguments.output)
