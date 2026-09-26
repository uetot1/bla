"""HEVC anchor for the segmentation axis.

Every mask number in this project so far is measured against the paper's own checkpoint,
which answers "did continued training help" but not "is this codec any good at
segmentation". The detection axis has an external anchor (x265 HEVC, where the paper
reaches -73%); the mask axis had none. This builds it.

The coding path is not reimplemented: the x265 encode, decode and colour conversion come
from ``evaluate_hevc.py`` unchanged, so the anchor is produced by the same audited
commands that made the detection anchor. Only the scoring differs -- masks from the frozen
segmentation model instead of boxes against labels -- and that part is imported from
``evaluate_mask_proxy`` so the results metadata is byte-identical and BD-rate accepts the
pair.

Run it with the same --bit-depth/--chroma-format as the detection anchor (8-bit 4:2:0 for
this project) or the two anchors describe different codecs.
"""

import argparse
import json
import shutil
from pathlib import Path

import torch

from dcvc_rt.src.utils.evaluation_protocol import ALL_FRAMES_PROTOCOL, dataset_summary, evaluation_id
from dcvc_rt.src.utils.vcm_eval_dataset import AnnotatedVideoDataset
from evaluate_hevc import (raw_frame_bytes, resolve_executable, rgb_to_yuv, safe_name,
                           write_concat_file, x265_decode, x265_encode, x265_version,
                           yuv_to_rgb_frames)
from multitask_exp.evaluate_mask_proxy import (COMPARISON_SCOPE, GROUND_TRUTH, SegPredictor,
                                              make_reference, score_decoded_frame,
                                              segmentation_config)
from multitask_exp.mask_map import MaskMAP
import numpy as np
from PIL import Image

MIN_RATE_POINT_COUNT = 4


def load_png(path):
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def score_sequence(predictor, dataset, sequence, decoded_frames, evaluator, first_image_id,
                   sequence_evaluator=None, reference=None):
    """Same comparison as the neural path, through the same scoring function."""
    image_id = first_image_id
    for frame_index, decoded_path in enumerate(decoded_frames):
        (predicted_masks, scores, classes,
         target_masks, target_classes) = score_decoded_frame(
            predictor, reference, dataset, sequence, frame_index, load_png(decoded_path))
        for metric in (evaluator, sequence_evaluator):
            if metric is not None:
                metric.add(image_id=image_id, predicted_masks=predicted_masks,
                           predicted_scores=scores, predicted_classes=classes,
                           target_masks=target_masks, target_classes=target_classes)
        image_id += 1
    return image_id


def evaluate(args):
    if len(args.qps) < MIN_RATE_POINT_COUNT or len(set(args.qps)) != len(args.qps):
        raise ValueError("At least four distinct x265 QPs are required")
    if not torch.cuda.is_available():
        raise RuntimeError("The segmentation model needs CUDA for a run of this size")

    ffmpeg = resolve_executable(args.ffmpeg, "FFmpeg")
    encoder = resolve_executable(args.x265, "x265")
    device = torch.device(f"cuda:{args.cuda_index}")
    dataset = AnnotatedVideoDataset(args.data_dir, args.dataset_manifest)
    sequences = list(dataset)
    if args.max_sequences is not None:
        sequences = sequences[: args.max_sequences]
    if not sequences:
        raise RuntimeError("No evaluation sequences were selected")

    predictor = SegPredictor(args.seg_weights, device, args.detector_size,
                             args.confidence_threshold, args.nms_iou_threshold,
                             args.max_detections)
    seg_config = segmentation_config(predictor, args)
    reference, ground_truth = make_reference(args, device)

    work = Path(args.work_dir) / safe_name(args.method_name)
    points = []
    for qp in args.qps:
        evaluator = MaskMAP()
        records = []
        next_image_id = 0
        for sequence in sequences:
            folder = work / f"qp_{qp:02d}" / safe_name(sequence.name)
            folder.mkdir(parents=True, exist_ok=True)
            concat_path = folder / "frames.ffconcat"
            source_yuv = folder / "source.yuv"
            bitstream = folder / "stream.hevc"
            decoded_yuv = folder / "decoded.yuv"
            decoded_dir = folder / "decoded"

            write_concat_file(sequence, concat_path)
            rgb_to_yuv(ffmpeg, sequence, concat_path, source_yuv, args.bit_depth,
                       args.chroma_format)
            expected = raw_frame_bytes(sequence.width, sequence.height, args.bit_depth,
                                       args.chroma_format) * sequence.frame_count
            if source_yuv.stat().st_size != expected:
                raise RuntimeError(f"{sequence.name}: raw YUV size mismatch before encoding")
            x265_encode(encoder, sequence, source_yuv, bitstream, qp, args.bit_depth,
                        args.chroma_format, args.preset, [])
            x265_decode(ffmpeg, sequence, bitstream, decoded_yuv, args.bit_depth,
                        args.chroma_format)
            frames = yuv_to_rgb_frames(ffmpeg, sequence, decoded_yuv, decoded_dir,
                                       args.bit_depth, args.chroma_format)

            sequence_evaluator = MaskMAP()
            next_image_id = score_sequence(predictor, dataset, sequence, frames, evaluator,
                                           next_image_id, sequence_evaluator, reference)
            actual_bits = bitstream.stat().st_size * 8
            coded_frames = sequence.frame_count
            record = {
                "name": sequence.name,
                "actual_bits": int(actual_bits),
                "actual_bpp": actual_bits / (coded_frames * sequence.width * sequence.height),
                "kbps": actual_bits * sequence.fps / (1000.0 * coded_frames),
                "fps": sequence.fps,
                "width": sequence.width,
                "height": sequence.height,
                "coded_frames": coded_frames,
                **sequence_evaluator.compute(),
            }
            records.append(record)
            print(f"  qp {qp:2d} {sequence.name:<20} bpp {record['actual_bpp']:.5f}  "
                  f"mask map50 {record['map50']:.4f}", flush=True)
            # The raw YUV and PNG frames are large; keep only the bitstream for the record.
            source_yuv.unlink(missing_ok=True)
            decoded_yuv.unlink(missing_ok=True)
            shutil.rmtree(decoded_dir, ignore_errors=True)

        total_bits = sum(r["actual_bits"] for r in records)
        total_pixels = sum(r["coded_frames"] * r["width"] * r["height"] for r in records)
        total_duration = sum(r["coded_frames"] / r["fps"] for r in records)
        points.append({
            "qp": qp,
            "actual_bits": int(total_bits),
            "actual_bpp": float(total_bits / total_pixels),
            "kbps": float(total_bits / total_duration / 1000.0),
            "coded_frames": int(sum(r["coded_frames"] for r in records)),
            **evaluator.compute(),
            "sequences": records,
        })
        print(f"qp {qp:2d} TOTAL  bpp {points[-1]['actual_bpp']:.5f}  "
              f"mask map50 {points[-1]['map50']:.4f}", flush=True)

    output = {
        "schema_version": 7,
        "method": args.method_name,
        "codec": f"x265 HEVC {args.bit_depth}-bit {args.chroma_format}",
        "codec_config": {
            "encoder_version": x265_version(encoder),
            "qps": list(args.qps),
            "preset": args.preset,
            "bit_depth": args.bit_depth,
            "chroma_format": args.chroma_format,
            "color_pipeline": "RGB -> BT.709 YUV -> x265 -> YUV -> RGB",
        },
        "protocol": ALL_FRAMES_PROTOCOL,
        "comparison_scope": COMPARISON_SCOPE,
        "rate_source": "complete independently decodable x265 HEVC bitstream bytes including headers",
        "rate_points": len(points),
        "task": "instance_segmentation",
        "task_model": "yolov5s-seg",
        "ground_truth": ground_truth,
        "evaluation_id": evaluation_id(dataset, sequences),
        "dataset": dataset_summary(dataset, sequences),
        "detector_config": seg_config,
        "machine_frontend": {"type": "frozen_pretrained_yolov5_seg",
                             "task_backend_weights_id": seg_config["weights_id"]},
        "points": points,
    }
    output_path = Path(args.output_dir) / f"{safe_name(args.method_name)}_mask_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Saved HEVC mask anchor to {output_path}")


def self_check(args):
    """CPU-only: the metadata must match what evaluate_mask_proxy writes, key for key."""
    from multitask_exp import evaluate_mask_proxy

    device = torch.device(args.device)
    predictor = SegPredictor(args.seg_weights, device, args.detector_size,
                             args.confidence_threshold, args.nms_iou_threshold,
                             args.max_detections)
    mine = segmentation_config(predictor, args)
    assert set(mine) == {"task_model", "weights_id", "input_size", "confidence_threshold",
                         "nms_iou_threshold", "max_detections", "class_count", "mask_space"}, mine
    assert evaluate_mask_proxy.COMPARISON_SCOPE == COMPARISON_SCOPE
    assert evaluate_mask_proxy.GROUND_TRUTH == GROUND_TRUTH
    # BD-rate refuses a pair whose fingerprints differ, so the two producers must agree
    # on every key AND value for the same settings.
    assert evaluate_mask_proxy.segmentation_config(predictor, args) == mine
    print("evaluate_mask_hevc self-check passed: results metadata is identical to the "
          f"neural mask path ({len(mine)} keys, weights {mine['weights_id'][:16]}...)")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir")
    parser.add_argument("--dataset-manifest")
    parser.add_argument("--seg-weights", default="multitask_exp/weights/yolov5s-seg.pt")
    parser.add_argument("--qps", type=int, nargs="+", default=[22, 27, 32, 37, 42, 47])
    parser.add_argument("--preset", default="medium")
    parser.add_argument("--bit-depth", type=int, choices=(8, 10), default=8)
    parser.add_argument("--chroma-format", choices=("420", "444"), default="420")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--x265", default="x265")
    parser.add_argument("--detector-size", type=int, default=640)
    parser.add_argument("--confidence-threshold", type=float, default=0.25)
    parser.add_argument("--nms-iou-threshold", type=float, default=0.45)
    parser.add_argument("--max-detections", type=int, default=300)
    parser.add_argument("--max-sequences", type=int, default=None)
    parser.add_argument("--work-dir", default="multitask_exp/output/hevc_mask_work")
    parser.add_argument("--output-dir", default="multitask_exp/output/mask_results")
    parser.add_argument("--method-name", default="HEVC_x265")
    parser.add_argument("--cuda-index", type=int, default=0)
    parser.add_argument("--device", default="cpu", help="self-check only")
    parser.add_argument("--ground-truth", choices=("proxy", "kitti-mots"), default="proxy")
    parser.add_argument("--gt-masks-dir", help="kitti-mots only: the prepared masks/ folder")
    parser.add_argument("--self_check", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.self_check:
        self_check(arguments)
    else:
        evaluate(arguments)
