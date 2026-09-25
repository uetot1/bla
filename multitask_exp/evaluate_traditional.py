"""Traditional-codec anchors on both task axes, so the RD figure has more than two curves.

The paper compares four methods and only one of them is a standard codec (HEVC/x265).
A reader of a codec paper expects the standard ladder -- AVC, HEVC, and ideally VVC -- so
the gain is read against the progression of the field, not against a single point. This
script produces those anchors on the SAME evaluation protocol as everything else:

* the coding path (RGB -> YUV -> encode -> decode -> RGB) is the one in ``evaluate_hevc.py``,
  imported unchanged, so the x265 curve reproduces the existing anchor exactly;
* x264 mirrors those settings one for one (Low-Delay-P: one leading I-frame, no B-frames,
  fixed GOP, scenecut off), so AVC and HEVC differ by the codec and nothing else;
* the detection scoring is ``evaluate_hevc.evaluate_reconstructions`` and the segmentation
  scoring is ``multitask_exp.evaluate_mask_proxy``, so both axes carry metadata identical to
  the neural results and BD-rate accepts every pair.

vvenc/vvdec are wired in the same way but only run when both binaries are present; they are
not installed on Kaggle by default, and this says so instead of failing halfway.
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from dcvc_rt.src.models.yolov5_extractor import load_yolov5
from dcvc_rt.src.utils.detection_map import DetectionMAP
from dcvc_rt.src.utils.evaluation_protocol import (ALL_FRAMES_PROTOCOL, dataset_summary,
                                                   detector_config, evaluation_id)
from dcvc_rt.src.utils.vcm_eval_dataset import AnnotatedVideoDataset
from evaluate_hevc import (evaluate_reconstructions, raw_frame_bytes, resolve_executable,
                           rgb_to_yuv, run_command, safe_name, write_concat_file, x265_decode,
                           x265_encode, x265_profile, x265_version, yuv_to_rgb_frames)
from multitask_exp.evaluate_mask_proxy import (COMPARISON_SCOPE, GROUND_TRUTH, SegPredictor,
                                               segmentation_config)
from multitask_exp.mask_map import MaskMAP

MIN_RATE_POINT_COUNT = 4


def x264_encode(encoder, sequence, input_yuv, bitstream_path, qp, bit_depth, chroma_format,
                preset, extra_arguments):
    """Same coding structure as x265_encode, so AVC and HEVC differ only by the standard."""
    rounded_fps = int(round(sequence.fps))
    if abs(sequence.fps - rounded_fps) > 1e-6:
        raise ValueError(f"x264 requires an integer frame rate; {sequence.name} uses {sequence.fps}")
    if chroma_format != "420" or bit_depth != 8:
        raise ValueError("This x264 path is built for the project's 8-bit 4:2:0 anchors")
    bitstream_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        encoder,
        "--input-res", f"{sequence.width}x{sequence.height}",
        "--fps", str(rounded_fps),
        "--frames", str(sequence.frame_count),
        "--input-csp", "i420",
        "--input-depth", str(bit_depth),
        "--profile", "high",
        "--qp", str(qp),
        "--preset", preset,
        # Low-Delay P, mirroring the HEVC anchor.
        "--bframes", "0",
        "--keyint", str(sequence.frame_count),
        "--min-keyint", str(sequence.frame_count),
        "--no-scenecut",
        # No --repeat-headers: that is an x265 option. x264 writes SPS/PPS with every IDR,
        # and this structure has exactly one, so the stream is self-contained anyway.
        "--log-level", "warning",
        *extra_arguments,
        "--output", str(bitstream_path),
        str(input_yuv),
    ]
    return run_command(command, f"x264 encode for {sequence.name} at QP {qp}")


def vvenc_encode(encoder, sequence, input_yuv, bitstream_path, qp, bit_depth, chroma_format,
                 preset, extra_arguments):
    if chroma_format != "420":
        raise ValueError("vvencapp codes 4:2:0")
    bitstream_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        encoder,
        "--input", str(input_yuv),
        "--size", f"{sequence.width}x{sequence.height}",
        "--framerate", str(int(round(sequence.fps))),
        "--frames", str(sequence.frame_count),
        "--format", f"yuv420{'_10' if bit_depth == 10 else ''}",
        "--qp", str(qp),
        "--preset", preset,
        "--intraperiod", str(sequence.frame_count),
        "--refreshsec", "0",
        *extra_arguments,
        "--output", str(bitstream_path),
    ]
    return run_command(command, f"vvenc encode for {sequence.name} at QP {qp}")


def ffmpeg_decode(ffmpeg, sequence, bitstream_path, reconstructed_yuv, bit_depth, chroma_format):
    """Decode any elementary stream FFmpeg understands into the shared raw-YUV path."""
    pixel_format = {("420", 8): "yuv420p", ("420", 10): "yuv420p10le"}[(chroma_format, bit_depth)]
    command = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-i", str(bitstream_path),
        "-frames:v", str(sequence.frame_count), "-pix_fmt", pixel_format,
        "-f", "rawvideo", "-y", str(reconstructed_yuv),
    ]
    run_command(command, f"FFmpeg decode for {sequence.name}")
    expected = raw_frame_bytes(sequence.width, sequence.height, bit_depth,
                               chroma_format) * sequence.frame_count
    if reconstructed_yuv.stat().st_size != expected:
        raise RuntimeError(f"{sequence.name}: decoded {reconstructed_yuv.stat().st_size} bytes, "
                           f"expected {expected}")


ENCODERS = {
    # name: (default binary, encode function, decode function, default preset, label)
    "x265": ("x265", x265_encode, x265_decode, "medium", "HEVC (x265)"),
    "x264": ("x264", x264_encode, ffmpeg_decode, "medium", "AVC (x264)"),
    "vvenc": ("vvencapp", vvenc_encode, ffmpeg_decode, "medium", "VVC (vvenc)"),
}


def box_scoring_arguments(detector, evaluator, dataset, sequence, frames, args, first_image_id,
                          qp):
    """Positional arguments for evaluate_hevc.evaluate_reconstructions, in its order.

    Built in one place so the self-check can bind them against the real signature: a call
    that missed three required parameters once passed every paper check and only failed on
    Kaggle, after the encodes.
    """
    return (detector, evaluator, dataset, sequence, frames, args.detector_size, first_image_id,
            "all-frames", f"{args.method_name}: QP {qp} YOLO {sequence.name}",
            args.detector_batch_size)


def load_png(path):
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def score_masks(predictor, dataset, sequence, frames, evaluator, first_image_id,
                sequence_evaluator):
    image_id = first_image_id
    for frame_index, decoded_path in enumerate(frames):
        source = dataset.load_frame(sequence.frame_paths[frame_index])
        source_uint8 = (source.permute(1, 2, 0).numpy().clip(0, 1) * 255).astype(np.uint8)
        target_masks, _, target_classes = predictor(source_uint8)
        predicted_masks, scores, classes = predictor(load_png(decoded_path))
        for metric in (evaluator, sequence_evaluator):
            metric.add(image_id=image_id, predicted_masks=predicted_masks,
                       predicted_scores=scores, predicted_classes=classes,
                       target_masks=target_masks, target_classes=target_classes)
        image_id += 1
    return image_id


def evaluate(args):
    if len(args.qps) < MIN_RATE_POINT_COUNT or len(set(args.qps)) != len(args.qps):
        raise ValueError("At least four distinct QPs are required")
    axes = ("box", "mask") if args.axis == "both" else (args.axis,)
    if "mask" in axes and not torch.cuda.is_available():
        raise RuntimeError("The task models need CUDA for a run of this size")

    binary, encode, decode, default_preset, label = ENCODERS[args.encoder]
    encoder = resolve_executable(args.encoder_binary or binary, args.encoder)
    ffmpeg = resolve_executable(args.ffmpeg, "FFmpeg")
    preset = args.preset or default_preset
    device = torch.device(f"cuda:{args.cuda_index}")
    dataset = AnnotatedVideoDataset(args.data_dir, args.dataset_manifest)
    sequences = list(dataset)
    if args.max_sequences is not None:
        sequences = sequences[: args.max_sequences]

    detector = seg_predictor = None
    if "box" in axes:
        detector = load_yolov5(args.task_model, repository=args.yolov5_repo,
                               weights=args.yolov5_weights).to(device).eval()
        for parameter in detector.parameters():
            parameter.requires_grad_(False)
        detector.conf = args.confidence_threshold
        detector.iou = args.nms_iou_threshold
        detector.max_det = args.max_detections
    if "mask" in axes:
        seg_predictor = SegPredictor(args.seg_weights, device, args.seg_detector_size,
                                     args.seg_confidence, args.seg_nms_iou, args.max_detections)

    work = Path(args.work_dir) / safe_name(args.method_name)
    points = {axis: [] for axis in axes}
    for qp in args.qps:
        evaluators = {"box": DetectionMAP() if "box" in axes else None,
                      "mask": MaskMAP() if "mask" in axes else None}
        records = {axis: [] for axis in axes}
        image_ids = {axis: 0 for axis in axes}
        for sequence in sequences:
            folder = work / f"qp_{qp:02d}" / safe_name(sequence.name)
            folder.mkdir(parents=True, exist_ok=True)
            source_yuv, bitstream = folder / "source.yuv", folder / "stream.bin"
            decoded_yuv, decoded_dir = folder / "decoded.yuv", folder / "decoded"

            write_concat_file(sequence, folder / "frames.ffconcat")
            rgb_to_yuv(ffmpeg, sequence, folder / "frames.ffconcat", source_yuv,
                       args.bit_depth, args.chroma_format)
            encode(encoder, sequence, source_yuv, bitstream, qp, args.bit_depth,
                   args.chroma_format, preset, [])
            decode(ffmpeg, sequence, bitstream, decoded_yuv, args.bit_depth, args.chroma_format)
            frames = yuv_to_rgb_frames(ffmpeg, sequence, decoded_yuv, decoded_dir,
                                       args.bit_depth, args.chroma_format)

            actual_bits = bitstream.stat().st_size * 8
            rate = {
                "name": sequence.name, "actual_bits": int(actual_bits),
                "actual_bpp": actual_bits / (sequence.frame_count * sequence.width * sequence.height),
                "kbps": actual_bits * sequence.fps / (1000.0 * sequence.frame_count),
                "fps": sequence.fps, "width": sequence.width, "height": sequence.height,
                "coded_frames": sequence.frame_count,
            }
            if "box" in axes:
                per_sequence = DetectionMAP()
                image_ids["box"] = evaluate_reconstructions(
                    *box_scoring_arguments(detector, evaluators["box"], dataset, sequence,
                                           frames, args, image_ids["box"], qp),
                    sequence_evaluator=per_sequence)
                records["box"].append({**rate, **per_sequence.compute()})
            if "mask" in axes:
                per_sequence = MaskMAP()
                image_ids["mask"] = score_masks(seg_predictor, dataset, sequence, frames,
                                                evaluators["mask"], image_ids["mask"], per_sequence)
                records["mask"].append({**rate, **per_sequence.compute()})
            print(f"  qp {qp:2d} {sequence.name:<20} bpp {rate['actual_bpp']:.5f}" +
                  "".join(f"  {axis} map50 {records[axis][-1]['map50']:.4f}" for axis in axes),
                  flush=True)
            source_yuv.unlink(missing_ok=True)
            decoded_yuv.unlink(missing_ok=True)
            shutil.rmtree(decoded_dir, ignore_errors=True)

        for axis in axes:
            group = records[axis]
            total_bits = sum(r["actual_bits"] for r in group)
            total_pixels = sum(r["coded_frames"] * r["width"] * r["height"] for r in group)
            total_duration = sum(r["coded_frames"] / r["fps"] for r in group)
            points[axis].append({
                "qp": qp, "actual_bits": int(total_bits),
                "actual_bpp": float(total_bits / total_pixels),
                "kbps": float(total_bits / total_duration / 1000.0),
                "coded_frames": int(sum(r["coded_frames"] for r in group)),
                **evaluators[axis].compute(), "sequences": group,
            })
            print(f"qp {qp:2d} TOTAL {axis}  bpp {points[axis][-1]['actual_bpp']:.5f}  "
                  f"map50 {points[axis][-1]['map50']:.4f}", flush=True)

    codec_config = {
        "encoder": args.encoder, "encoder_version": (x265_version(encoder)
                                                     if args.encoder == "x265" else "n/a"),
        "qps": list(args.qps), "preset": preset, "bit_depth": args.bit_depth,
        "chroma_format": args.chroma_format,
        "coding_structure": "Low-Delay-P: one leading I-frame, no B-frames, scenecut disabled",
        "color_pipeline": "RGB -> BT.709 YUV -> encode -> decode -> RGB",
    }
    for axis in axes:
        if axis == "box":
            task_metadata = {
                "task": "object_detection", "task_model": args.task_model,
                "ground_truth": "normalized YOLO labels from evaluation manifest",
                "comparison_scope": "end-to-end VCM system",
                "detector_config": detector_config(args.task_model, args.detector_size,
                                                   args.confidence_threshold,
                                                   args.nms_iou_threshold, args.max_detections,
                                                   args.yolov5_weights),
                "machine_frontend": {"type": "pretrained_yolov5_frontend"},
            }
            suffix = "results"
        else:
            seg_config = segmentation_config(seg_predictor, argparse.Namespace(
                detector_size=args.seg_detector_size, confidence_threshold=args.seg_confidence,
                nms_iou_threshold=args.seg_nms_iou, max_detections=args.max_detections))
            task_metadata = {
                "task": "instance_segmentation", "task_model": "yolov5s-seg",
                "ground_truth": GROUND_TRUTH, "comparison_scope": COMPARISON_SCOPE,
                "detector_config": seg_config,
                "machine_frontend": {"type": "frozen_pretrained_yolov5_seg",
                                     "task_backend_weights_id": seg_config["weights_id"]},
            }
            suffix = "mask_results"
        output = {
            "schema_version": 7, "method": args.method_name, "codec": label,
            "codec_config": codec_config, "protocol": ALL_FRAMES_PROTOCOL,
            "rate_source": f"complete {label} elementary stream bytes including headers",
            "rate_points": len(points[axis]),
            "evaluation_id": evaluation_id(dataset, sequences),
            "dataset": dataset_summary(dataset, sequences),
            **task_metadata, "points": points[axis],
        }
        path = Path(args.output_dir) / f"{safe_name(args.method_name)}_{suffix}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(output, indent=2), encoding="utf-8")
        print(f"Saved {axis} results to {path}")


def self_check(args):
    """The x264 command must mirror the HEVC anchor's coding structure, flag for flag."""
    from types import SimpleNamespace

    sequence = SimpleNamespace(name="test", width=416, height=240, fps=30.0, frame_count=50)
    captured = {}

    def fake_run(command, label):
        captured["command"] = command
        return ""

    import evaluate_hevc
    import multitask_exp.evaluate_traditional as module

    original = evaluate_hevc.run_command
    module_original = module.run_command
    try:
        evaluate_hevc.run_command = fake_run
        module.run_command = fake_run
        module.x264_encode("x264", sequence, Path("in.yuv"), Path("out/stream.bin"), 32, 8,
                           "420", "medium", [])
        avc = captured["command"]
        evaluate_hevc.x265_encode("x265", sequence, Path("in.yuv"), Path("out/stream.bin"), 32,
                                  8, "420", "medium", [])
        hevc = captured["command"]
    finally:
        evaluate_hevc.run_command = original
        module.run_command = module_original
        shutil.rmtree("out", ignore_errors=True)

    for flag, value in (("--qp", "32"), ("--preset", "medium"), ("--bframes", "0"),
                        ("--keyint", "50"), ("--min-keyint", "50")):
        for name, command in (("x264", avc), ("x265", hevc)):
            assert flag in command and command[command.index(flag) + 1] == value, (name, flag)
    assert "--no-scenecut" in avc and "--scenecut" in hevc, "scenecut must be off in both"
    assert set(ENCODERS) == {"x264", "x265", "vvenc"}, ENCODERS.keys()

    # The box-scoring call must bind to the real function signature.
    import inspect

    fake_args = SimpleNamespace(detector_size=640, method_name="AVC_x264", detector_batch_size=16)
    bound = inspect.signature(evaluate_reconstructions).bind(
        *box_scoring_arguments(None, None, None, sequence, [], fake_args, 0, 32),
        sequence_evaluator=None)
    assert bound.arguments["protocol_key"] == "all-frames", bound.arguments
    print("evaluate_traditional self-check passed: x264 mirrors the HEVC anchor's coding "
          f"structure (qp, preset, bframes 0, fixed GOP {sequence.frame_count}, scenecut off), "
          "and the box-scoring call binds to evaluate_reconstructions' real signature")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--encoder", choices=tuple(ENCODERS), default="x264")
    parser.add_argument("--encoder-binary")
    parser.add_argument("--axis", choices=("box", "mask", "both"), default="both")
    parser.add_argument("--data-dir")
    parser.add_argument("--dataset-manifest")
    parser.add_argument("--qps", type=int, nargs="+", default=[22, 27, 32, 37, 42, 47])
    parser.add_argument("--preset")
    parser.add_argument("--bit-depth", type=int, choices=(8, 10), default=8)
    parser.add_argument("--chroma-format", choices=("420", "444"), default="420")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--task-model", default="yolov5s")
    parser.add_argument("--yolov5-repo")
    parser.add_argument("--yolov5-weights")
    parser.add_argument("--detector-size", type=int, default=640)
    parser.add_argument("--confidence-threshold", type=float, default=0.001)
    parser.add_argument("--nms-iou-threshold", type=float, default=0.6)
    parser.add_argument("--detector-batch-size", type=int, default=16,
                        help="same default as evaluate_hevc.py; changes speed only")
    parser.add_argument("--seg-weights", default="multitask_exp/weights/yolov5s-seg.pt")
    parser.add_argument("--seg-detector-size", type=int, default=640)
    parser.add_argument("--seg-confidence", type=float, default=0.25)
    parser.add_argument("--seg-nms-iou", type=float, default=0.45)
    parser.add_argument("--max-detections", type=int, default=300)
    parser.add_argument("--max-sequences", type=int, default=None)
    parser.add_argument("--work-dir", default="multitask_exp/output/traditional_work")
    parser.add_argument("--output-dir", default="multitask_exp/output/traditional")
    parser.add_argument("--method-name", default="AVC_x264")
    parser.add_argument("--cuda-index", type=int, default=0)
    parser.add_argument("--self_check", action="store_true")
    parser.add_argument("--probe", action="store_true",
                        help="encode + decode 4 tiny frames with the real binary, then exit")
    return parser.parse_args()


def probe(args):
    """Encode and decode four tiny frames with the REAL binary before a long run.

    The self-check compares command lines on paper; it cannot know which options a given
    build accepts. A flag copied from x265 cost a whole Kaggle start-up once. This fails in
    seconds instead, with the encoder's own error message.
    """
    import os
    from types import SimpleNamespace

    binary, encode, decode, default_preset, label = ENCODERS[args.encoder]
    encoder = resolve_executable(args.encoder_binary or binary, args.encoder)
    ffmpeg = resolve_executable(args.ffmpeg, "FFmpeg")
    sequence = SimpleNamespace(name="probe", width=64, height=64, fps=30.0, frame_count=4)
    folder = Path(args.work_dir) / "_probe" / args.encoder
    folder.mkdir(parents=True, exist_ok=True)
    source = folder / "source.yuv"
    source.write_bytes(os.urandom(raw_frame_bytes(64, 64, args.bit_depth, args.chroma_format) * 4))
    bitstream, decoded = folder / "stream.bin", folder / "decoded.yuv"
    encode(encoder, sequence, source, bitstream, 32, args.bit_depth, args.chroma_format,
           args.preset or default_preset, [])
    decode(ffmpeg, sequence, bitstream, decoded, args.bit_depth, args.chroma_format)
    size = bitstream.stat().st_size
    shutil.rmtree(folder, ignore_errors=True)
    print(f"probe passed: {label} encoded and decoded 4 frames ({size} bytes) with {encoder}")


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.self_check:
        self_check(arguments)
    elif arguments.probe:
        probe(arguments)
    else:
        evaluate(arguments)
