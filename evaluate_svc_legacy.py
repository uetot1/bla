"""Evaluate the released SVC Base checkpoints on the common detection set.

The upstream SVC test path reports entropy-likelihood BPP and does not emit an
arithmetic-coded bitstream.  This evaluator intentionally preserves that rate
definition and records it as ``estimated_bpp``; it must not be presented as
actual-bitstream BPP.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import torch
import yaml
from tqdm import tqdm


PROJECT = Path(__file__).resolve().parent
LEGACY = PROJECT / "legacy_svc_base"
ORIGINAL_SVC_COMMIT = "c32dd98"

sys.path.insert(0, str(LEGACY))
sys.path.insert(1, str(PROJECT))

from entropy_models import estimate_bpp  # noqa: E402
from networks_canf import (  # noqa: E402
    AugmentedNormalizedFlowHyperPriorCoder,
    __CODER_TYPES__,
)
from test_base import Pframe  # noqa: E402
from Utils import Alignment  # noqa: E402

from dcvc_rt.src.models.yolov5_extractor import (  # noqa: E402
    _allow_legacy_yolov5_checkpoint_loading,
)
from dcvc_rt.src.utils.detection_map import DetectionMAP  # noqa: E402
from dcvc_rt.src.utils.evaluation_protocol import (  # noqa: E402
    ALL_FRAMES_PROTOCOL,
    dataset_summary,
    detector_config,
    evaluation_id,
)
from dcvc_rt.src.utils.vcm_eval_dataset import AnnotatedVideoDataset  # noqa: E402
from models.common import AutoShape, DetectMultiBackend  # noqa: E402


LEGACY_ASSETS = ("SDCNet_3M_ref3.ckpt", "pwc_net.pth.tar")


def restore_legacy_assets() -> None:
    """Restore large initialization files from the repository's SVC commit."""
    destination = LEGACY / "models"
    destination.mkdir(exist_ok=True)
    for name in LEGACY_ASSETS:
        path = destination / name
        if path.is_file():
            continue
        with path.open("wb") as output:
            subprocess.run(
                ["git", "show", f"{ORIGINAL_SVC_COMMIT}:models/{name}"],
                cwd=PROJECT,
                stdout=output,
                check=True,
            )
        if path.stat().st_size == 0:
            raise RuntimeError(f"Failed to restore {path}")


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as source:
        return yaml.safe_load(source)


def make_coder(config_name: str):
    config = load_yaml(LEGACY / "config" / config_name)
    architecture = __CODER_TYPES__[config["model_architecture"]]
    return architecture(**config["model_params"])


def build_system(checkpoint_path: Path, yolo_weights: Path, device: torch.device):
    """Build the exact released Base model and load its checkpoint detector."""
    previous_directory = Path.cwd()
    os.chdir(LEGACY)
    try:
        model = Pframe(
            make_coder("DVC_motion.yml"),
            make_coder("CANF_motion_predprior.yml"),
            make_coder("CANF_inter_coder.yml"),
        )
    finally:
        os.chdir(previous_directory)

    with _allow_legacy_yolov5_checkpoint_loading():
        initial_detector = DetectMultiBackend(
            weights=yolo_weights,
            device=device,
            dnn=False,
            fp16=False,
        )
    model.feature_extractor = copy.deepcopy(initial_detector)
    del initial_detector

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    state = checkpoint.get("state_dict", checkpoint)
    state = {key.removeprefix("module."): value for key, value in state.items()}
    model.load_state_dict(state, strict=True)
    model = model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    detector = AutoShape(model.feature_extractor, verbose=False).to(device).eval()
    return model, detector


def as_yolo_image(frame: torch.Tensor):
    return (
        frame.detach()
        .squeeze(0)
        .permute(1, 2, 0)
        .cpu()
        .numpy()
        .clip(0, 1)
        * 255
    ).astype("uint8")


@torch.inference_mode()
def evaluate_checkpoint(
    checkpoint_path: Path,
    checkpoint_number: int,
    dataset: AnnotatedVideoDataset,
    sequences,
    yolo_weights: Path,
    device: torch.device,
    args,
) -> dict:
    model, detector = build_system(checkpoint_path, yolo_weights, device)
    detector.conf = args.confidence_threshold
    detector.iou = args.nms_iou_threshold
    detector.max_det = args.max_detections

    evaluator = DetectionMAP()
    sequence_records = []
    image_id = 0

    progress = tqdm(sequences, desc=f"Original SVC Base checkpoint {checkpoint_number}")
    for sequence in progress:
        model.frame_buffer = []
        model.flow_buffer = []
        model.MWNet.clear_buffer()
        alignment = Alignment().to(device)
        estimated_bits = 0.0
        reference = None
        p_order = 1

        for frame_index, frame_path in enumerate(sequence.frame_paths):
            coding_frame = dataset.load_frame(frame_path).unsqueeze(0).to(device)
            if frame_index % args.gop == 0:
                reconstructed, likelihoods, _ = model.if_model(
                    alignment.align(coding_frame)
                )
                reconstructed = alignment.resume(reconstructed).clamp(0, 1)
                frame_bpp = estimate_bpp(likelihoods, input=reconstructed).mean()
                p_order = 1
            else:
                if p_order == 1:
                    model.frame_buffer = [alignment.align(reference)]
                reconstructed_aligned, likelihoods = model.forward_pair(
                    alignment.align(reference),
                    alignment.align(coding_frame),
                    p_order,
                )
                reconstructed_aligned = reconstructed_aligned.clamp(0, 1)
                model.frame_buffer.append(reconstructed_aligned)
                reconstructed = alignment.resume(reconstructed_aligned)
                frame_bpp = estimate_bpp(likelihoods, input=coding_frame).mean()
                p_order += 1

            if len(model.frame_buffer) == 4:
                model.frame_buffer.pop(0)
            reference = reconstructed
            estimated_bits += float(frame_bpp) * sequence.width * sequence.height

            detections = detector(
                [as_yolo_image(reconstructed)],
                size=args.detector_size,
            ).xyxy[0].detach().cpu()
            target_boxes, target_classes = dataset.load_ground_truth(
                sequence.label_paths[frame_index],
                sequence.width,
                sequence.height,
            )
            evaluator.add(
                image_id=image_id,
                predicted_boxes=detections[:, :4],
                predicted_scores=detections[:, 4],
                predicted_classes=detections[:, 5].long(),
                target_boxes=target_boxes,
                target_classes=target_classes,
            )
            image_id += 1

        pixels = sequence.frame_count * sequence.width * sequence.height
        sequence_records.append(
            {
                "name": sequence.name,
                "estimated_entropy_bits": estimated_bits,
                "estimated_bpp": estimated_bits / pixels,
                "width": sequence.width,
                "height": sequence.height,
                "coded_frames": sequence.frame_count,
                "fps": sequence.fps,
            }
        )

    total_bits = sum(record["estimated_entropy_bits"] for record in sequence_records)
    total_pixels = sum(
        record["coded_frames"] * record["width"] * record["height"]
        for record in sequence_records
    )
    total_frames = sum(record["coded_frames"] for record in sequence_records)
    point = {
        "checkpoint_number": checkpoint_number,
        "checkpoint": str(checkpoint_path.resolve()),
        "estimated_entropy_bits": total_bits,
        "estimated_bpp": total_bits / total_pixels,
        "coded_frames": total_frames,
        "sequences": sequence_records,
        **evaluator.compute(),
    }

    del detector, model
    torch.cuda.empty_cache()
    return point


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, nargs=4, required=True)
    parser.add_argument("--yolov5-weights", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gop", type=int, default=32)
    parser.add_argument("--detector-size", type=int, default=640)
    parser.add_argument("--confidence-threshold", type=float, default=0.001)
    parser.add_argument("--nms-iou-threshold", type=float, default=0.6)
    parser.add_argument("--max-detections", type=int, default=300)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Original SVC evaluation requires a CUDA GPU")
    missing = [str(path) for path in [*args.checkpoints, args.yolov5_weights] if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required file: " + missing[0])

    restore_legacy_assets()
    device = torch.device("cuda:0")
    dataset = AnnotatedVideoDataset(args.data_dir, args.dataset_manifest)
    sequences = list(dataset)
    points = [
        evaluate_checkpoint(
            checkpoint,
            number,
            dataset,
            sequences,
            args.yolov5_weights,
            device,
            args,
        )
        for number, checkpoint in enumerate(args.checkpoints, start=1)
    ]

    output = {
        "schema_version": 7,
        "method": "Original SVC Base (estimated BPP)",
        "codec": "Released SVC Base / LCCM-VC-CANF reference implementation",
        "rate_source": "estimated_entropy_likelihoods_from_official_test_base",
        "rate_is_actual_bitstream": False,
        "warning": "Do not present estimated_bpp as actual_bpp.",
        "protocol": ALL_FRAMES_PROTOCOL,
        "comparison_scope": "end-to-end VCM system",
        "task": "object_detection",
        "task_model": "checkpoint-embedded YOLOv5s",
        "ground_truth": "normalized YOLO labels from evaluation manifest",
        "evaluation_id": evaluation_id(dataset, sequences),
        "dataset": dataset_summary(dataset, sequences),
        "detector_config": detector_config(
            "yolov5s",
            args.detector_size,
            args.confidence_threshold,
            args.nms_iou_threshold,
            args.max_detections,
            args.yolov5_weights,
        ),
        "machine_frontend": {
            "type": "checkpoint_embedded_original_svc_yolov5",
            "checkpoint_specific": True,
        },
        "codec_config": {"gop": args.gop, "checkpoints": [str(p) for p in args.checkpoints]},
        "rate_points": len(points),
        "points": points,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "Original_SVC_Base_estimated_results.json"
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Saved Original SVC estimated-BPP results to {output_path}")


if __name__ == "__main__":
    main()
