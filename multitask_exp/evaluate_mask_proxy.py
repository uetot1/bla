"""Segmentation-axis evaluation for the multi-task extension, on the SFU sequences.

Every number in this project so far lives on the detection axis. Multi-task is a
trade-off question, so one axis answers nothing: R1 gives up 23-35 BD-rate points of
detection, and we have no idea what it bought. This script measures the other axis.

SFU has no instance-mask labels, so the reference here is what the frozen
``yolov5s-seg`` predicts on the **uncompressed** frame. The candidate is what the same
frozen model predicts on the decoded frame. The metric is therefore "how much does
coding this bitstream damage segmentation", which is exactly the VCM question and
exactly what the training loss targets -- but it is NOT accuracy against human labels.
For that, KITTI-MOTS is still required; this is the cheap first axis that works on the
checkpoints already trained, on the sequences already evaluated for detection.

The codec path is not reimplemented: ``encode_sequence``, ``sequence_rate_record`` and
``aggregate_rate`` are imported from ``evaluate_vcm`` unchanged, so bits are counted by
the same audited code, and the output uses its schema v7 so ``evaluate_vcm.py --compare``
computes BD-rate on the mask axis with no new BD-rate code.

Masks are compared in the detector's letterboxed input space. Reference and candidate go
through an identical transform, so the comparison is fair; absolute values would shift
slightly if masks were resampled back to source resolution first.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from dcvc_rt.src.models.image_model import DMCI
from dcvc_rt.src.models.video_model import DMC
from dcvc_rt.src.utils.evaluation_protocol import (ALL_FRAMES_PROTOCOL, dataset_summary,
                                                   evaluation_id, state_dict_sha256)
from dcvc_rt.src.utils.transforms import rgb2ycbcr, ycbcr2rgb
from dcvc_rt.src.utils.vcm_bitstream import VCMSequenceReader
from dcvc_rt.src.utils.vcm_eval_dataset import AnnotatedVideoDataset
from evaluate_vcm import (QP_OFFSETS, aggregate_rate, encode_sequence, load_codec_checkpoint,
                          safe_name, sequence_rate_record, use_two_entropy_coders)
from multitask_exp.mask_map import MaskMAP

MIN_RATE_POINT_COUNT = 4


def to_uint8_rgb(frame: torch.Tensor) -> np.ndarray:
    """[1,3,H,W] or [3,H,W] float in 0..1 -> HWC uint8 RGB, as YOLO expects."""
    if frame.dim() == 4:
        frame = frame[0]
    return (frame.detach().permute(1, 2, 0).cpu().numpy().clip(0, 1) * 255).astype(np.uint8)


class SegPredictor:
    """Frozen yolov5-seg run manually: AutoShape does not handle segmentation heads."""

    def __init__(self, weights, device, size, confidence, nms_iou, max_detections):
        from models.experimental import attempt_load

        self.model = attempt_load(weights, device=device, fuse=True).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.device = device
        self.size = int(size)
        self.confidence = float(confidence)
        self.nms_iou = float(nms_iou)
        self.max_detections = int(max_detections)
        self.stride = int(self.model.stride.max())
        self.names = self.model.names

    @torch.inference_mode()
    def __call__(self, image: np.ndarray):
        """HWC uint8 RGB -> (masks [n,h,w] bool, scores [n], classes [n]) in letterbox space."""
        from utils.augmentations import letterbox
        from utils.general import non_max_suppression
        from utils.segment.general import process_mask

        padded, _, _ = letterbox(image, new_shape=self.size, stride=self.stride, auto=False)
        tensor = torch.from_numpy(np.ascontiguousarray(padded)).to(self.device)
        tensor = tensor.permute(2, 0, 1).float().div_(255.0).unsqueeze(0)
        prediction, proto = self.model(tensor)[:2]
        detections = non_max_suppression(
            prediction, self.confidence, self.nms_iou, nm=32,
            max_det=self.max_detections)[0]
        if not len(detections):
            empty = torch.zeros((0, tensor.shape[2], tensor.shape[3]), dtype=torch.bool,
                                device=self.device)
            return empty, detections[:, 4], detections[:, 5].long()
        masks = process_mask(proto[0], detections[:, 6:], detections[:, :4],
                             tensor.shape[2:], upsample=True)
        return masks.bool(), detections[:, 4], detections[:, 5].long()


@torch.inference_mode()
def decode_and_evaluate_masks(image_model, model, predictor, evaluator, dataset, sequence,
                              bitstream_path, device, first_image_id, sequence_evaluator=None):
    """Decode a sequence and score its masks against the uncompressed-frame reference."""
    model.clear_dpb()
    model.set_curr_poc(0)
    with VCMSequenceReader(bitstream_path) as reader:
        header = reader.header
        if (header.width, header.height) != (sequence.width, sequence.height):
            raise ValueError(f"Bitstream resolution mismatch for {sequence.name}")
        if header.coded_frames != sequence.frame_count:
            raise ValueError(f"Bitstream frame-count mismatch for {sequence.name}")
        sps = {"height": sequence.height, "width": sequence.width,
               "ec_part": int(header.two_entropy_coders), "use_ada_i": 0}
        image_id = first_image_id
        for frame_index, packet in enumerate(reader.frames()):
            if frame_index == 0:
                decoded = image_model.decompress(packet.bitstream, sps, packet.qp)
                model.add_ref_frame(feature=None, frame=decoded["x_hat"])
            else:
                if header.reset_interval > 0 and frame_index % header.reset_interval == 1:
                    model.reset_ref_feature()
                decoded = model.decompress(packet.bitstream, sps, packet.qp)
            reconstructed = ycbcr2rgb(
                decoded["x_hat"][:, :, : sequence.height, : sequence.width])

            source = dataset.load_frame(sequence.frame_paths[frame_index])
            target_masks, _, target_classes = predictor(to_uint8_rgb(source))
            predicted_masks, scores, classes = predictor(to_uint8_rgb(reconstructed))
            for metric in (evaluator, sequence_evaluator):
                if metric is not None:
                    metric.add(image_id=image_id, predicted_masks=predicted_masks,
                               predicted_scores=scores, predicted_classes=classes,
                               target_masks=target_masks, target_classes=target_classes)
            image_id += 1
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return image_id


def evaluate(args):
    if len(args.qps) < MIN_RATE_POINT_COUNT or len(set(args.qps)) != len(args.qps):
        raise ValueError("At least four distinct base QPs are required")
    if not torch.cuda.is_available():
        raise RuntimeError("Actual DMC bitstream coding requires CUDA")

    device = torch.device(f"cuda:{args.cuda_index}")
    dataset = AnnotatedVideoDataset(args.data_dir, args.dataset_manifest)
    sequences = list(dataset)
    if args.max_sequences is not None:
        sequences = sequences[: args.max_sequences]
    if not sequences:
        raise RuntimeError("No evaluation sequences were selected")

    video_paths = [Path(path) for path in args.video_ckpt]
    if len(video_paths) == 1:
        video_paths *= len(args.qps)
    elif len(video_paths) != len(args.qps):
        raise ValueError("--video-ckpt requires one path or one path per QP")

    image_model = DMCI().to(device).eval()
    load_codec_checkpoint(image_model, args.image_ckpt, state_key="dmci_state_dict")
    image_model.update(force_zero_thres=args.force_zero_thres)

    predictor = SegPredictor(args.seg_weights, device, args.detector_size,
                             args.confidence_threshold, args.nms_iou_threshold,
                             args.max_detections)
    seg_config = {
        "task_model": "yolov5s-seg",
        "weights_id": state_dict_sha256(predictor.model.state_dict()),
        "input_size": int(args.detector_size),
        "confidence_threshold": float(args.confidence_threshold),
        "nms_iou_threshold": float(args.nms_iou_threshold),
        "max_detections": int(args.max_detections),
        "class_count": len(predictor.names),
        "mask_space": "detector letterbox input, identical transform for both sides",
    }

    method_name = safe_name(args.method_name)
    bitstream_root = Path(args.bitstream_dir) / method_name
    points = []
    for base_qp, video_path in zip(args.qps, video_paths):
        model = DMC().to(device).eval()
        checkpoint = load_codec_checkpoint(model, video_path)
        model.update(force_zero_thres=args.force_zero_thres)
        hierarchical_qp = bool(checkpoint.get("hierarchical_qp", True))

        evaluator = MaskMAP()
        sequence_records = []
        next_image_id = 0
        for sequence in sequences:
            bitstream_path = bitstream_root / f"qp_{base_qp:02d}" / f"{safe_name(sequence.name)}.vcm"
            bitstream_path.parent.mkdir(parents=True, exist_ok=True)
            estimated_bits = encode_sequence(image_model, model, dataset, sequence, base_qp,
                                             bitstream_path, device, args.reset_interval,
                                             hierarchical_qp)
            sequence_evaluator = MaskMAP()
            next_image_id = decode_and_evaluate_masks(
                image_model, model, predictor, evaluator, dataset, sequence, bitstream_path,
                device, next_image_id, sequence_evaluator)
            record = {**sequence_rate_record(sequence, bitstream_path, estimated_bits),
                      **sequence_evaluator.compute()}
            sequence_records.append(record)
            print(f"  qp {base_qp:2d} {sequence.name:<20} bpp {record['actual_bpp']:.5f}  "
                  f"mask map50 {record['map50']:.4f}  map5095 {record['map5095']:.4f}",
                  flush=True)
            if not args.keep_bitstreams:
                bitstream_path.unlink()

        points.append({
            "base_qp": base_qp,
            "video_checkpoint": str(video_path.resolve()),
            "hierarchical_qp": hierarchical_qp,
            **aggregate_rate(sequence_records),
            **evaluator.compute(),
            "sequences": sequence_records,
        })
        print(f"qp {base_qp:2d} TOTAL  bpp {points[-1]['actual_bpp']:.5f}  "
              f"mask map50 {points[-1]['map50']:.4f}", flush=True)
        del model, checkpoint

    output = {
        "schema_version": 7,
        "method": args.method_name,
        "codec": "DCVC-RT DMCI + DMC all-frame VCM",
        "codec_config": {
            "image_checkpoint": str(Path(args.image_ckpt).resolve()),
            "checkpoints": [str(path.resolve()) for path in video_paths],
            "base_qps": list(args.qps),
            "qp_offsets": list(QP_OFFSETS if points[0]["hierarchical_qp"] else (0,) * 8),
            "reset_interval": args.reset_interval,
            "color_pipeline": "RGB -> full-range BT.709 YCbCr444 -> codec -> RGB",
        },
        "protocol": ALL_FRAMES_PROTOCOL,
        "comparison_scope": "end-to-end VCM system, segmentation axis",
        "rate_source": "actual sequence-container bytes including headers",
        "rate_points": len(points),
        "task": "instance_segmentation",
        "task_model": "yolov5s-seg",
        "ground_truth": ("instance masks predicted by the frozen segmentation model on the "
                         "uncompressed source frames (proxy reference, not human labels)"),
        "evaluation_id": evaluation_id(dataset, sequences),
        "dataset": dataset_summary(dataset, sequences),
        "detector_config": seg_config,
        "machine_frontend": {"type": "frozen_pretrained_yolov5_seg",
                             "task_backend_weights_id": seg_config["weights_id"]},
        "points": points,
    }
    output_path = Path(args.output_dir) / f"{method_name}_mask_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Saved mask-axis results to {output_path}")


def self_check(args):
    """CPU-only: the metric, and the predictor's shapes and determinism. No codec."""
    from multitask_exp.mask_map import self_check as metric_self_check

    metric_self_check()

    device = torch.device(args.device)
    predictor = SegPredictor(args.seg_weights, device, args.detector_size,
                             args.confidence_threshold, args.nms_iou_threshold,
                             args.max_detections)
    generator = torch.Generator().manual_seed(5)
    frame = to_uint8_rgb(torch.rand(3, 240, 416, generator=generator))
    masks, scores, classes = predictor(frame)
    assert masks.dtype == torch.bool and masks.dim() == 3, masks.shape
    assert masks.shape[0] == len(scores) == len(classes), (masks.shape, scores.shape)
    again = predictor(frame)
    assert torch.equal(masks, again[0]) and torch.equal(scores, again[1]), "predictor is not deterministic"

    # A frame scored against itself must be a perfect 1.0: this is the zero-distortion
    # end of the axis, so any value below 1.0 means the harness itself loses masks.
    # Random noise contains no objects, so pass --sample-frame <a real frame> to make
    # this case bite; on Kaggle that is one of the SFU frames being evaluated.
    if args.sample_frame:
        photo = to_uint8_rgb(AnnotatedVideoDataset.load_frame(Path(args.sample_frame)))
    else:
        photo = to_uint8_rgb(torch.rand(3, 240, 416, generator=generator).mul(0.4).add(0.3))
    masks, scores, classes = predictor(photo)
    if args.sample_frame and not len(masks):
        raise AssertionError(f"No instance found on {args.sample_frame}; the reference "
                             "would be empty, so the mask axis cannot be measured there")
    if len(masks):
        metric = MaskMAP()
        metric.add(image_id=0, predicted_masks=masks, predicted_scores=scores,
                   predicted_classes=classes, target_masks=masks, target_classes=classes)
        result = metric.compute()
        assert abs(result["map50"] - 1.0) < 1e-9, result
        identity = f"identical frame scores map50 {result['map50']:.4f}"
    else:
        identity = "no instance found on the synthetic frame, identity case not exercised"

    print(f"evaluate_mask_proxy self-check passed: predictor deterministic, "
          f"{len(scores)} instances on a synthetic frame, {identity}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Segmentation-axis (mask mAP) evaluation against an uncompressed-frame reference")
    parser.add_argument("--data-dir")
    parser.add_argument("--dataset-manifest")
    parser.add_argument("--image-ckpt", default="./checkpoints/dcvc_rt/cvpr2025_image.pth.tar")
    parser.add_argument("--video-ckpt", nargs="+", default=[])
    parser.add_argument("--qps", type=int, nargs="+", default=[])
    parser.add_argument("--seg-weights", default="multitask_exp/weights/yolov5s-seg.pt")
    parser.add_argument("--detector-size", type=int, default=640)
    parser.add_argument("--confidence-threshold", type=float, default=0.25)
    parser.add_argument("--nms-iou-threshold", type=float, default=0.45)
    parser.add_argument("--max-detections", type=int, default=300)
    parser.add_argument("--reset-interval", type=int, default=32)
    parser.add_argument("--force-zero-thres", type=float, default=None)
    parser.add_argument("--max-sequences", type=int, default=None)
    parser.add_argument("--keep-bitstreams", action="store_true",
                        help="Keep the .vcm files (they are deleted after scoring by default)")
    parser.add_argument("--bitstream-dir", default="multitask_exp/output/mask_bitstreams")
    parser.add_argument("--output-dir", default="multitask_exp/output/mask_results")
    parser.add_argument("--method-name", default="proposed")
    parser.add_argument("--cuda-index", type=int, default=0)
    parser.add_argument("--device", default="cpu", help="self-check only")
    parser.add_argument("--sample-frame", help="self-check only: a real frame to exercise "
                                               "the identity case on")
    parser.add_argument("--self_check", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.self_check:
        self_check(arguments)
    else:
        evaluate(arguments)
