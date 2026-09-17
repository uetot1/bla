"""Check that a YOLOv5 segmentation teacher can join the detection teacher.

Experimental (multitask_exp): imports the existing training code read-only.
Run from the repository root:
    python -m multitask_exp.check_task_teachers

Runs the exact training code path (make_yolo_teacher_and_clone and
extract_teacher_feature) for both weights and reports:
  1. model type and depth,
  2. layer-4 feature shapes on a training-size crop,
  3. how different the frontend (layers 0-4) weights are,
  4. that gradients reach the cloned segmentation frontend,
  5. how different the two teachers' features are on real image crops,
     including linear CKA, which is blind to channel order (cosine is not).
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from svc_machine.feature_extractor import (
    FRONTEND_LAST_LAYER,
    extract_teacher_feature,
    make_yolo_teacher_and_clone,
)


DEFAULT_IMAGES = (
    "rivf_paper/figures/qualitative_original.png",
)


def sha256(path):
    import hashlib

    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def frontend_weight_difference(det_teacher, seg_teacher):
    report = []
    for index in range(FRONTEND_LAST_LAYER + 1):
        det_state = det_teacher.model[index].state_dict()
        seg_state = seg_teacher.model[index].state_dict()
        if det_state.keys() != seg_state.keys():
            raise ValueError(f"Layer {index} has different parameter names")
        numerator = 0.0
        denominator = 0.0
        identical = True
        for key, det_value in det_state.items():
            seg_value = seg_state[key]
            if det_value.shape != seg_value.shape:
                raise ValueError(f"Layer {index} {key}: {det_value.shape} vs {seg_value.shape}")
            if not det_value.is_floating_point():
                continue
            numerator += (det_value - seg_value).float().pow(2).sum().item()
            denominator += det_value.float().pow(2).sum().item()
            identical = identical and torch.equal(det_value, seg_value)
        report.append({
            "layer": index,
            "type": type(det_teacher.model[index]).__name__,
            "identical": identical,
            "relative_l2_difference": (numerator / max(denominator, 1e-12)) ** 0.5,
        })
    return report


def random_crops(paths, crop, count, seed):
    rng = random.Random(seed)
    crops = []
    for path in paths:
        image = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
        height, width = image.shape[:2]
        if height < crop or width < crop:
            raise ValueError(f"{path} is smaller than {crop}x{crop}")
        for _ in range(count):
            top = rng.randint(0, height - crop)
            left = rng.randint(0, width - crop)
            patch = image[top:top + crop, left:left + crop]
            crops.append(torch.from_numpy(patch).permute(2, 0, 1))
    return torch.stack(crops)


def linear_cka(features_a, features_b):
    # Rows are samples. Invariant to channel permutation, rotation and isotropic scale.
    a = features_a.double() - features_a.double().mean(0, keepdim=True)
    b = features_b.double() - features_b.double().mean(0, keepdim=True)
    cross = (b.T @ a).pow(2).sum()
    return (cross / ((a.T @ a).pow(2).sum().sqrt() * (b.T @ b).pow(2).sum().sqrt())).item()


def spatial_samples(feature):
    # [B, C, H, W] -> [B*H*W, C]: every spatial position is one sample.
    return feature.permute(0, 2, 3, 1).reshape(-1, feature.shape[1])


@torch.no_grad()
def layer_output(teacher, batch, layer):
    return teacher(batch, cut_model=1, cutting_layer=layer)


@torch.no_grad()
def cka_by_layer(det_teacher, seg_teacher, batch, layers):
    blurred = torch.nn.functional.avg_pool2d(batch, 3, stride=1, padding=1)
    rows = []
    for layer in layers:
        det_map = layer_output(det_teacher, batch, layer)
        seg_map = layer_output(seg_teacher, batch, layer)
        if det_map.shape != seg_map.shape:
            raise ValueError(f"Layer {layer}: {tuple(det_map.shape)} vs {tuple(seg_map.shape)}")
        rows.append({
            "layer": layer,
            "module": type(det_teacher.model[layer]).__name__,
            "shape": list(det_map.shape[1:]),
            "cka_det_vs_seg": linear_cka(spatial_samples(det_map), spatial_samples(seg_map)),
            "cka_det_vs_det_blurred": linear_cka(
                spatial_samples(det_map),
                spatial_samples(layer_output(det_teacher, blurred, layer)),
            ),
        })
    return rows


@torch.no_grad()
def teacher_feature_difference(det_teacher, seg_teacher, batch):
    det_map = extract_teacher_feature(det_teacher, batch)
    seg_map = extract_teacher_feature(seg_teacher, batch)
    cka = {
        "det_vs_seg": linear_cka(spatial_samples(det_map), spatial_samples(seg_map)),
        # Reference point: the same det teacher on a lightly blurred copy of the
        # crops, i.e. a pair of representations that should stay very similar.
        "det_vs_det_blurred": linear_cka(
            spatial_samples(det_map),
            spatial_samples(extract_teacher_feature(
                det_teacher,
                torch.nn.functional.avg_pool2d(batch, 3, stride=1, padding=1),
            )),
        ),
        "samples": int(det_map.shape[0] * det_map.shape[2] * det_map.shape[3]),
    }
    det_feature = det_map.flatten(1)
    seg_feature = seg_map.flatten(1)
    relative_mse = (
        (det_feature - seg_feature).pow(2).sum(1) / det_feature.pow(2).sum(1).clamp_min(1e-12)
    )
    cosine = torch.nn.functional.cosine_similarity(det_feature, seg_feature, dim=1)
    return {
        "linear_cka": cka,
        "crops": int(batch.shape[0]),
        "relative_mse_mean": relative_mse.mean().item(),
        "relative_mse_min": relative_mse.min().item(),
        "cosine_mean": cosine.mean().item(),
        "cosine_min": cosine.min().item(),
        "cosine_max": cosine.max().item(),
        "det_feature_mean_square": det_feature.pow(2).mean().item(),
        "seg_feature_mean_square": seg_feature.pow(2).mean().item(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--det-weights", default="yolov5s.pt")
    parser.add_argument("--seg-weights", default="multitask_exp/weights/yolov5s-seg.pt")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--crop", type=int, default=256)
    parser.add_argument("--images", nargs="*", default=list(DEFAULT_IMAGES))
    parser.add_argument("--crops-per-image", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cka-layers", default="4,6,9,13,17,20,23",
                        help="Comma-separated layer indices for the per-layer CKA sweep")
    parser.add_argument("--json-out",
                        default="multitask_exp/output/teacher_check/teacher_check.json")
    args = parser.parse_args()

    device = torch.device(args.device)
    det_teacher, _ = make_yolo_teacher_and_clone(args.det_weights, device)
    seg_teacher, seg_clone = make_yolo_teacher_and_clone(args.seg_weights, device)

    probe = torch.zeros(2, 3, args.crop, args.crop, device=device)
    det_shape = tuple(extract_teacher_feature(det_teacher, probe).shape)
    seg_shape = tuple(extract_teacher_feature(seg_teacher, probe).shape)

    seg_clone.eval()
    clone_input = torch.rand(2, 3, args.crop, args.crop, device=device)
    with torch.no_grad():
        clone_matches_teacher = torch.allclose(
            seg_clone(clone_input),
            extract_teacher_feature(seg_teacher, clone_input),
            atol=1e-5,
        )
    seg_clone.train()
    seg_clone(clone_input).pow(2).mean().backward()
    gradient_norms = [
        parameter.grad.norm().item()
        for parameter in seg_clone.parameters()
        if parameter.grad is not None
    ]

    report = {
        "weights": {
            "det": {"path": args.det_weights, "sha256": sha256(args.det_weights)},
            "seg": {"path": args.seg_weights, "sha256": sha256(args.seg_weights)},
        },
        "device": str(device),
        "model_type": {
            "det": type(det_teacher).__name__,
            "seg": type(seg_teacher).__name__,
        },
        "model_depth": {"det": len(det_teacher.model), "seg": len(seg_teacher.model)},
        "feature_shape": {"det": det_shape, "seg": seg_shape},
        "shapes_match": det_shape == seg_shape,
        "seg_clone_matches_teacher_at_init": clone_matches_teacher,
        "seg_clone_gradients": {
            "parameters_with_grad": len(gradient_norms),
            "parameters_total": sum(1 for _ in seg_clone.parameters()),
            "all_finite_and_nonzero": bool(gradient_norms) and all(
                np.isfinite(norm) and norm > 0 for norm in gradient_norms
            ),
        },
        "frontend_weight_difference": frontend_weight_difference(det_teacher, seg_teacher),
    }

    if report["shapes_match"] and args.images:
        batch = random_crops(args.images, args.crop, args.crops_per_image, args.seed).to(device)
        report["feature_difference"] = teacher_feature_difference(det_teacher, seg_teacher, batch)
        report["feature_difference"]["images"] = list(args.images)
        layers = [int(value) for value in args.cka_layers.split(",") if value.strip()]
        report["cka_by_layer"] = cka_by_layer(det_teacher, seg_teacher, batch, layers)

    output_path = Path(args.json_out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
