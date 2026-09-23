"""How much object signal does the segmentation teacher actually see during training?

The task losses are self-supervised -- no labels -- so any video can be used, and the
project trains on Vimeo-90K Septuplet crops. But what the codec learns to preserve is
decided by what the frozen teacher reacts to on THOSE crops. If a 256x256 Vimeo crop
contains almost no instances the teacher is confident about, the layer-17 feature
difference is dominated by background texture, and the segmentation loss spends its
gradient on something other than objects. That is a different failure from noise, it is
cheap to check, and it decides whether a different training set is needed.

This counts, over random training crops, how many instances each frozen teacher finds,
and how often a crop contains none at all. Point it at the evaluation sequences too
(--manifest) to see the same numbers on the domain the thesis reports on.
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from multitask_exp.evaluate_mask_proxy import SegPredictor, to_uint8_rgb


def summarise(name, counts, scores):
    counts = np.asarray(counts, dtype=np.float64)
    empty = float((counts == 0).mean() * 100.0)
    best = float(np.mean(scores)) if scores else float("nan")
    print(f"{name:>28}  anh {len(counts):>4}  trung binh {counts.mean():>5.2f} instance  "
          f"khong co gi {empty:>5.1f} %  do tin cay cao nhat {best:.3f}")
    return {"images": int(len(counts)), "mean_instances": float(counts.mean()),
            "empty_percent": empty, "mean_top_score": best,
            "max_instances": int(counts.max()) if len(counts) else 0}


@torch.inference_mode()
def measure(predictor, frames, loader, crop, generator):
    counts, scores = [], []
    for path in frames:
        image = loader(path)
        if crop and image.shape[-2] >= crop and image.shape[-1] >= crop:
            top = int(torch.randint(0, image.shape[-2] - crop + 1, (1,), generator=generator))
            left = int(torch.randint(0, image.shape[-1] - crop + 1, (1,), generator=generator))
            image = image[:, top:top + crop, left:left + crop]
        masks, confidences, _ = predictor(to_uint8_rgb(image))
        counts.append(len(masks))
        if len(confidences):
            scores.append(float(confidences.max()))
    return counts, scores


def main(args):
    from PIL import Image
    from torchvision.transforms import functional as transforms

    device = torch.device(args.device)
    generator = torch.Generator().manual_seed(args.seed)
    random.seed(args.seed)

    def load(path):
        with Image.open(path) as image:
            return transforms.to_tensor(image.convert("RGB"))

    seg = SegPredictor(args.seg_weights, device, args.detector_size, args.confidence_threshold,
                       args.nms_iou_threshold, args.max_detections)
    det = SegPredictor(args.det_weights, device, args.detector_size, args.confidence_threshold,
                       args.nms_iou_threshold, args.max_detections) if args.det_weights else None

    report = {}
    if args.dataset:
        root = Path(args.dataset)
        clips = [line.strip() for line in (root / args.list_name).read_text().splitlines()
                 if line.strip()]
        random.shuffle(clips)
        frames = [root / "sequences" / clip / "im1.png" for clip in clips[: args.samples]]
        frames = [path for path in frames if path.is_file()]
        print(f"Vimeo: {len(frames)} clip, crop {args.crop}x{args.crop} nhu luc huan luyen")
        report["vimeo_seg"] = summarise("Vimeo · teacher seg", *measure(
            seg, frames, load, args.crop, generator))
        if det is not None:
            report["vimeo_det"] = summarise("Vimeo · teacher det", *measure(
                det, frames, load, args.crop, generator))

    if args.manifest:
        manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
        data_dir = Path(args.data_dir or Path(args.manifest).parent)
        frames = []
        for entry in manifest["sequences"]:
            paths = [data_dir / name for name in entry["frames"]] if "frames" in entry else []
            frames.extend(paths[:: max(1, len(paths) // max(1, args.samples // len(manifest["sequences"])))])
        frames = [path for path in frames if path.is_file()][: args.samples]
        print(f"\nBo danh gia: {len(frames)} khung, KHONG cat (dung kich thuoc that)")
        if frames:
            report["eval_seg"] = summarise("Danh gia · teacher seg", *measure(
                seg, frames, load, 0, generator))

    if report:
        print("\nDoc ket qua: neu crop Vimeo gan nhu khong co instance nao, loss segmentation "
              "khong day codec ve doi tuong ma ve ket cau nen -- doi/loc du lieu huan luyen "
              "moi la cach sua, khong phai them seed.")
    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print("saved", args.output)


def self_check(args):
    """The counter must return one number per image and never invent instances."""
    device = torch.device(args.device)
    predictor = SegPredictor(args.seg_weights, device, args.detector_size,
                             args.confidence_threshold, args.nms_iou_threshold,
                             args.max_detections)
    generator = torch.Generator().manual_seed(0)
    images = [torch.rand(3, 300, 400) for _ in range(3)]
    counts, scores = measure(predictor, images, lambda image: image, 256, generator)
    assert len(counts) == len(images), counts
    assert all(isinstance(value, int) and value >= 0 for value in counts), counts
    assert len(scores) <= len(images), scores
    assert sum(counts) == 0 or scores, "instances found but no confidence recorded"
    print(f"count_teacher_instances self-check passed: {len(counts)} anh nhieu ngau nhien "
          f"-> {counts} instance (ky vong 0 tren nhieu)")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", help="Vimeo-90K Septuplet root")
    parser.add_argument("--list_name", default="sep_trainlist.txt")
    parser.add_argument("--manifest", help="evaluation manifest, to compare domains")
    parser.add_argument("--data-dir", help="root the manifest paths are relative to")
    parser.add_argument("--seg-weights", default="multitask_exp/weights/yolov5s-seg.pt")
    parser.add_argument("--det-weights", default="", help="optional, e.g. yolov5s.pt")
    parser.add_argument("--samples", type=int, default=300)
    parser.add_argument("--crop", type=int, default=256)
    parser.add_argument("--detector-size", type=int, default=640)
    parser.add_argument("--confidence-threshold", type=float, default=0.25)
    parser.add_argument("--nms-iou-threshold", type=float, default=0.45)
    parser.add_argument("--max-detections", type=int, default=300)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output")
    parser.add_argument("--self_check", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.self_check:
        self_check(arguments)
    else:
        main(arguments)
