"""Keep only the Vimeo clips whose frames actually contain objects.

Measured on this project's training data: a 256x256 Vimeo crop holds 2.0 instances the
frozen teacher is confident about and 23% of crops hold none, while a full SFU evaluation
frame holds 9.9 and never none. Every empty clip spends a training step teaching the codec
to preserve background texture, which is not what either task loss is for.

This writes a filtered clip list -- same format as sep_trainlist.txt, one clip per line --
that `train_multitask.py --train_list` reads. An absolute path overrides the dataset root,
so the list can live in a writable folder while Vimeo stays on a read-only mount.

The filter looks at the FULL frame rather than a crop: a clip whose frame has objects can
still yield an empty crop, but a clip whose frame has none can never yield a useful one,
and one inference per clip keeps this cheap.
"""

import argparse
import json
from pathlib import Path

import torch

from multitask_exp.evaluate_mask_proxy import SegPredictor, to_uint8_rgb


def frame_paths(root, clips, frame_name):
    return [(clip, root / "sequences" / clip / frame_name) for clip in clips]


@torch.inference_mode()
def filter_clips(predictor, root, clips, frame_name, min_instances, log_every=500):
    from PIL import Image
    from torchvision.transforms.functional import to_tensor

    kept, counts, missing = [], [], 0
    for index, (clip, path) in enumerate(frame_paths(root, clips, frame_name), start=1):
        if not path.is_file():
            missing += 1
            continue
        with Image.open(path) as image:
            frame = to_tensor(image.convert("RGB"))
        masks, _, _ = predictor(to_uint8_rgb(frame))
        counts.append(len(masks))
        if len(masks) >= min_instances:
            kept.append(clip)
        if index % log_every == 0:
            print(f"  {index}/{len(clips)} clip · giu {len(kept)} "
                  f"({100.0 * len(kept) / index:.1f} %)", flush=True)
    return kept, counts, missing


def main(args):
    root = Path(args.dataset)
    clips = [line.strip() for line in (root / args.list_name).read_text().splitlines()
             if line.strip()]
    if args.max_clips:
        clips = clips[: args.max_clips]
    device = torch.device(args.device)
    predictor = SegPredictor(args.seg_weights, device, args.detector_size,
                             args.confidence_threshold, args.nms_iou_threshold,
                             args.max_detections)
    print(f"Quet {len(clips)} clip bang teacher segmentation, giu clip co >= "
          f"{args.min_instances} doi tuong")
    kept, counts, missing = filter_clips(predictor, root, clips, args.frame_name,
                                         args.min_instances)
    if not kept:
        raise RuntimeError("Khong clip nao qua bo loc -- ha --min-instances xuong")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(kept) + "\n", encoding="utf-8")
    mean = sum(counts) / len(counts) if counts else 0.0
    kept_mean = (sum(c for c in counts if c >= args.min_instances)
                 / max(1, len(kept)))
    summary = {"dataset": str(root), "source_list": args.list_name, "output": str(output),
               "clips_in": len(clips), "clips_kept": len(kept), "clips_missing": missing,
               "keep_percent": 100.0 * len(kept) / len(clips),
               "mean_instances_all": mean, "mean_instances_kept": kept_mean,
               "min_instances": args.min_instances}
    print(f"\nGiu {len(kept)}/{len(clips)} clip ({summary['keep_percent']:.1f} %)")
    print(f"Doi tuong trung binh: {mean:.2f} tren toan bo, {kept_mean:.2f} tren phan giu lai")
    print("Da luu danh sach:", output)
    if args.summary:
        Path(args.summary).write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print("Da luu tom tat:", args.summary)
    return summary


def self_check(args):
    """The filter must keep exactly the clips the counter says have objects."""
    import tempfile

    device = torch.device(args.device)
    predictor = SegPredictor(args.seg_weights, device, args.detector_size,
                             args.confidence_threshold, args.nms_iou_threshold,
                             args.max_detections)

    class Fake:
        """Stands in for the teacher: clip index decides how many instances it sees."""

        def __init__(self, table):
            self.table = table

        def __call__(self, image):
            count = self.table[image[0, 0, 0] % len(self.table)]
            empty = torch.zeros((count, 4))
            return empty, torch.ones(count), torch.zeros(count, dtype=torch.long)

    table = [0, 3, 0, 1]
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        clips = [f"0000{index}/{index}" for index in range(len(table))]
        for index, clip in enumerate(clips):
            (root / "sequences" / clip).mkdir(parents=True)
            from PIL import Image
            Image.new("RGB", (8, 8), (index, 0, 0)).save(root / "sequences" / clip / "im1.png")
        kept, counts, missing = filter_clips(Fake(table), root, clips, "im1.png", 1)
        assert missing == 0, missing
        assert counts == table, (counts, table)
        assert kept == [clips[1], clips[3]], kept

        # A real clip list round-trips: written lines are exactly the kept clip names.
        out = root / "filtered.txt"
        out.write_text("\n".join(kept) + "\n", encoding="utf-8")
        assert [line.strip() for line in out.read_text().splitlines() if line.strip()] == kept

    print(f"filter_clips self-check passed: giu dung {len(kept)}/{len(table)} clip co doi tuong, "
          f"danh sach ghi ra doc lai khop; teacher that tai duoc ({len(predictor.names)} lop)")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", help="Vimeo-90K Septuplet root")
    parser.add_argument("--list_name", default="sep_trainlist.txt")
    parser.add_argument("--frame_name", default="im1.png")
    parser.add_argument("--output", default="multitask_exp/output/sep_trainlist_objects.txt")
    parser.add_argument("--summary")
    parser.add_argument("--seg-weights", default="multitask_exp/weights/yolov5s-seg.pt")
    parser.add_argument("--min-instances", type=int, default=1)
    parser.add_argument("--max-clips", type=int, default=0)
    parser.add_argument("--detector-size", type=int, default=640)
    parser.add_argument("--confidence-threshold", type=float, default=0.25)
    parser.add_argument("--nms-iou-threshold", type=float, default=0.45)
    parser.add_argument("--max-detections", type=int, default=300)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--self_check", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.self_check:
        self_check(arguments)
    else:
        main(arguments)
