"""Measure seg_scale and per-layer teacher CKA on Vimeo-90K validation clips.

Experimental (multitask_exp). Run from the repo root (on Kaggle, with Vimeo-90K):
    python -m multitask_exp.measure_task_scales --dataset /path/vimeo_septuplet \
        [--init_checkpoint paper_best.pth]
Local code-path test without data or DCVC-RT weights:
    python -m multitask_exp.measure_task_scales --synthetic --clips 4

seg_scale = geometric mean over base QPs of mean(D_det) / mean(D_seg), measured
with the codec that training will start from, so both loss terms start at the
same magnitude. The detection clone and all YOLO models run in eval mode here.
"""

import argparse
import json
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dcvc_rt.src.models.image_model import DMCI
from dcvc_rt.src.models.video_model import DMC
from dcvc_rt.src.utils.common import get_state_dict
from dcvc_rt.src.utils.transforms import rgb2ycbcr, ycbcr2rgb
from svc_machine.feature_extractor import extract_teacher_feature, make_yolo_teacher_and_clone
from svc_machine.feature_loss import feature_mse_loss
from train_base import INDEX_MAP, VimeoSeptuplet

from multitask_exp.check_task_teachers import cka_by_layer
from multitask_exp.checkpoints import warm_start
from multitask_exp.frozen_feature import FrozenYoloFeature


def load_models(args, device):
    image_model = DMCI().to(device).eval()
    video_model = DMC().to(device).eval()
    if not args.synthetic:
        image_model.load_state_dict(get_state_dict(args.model_path_i))
        video_model.load_state_dict(get_state_dict(args.model_path_p))
    det_teacher, det_clone = make_yolo_teacher_and_clone(args.det_weights, device)
    clone_source = "pretrained_yolov5s"
    if args.init_checkpoint:
        _, clone_source = warm_start(video_model, det_clone, args.init_checkpoint)
    det_clone.eval()
    seg_branch = FrozenYoloFeature(args.seg_weights, args.seg_layer, device)
    return image_model, video_model, det_teacher, det_clone, seg_branch, clone_source


def clip_batches(args):
    if args.synthetic:
        generator = torch.Generator().manual_seed(0)
        for start in range(0, args.clips, args.batch_size):
            size = min(args.batch_size, args.clips - start)
            yield torch.rand(size, args.group_size + 1, 3, 256, 256, generator=generator)
        return
    dataset = VimeoSeptuplet(args.dataset, 256, args.group_size,
                             list_name=args.validation_list, random_crop=False)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, drop_last=False)
    seen = 0
    for batch in loader:
        batch = batch[: args.clips - seen]
        yield batch
        seen += batch.shape[0]
        if seen >= args.clips:
            return


@torch.no_grad()
def measure(args):
    device = torch.device(args.device)
    image_model, video_model, det_teacher, det_clone, seg_branch, clone_source = \
        load_models(args, device)
    qps = [int(value) for value in args.qps.split(",")]
    sums = {qp: {"d_det": 0.0, "d_seg": 0.0, "bpp": 0.0, "frames": 0} for qp in qps}
    cka_frames = []
    clips = 0

    for batch in clip_batches(args):
        batch = batch.to(device)
        clips += batch.shape[0]
        if len(cka_frames) * batch.shape[0] < args.cka_clips:
            cka_frames.append(batch[:, 1:].reshape(-1, 3, 256, 256))
        originals = batch[:, 1:]
        det_targets = [extract_teacher_feature(det_teacher, originals[:, i])
                       for i in range(args.group_size)]
        seg_targets = [seg_branch.target(originals[:, i]) for i in range(args.group_size)]
        for qp in qps:
            video_model.clear_dpb()
            video_model.set_curr_poc(0)
            reference = image_model.forward_reconstruction(rgb2ycbcr(batch[:, 0]), qp)
            video_model.add_ref_frame(None, reference)
            for i in range(args.group_size):
                frame_qp = video_model.shift_qp(qp, INDEX_MAP[(i + 1) % 8])
                reconstructed, bpp = video_model.forward_train(rgb2ycbcr(originals[:, i]), frame_qp)
                rgb = ycbcr2rgb(reconstructed)
                sums[qp]["d_det"] += feature_mse_loss(det_clone(rgb), det_targets[i]).item()
                sums[qp]["d_seg"] += feature_mse_loss(seg_branch(rgb), seg_targets[i]).item()
                sums[qp]["bpp"] += float(bpp)
                sums[qp]["frames"] += 1
            video_model.clear_dpb()

    per_qp = []
    log_ratios = []
    for qp in qps:
        frames = sums[qp]["frames"]
        d_det = sums[qp]["d_det"] / frames
        d_seg = sums[qp]["d_seg"] / frames
        ratio = d_det / d_seg
        log_ratios.append(math.log(ratio))
        per_qp.append({"base_qp": qp, "mean_d_det": d_det, "mean_d_seg": d_seg,
                       "d_det_over_d_seg": ratio, "mean_bpp": sums[qp]["bpp"] / frames})
    seg_scale = math.exp(sum(log_ratios) / len(log_ratios))

    cka_batch = torch.cat(cka_frames)[: args.cka_clips * args.group_size]
    layers = [int(value) for value in args.cka_layers.split(",")]
    report = {
        "seg_scale": seg_scale,
        "seg_scale_definition": "geometric mean over base QPs of mean(D_det)/mean(D_seg)",
        "per_qp": per_qp,
        "cka_by_layer": cka_by_layer(det_teacher, seg_branch.model, cka_batch, layers),
        "cka_images": int(cka_batch.shape[0]),
        "clips": clips,
        "synthetic": args.synthetic,
        "seg_layer": args.seg_layer,
        "det_clone_source": clone_source,
        "init_checkpoint": args.init_checkpoint,
        "group_size": args.group_size,
    }
    output = Path(args.json_out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Measure seg_scale and teacher CKA on Vimeo-90K")
    parser.add_argument("--dataset", help="Vimeo-90K Septuplet root")
    parser.add_argument("--validation_list", default="sep_testlist.txt")
    parser.add_argument("--model_path_i", default="./checkpoints/dcvc_rt/cvpr2025_image.pth.tar")
    parser.add_argument("--model_path_p", default="./checkpoints/dcvc_rt/cvpr2025_video.pth.tar")
    parser.add_argument("--init_checkpoint")
    parser.add_argument("--det_weights", default="./yolov5s.pt")
    parser.add_argument("--seg_weights", default="multitask_exp/weights/yolov5s-seg.pt")
    parser.add_argument("--seg_layer", type=int, default=17)
    parser.add_argument("--clips", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--group_size", type=int, choices=(5, 6), default=6)
    parser.add_argument("--qps", default="0,21,42,63")
    parser.add_argument("--cka_layers", default="4,6,9,13,17,20,23")
    parser.add_argument("--cka_clips", type=int, default=16)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--synthetic", action="store_true",
                        help="Random clips and random DMC: tests the code path only")
    parser.add_argument("--json_out", default="multitask_exp/output/task_scales/task_scales.json")
    args = parser.parse_args()
    if not args.synthetic and not args.dataset:
        raise ValueError("--dataset is required unless --synthetic")
    if args.synthetic:
        args.json_out = str(Path(args.json_out).with_name("task_scales_synthetic.json"))
    measure(args)


if __name__ == "__main__":
    main()
