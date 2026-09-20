"""Train DCVC-RT-VCM with detection-clone + frozen-segmentation supervision.

Experimental (multitask_exp): reuses train_base.py read-only. Run from the repo root:
    python -m multitask_exp.train_multitask --self_check
    torchrun --nproc_per_node=2 -m multitask_exp.train_multitask \
        --dataset /path/vimeo_septuplet --init_checkpoint paper_best.pth \
        --alpha_det 0.5 --task_scale_file task_scales.json --run_name R2

alpha_det = 1 -> R0 (detection only), 0 -> R1 (segmentation only), 0.5 -> R2.
Evaluation checkpoints keep train_base.py keys so evaluate_vcm.py loads them
unchanged (always evaluate with --force-pretrained-frontend).
"""

import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from dcvc_rt.src.models.image_model import DMCI
from dcvc_rt.src.models.video_model import DMC
from dcvc_rt.src.utils.common import get_state_dict
from dcvc_rt.src.utils.transforms import rgb2ycbcr
from svc_machine.feature_extractor import (
    FRONTEND_LAST_LAYER,
    extract_teacher_feature,
    make_yolo_teacher_and_clone,
)
from train_base import (
    DISTORTION_WEIGHTS,
    INDEX_MAP,
    QP_OFFSETS,
    VALIDATION_QPS,
    gather_rng_states,
    move_optimizer_state,
    restore_rng_state,
    setup_distributed,
    synchronized_qp,
)

from multitask_exp.checkpoints import warm_start
from multitask_exp.data import VimeoSeptupletFlip
from multitask_exp.frozen_feature import FrozenYoloFeature
from multitask_exp.lambda_schedule import (
    LAMBDA_MAPPING,
    LAMBDA_MAX,
    LAMBDA_MIN,
    lambda_for_qp,
)
from multitask_exp.multitask_system import MultiTaskMachineSystem


SCHEMA_VERSION = "multitask_exp-1"


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_seg_scale(args):
    if args.alpha_det >= 1.0:
        return 1.0
    if args.task_scale_file:
        return float(json.loads(Path(args.task_scale_file).read_text())["seg_scale"])
    if args.seg_scale is None:
        raise ValueError(
            "Segmentation is active: pass --task_scale_file (from measure_task_scales) "
            "or an explicit --seg_scale")
    return float(args.seg_scale)


def run_config(args, seg_scale):
    alpha_seg = 1.0 - args.alpha_det
    det_active = args.alpha_det > 0
    config = {
        "alpha_det": args.alpha_det,
        "alpha_seg": alpha_seg,
        "seg_scale": seg_scale,
        "det_layer": FRONTEND_LAST_LAYER,
        "det_mode": args.det_mode if det_active else None,
        # Clone BatchNorm kept in eval mode, as in the paper's training script. Changes the
        # config so --resume refuses the earlier R0/R2 runs trained with train-mode BN.
        "det_clone_batchnorm": "frozen_eval" if det_active and args.det_mode == "clone" else None,
        "seg_layer": args.seg_layer if alpha_seg > 0 else None,
        "seg_branch": "frozen_pretrained" if alpha_seg > 0 else None,
        "group_size": args.group_size,
        "hflip": args.hflip,
        "lambda_range": [LAMBDA_MIN, LAMBDA_MAX],
        "lambda_mapping": LAMBDA_MAPPING,
        "det_weights_sha256": sha256(args.det_weights) if args.alpha_det > 0 else None,
        "seg_weights_sha256": sha256(args.seg_weights) if alpha_seg > 0 else None,
        "init_checkpoint_sha256": sha256(args.init_checkpoint) if args.init_checkpoint else None,
    }
    return config


def build_system(args, config, device):
    image_model = DMCI().to(device)
    video_model = DMC().to(device)
    if not (args.self_check or args.random_init):
        image_model.load_state_dict(get_state_dict(args.model_path_i))
        video_model.load_state_dict(get_state_dict(args.model_path_p))
    image_model.eval()
    for parameter in image_model.parameters():
        parameter.requires_grad_(False)

    det_teacher = det_clone = det_frozen = None
    if config["alpha_det"] > 0:
        if config["det_mode"] == "frozen":
            # No trainable parameters at all -- structurally identical to the
            # segmentation branch, which does not suffer the clone-drift collapse
            # measured for det_mode="clone" (see MultiTaskMachineSystem docstring).
            det_frozen = FrozenYoloFeature(args.det_weights, FRONTEND_LAST_LAYER, device)
        else:
            det_teacher, det_clone = make_yolo_teacher_and_clone(args.det_weights, device)
    seg_branch = None
    if config["alpha_seg"] > 0:
        seg_branch = FrozenYoloFeature(args.seg_weights, args.seg_layer, device)

    clone_source = "pretrained_yolov5s" if det_clone is not None else None
    if args.init_checkpoint:
        _, clone_source = warm_start(video_model, det_clone, args.init_checkpoint)
        if det_clone is None:
            clone_source = None
    config["det_clone_init"] = clone_source

    system = MultiTaskMachineSystem(
        video_model, det_clone, seg_branch,
        config["alpha_det"], config["alpha_seg"], config["seg_scale"],
        det_frozen=det_frozen).to(device)
    return image_model, det_teacher, system


def forward_group(model, system, image_model, det_teacher, sequences, base_qp,
                  lambda_task, group_size):
    video_model = system.video_model
    video_model.clear_dpb()
    video_model.set_curr_poc(0)
    with torch.no_grad():
        reference = image_model.forward_reconstruction(rgb2ycbcr(sequences[:, 0]), base_qp)
    video_model.add_ref_frame(None, reference)

    rgb_frames = sequences[:, 1:]
    ycbcr_frames = torch.stack(
        [rgb2ycbcr(rgb_frames[:, index]) for index in range(group_size)], dim=1)
    if system.det_clone is not None:
        det_targets = torch.stack(
            [extract_teacher_feature(det_teacher, rgb_frames[:, index]) for index in range(group_size)],
            dim=1)
    elif system.det_frozen is not None:
        det_targets = torch.stack(
            [system.det_frozen.target(rgb_frames[:, index]) for index in range(group_size)], dim=1)
    else:
        det_targets = None
    seg_targets = torch.stack(
        [system.seg_branch.target(rgb_frames[:, index]) for index in range(group_size)],
        dim=1) if system.seg_branch is not None else None
    qps = tuple(
        video_model.shift_qp(base_qp, INDEX_MAP[frame_index % 8])
        for frame_index in range(1, group_size + 1))
    weights = tuple(
        DISTORTION_WEIGHTS[frame_index % len(DISTORTION_WEIGHTS)]
        for frame_index in range(group_size))
    return model(ycbcr_frames, det_targets, seg_targets, qps, lambda_task, weights)


def trainable_components(system):
    names = ["dcvc_rt_dmc"]
    if system.det_clone is not None:
        names.append("yolo_cloned_frontend")
    return names


def frozen_components(system):
    names = ["dmci"]
    if system.det_clone is not None:
        names += ["yolo_teacher", "yolo_backend"]
    elif system.det_frozen is not None:
        names.append("yolov5_detection_pretrained")
    if system.seg_branch is not None:
        names.append("yolov5_seg_pretrained")
    return names


def evaluation_payload(system, config, epoch):
    payload = {
        "schema_version": SCHEMA_VERSION,
        "epoch": epoch,
        "mode": "variable",
        "lambda_task": None,
        "lambda_range": (LAMBDA_MIN, LAMBDA_MAX),
        "lambda_mapping": LAMBDA_MAPPING,
        "qp_sampling": ("uniform", 0, DMCI.get_qp_num() - 1),
        "hierarchical_qp": True,
        "hierarchical_qp_offsets": QP_OFFSETS,
        "hierarchical_distortion_weights": DISTORTION_WEIGHTS,
        "feature_objective": {
            "task_model": "yolov5s",
            "cloned_frontend_last_layer": FRONTEND_LAST_LAYER,
            "layer_indices": (FRONTEND_LAST_LAYER,),
        },
        "multitask": config,
        "state_dict": system.video_model.state_dict(),
        "trainable_components": trainable_components(system),
        "frozen_components": frozen_components(system),
    }
    if system.det_clone is not None:
        payload["cloned_frontend_state_dict"] = system.det_clone.state_dict()
    return payload


def atomic_save(payload, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def reduce_sums(values, device, world_size):
    tensor = torch.tensor(values, device=device, dtype=torch.float64)
    if world_size > 1:
        dist.all_reduce(tensor)
    return tensor.tolist()


def nan_to_zero(value):
    value = float(value)
    return 0.0 if math.isnan(value) else value


@torch.no_grad()
def validate(model, system, image_model, det_teacher, loader, args, device, rank, world_size):
    model.eval()
    sums = [0.0] * 5  # loss, bpp, d_det, d_seg, batches
    for batch_index, sequences in enumerate(tqdm(loader, desc="validation", disable=rank != 0)):
        if args.max_val_batches and batch_index >= args.max_val_batches:
            break
        base_qp = VALIDATION_QPS[batch_index % len(VALIDATION_QPS)]
        sequences = sequences.to(device, non_blocking=True)
        loss, rate, d_det, d_seg = forward_group(
            model, system, image_model, det_teacher, sequences, base_qp,
            lambda_for_qp(base_qp), args.group_size)
        system.video_model.clear_dpb()
        for slot, value in enumerate((loss, rate, d_det, d_seg, 1.0)):
            sums[slot] += nan_to_zero(value)
    model.train()
    loss, rate, d_det, d_seg, batches = reduce_sums(sums, device, world_size)
    if batches == 0:
        raise RuntimeError("No validation batch was produced")
    return {
        "val_total_loss": loss / batches,
        "val_bpp": rate / batches,
        "val_feature_mse_det": d_det / batches if system.det_clone is not None else None,
        "val_feature_mse_seg": d_seg / batches if system.seg_branch is not None else None,
    }


def train_worker(args, device, rank, world_size, local_rank):
    if args.batch_size % world_size:
        raise ValueError("Global batch_size must be divisible by the world size")
    random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    seg_scale = resolve_seg_scale(args)
    config = run_config(args, seg_scale)
    image_model, det_teacher, system = build_system(args, config, device)

    dataset = VimeoSeptupletFlip(
        args.dataset, args.crop_size, args.group_size, hflip=args.hflip)
    validation_dataset = VimeoSeptupletFlip(
        args.dataset, args.crop_size, args.group_size,
        list_name=args.validation_list, random_crop=False, hflip=False)
    sampler = DistributedSampler(dataset, shuffle=True) if world_size > 1 else None
    validation_sampler = DistributedSampler(
        validation_dataset, shuffle=False, drop_last=True) if world_size > 1 else None
    per_rank_batch = args.batch_size // world_size
    loader = DataLoader(dataset, batch_size=per_rank_batch, shuffle=sampler is None,
                        sampler=sampler, num_workers=args.workers,
                        pin_memory=device.type == "cuda", drop_last=True)
    validation_loader = DataLoader(validation_dataset, batch_size=per_rank_batch,
                                   shuffle=False, sampler=validation_sampler,
                                   num_workers=args.workers,
                                   pin_memory=device.type == "cuda", drop_last=True)

    model = DistributedDataParallel(system, device_ids=[local_rank], output_device=local_rank) \
        if world_size > 1 else system
    model.train()
    parameters = tuple(p for p in system.parameters() if p.requires_grad)
    optimizer = torch.optim.Adam(parameters, lr=args.learning_rate)

    save_dir = Path(args.save_dir) / args.run_name
    last_path = save_dir / "last.pth.tar"
    history_path = save_dir / "training_history.json"
    start_epoch, history, best_val_loss = 1, [], math.inf

    if args.resume:
        checkpoint = torch.load(last_path, map_location="cpu", weights_only=False)
        if checkpoint["multitask"] != config:
            raise ValueError(
                "Resume checkpoint was trained with a different multi-task configuration:\n"
                f"checkpoint={checkpoint['multitask']}\nnow={config}")
        system.video_model.load_state_dict(checkpoint["state_dict"])
        if system.det_clone is not None:
            system.det_clone.load_state_dict(checkpoint["cloned_frontend_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        move_optimizer_state(optimizer, device)
        restore_rng_state(checkpoint["rng_states"], rank, device)
        start_epoch = checkpoint["epoch"] + 1
        history = checkpoint["training_history"]
        best_val_loss = min((r["val_total_loss"] for r in history), default=math.inf)
        if rank == 0:
            print(f"Resumed {last_path} from epoch {checkpoint['epoch']}")

    longest_epoch_seconds = args.epoch_minutes_estimate * 60.0
    stopped_by_deadline = False
    for epoch in range(start_epoch, args.epochs + 1):
        if args.deadline_unix:
            needed = 1.1 * longest_epoch_seconds
            stop = torch.tensor(int(time.time() + needed > args.deadline_unix), device=device)
            if world_size > 1:
                dist.broadcast(stop, 0)
            if stop.item():
                stopped_by_deadline = True
                if rank == 0:
                    print(f"Deadline: stopping before epoch {epoch} "
                          f"(needs ~{needed / 60:.0f} min); resume later with --resume")
                break
        epoch_started_wall = time.time()
        if sampler is not None:
            sampler.set_epoch(epoch)
        sums = [0.0] * 6  # loss, bpp, d_det, d_seg, grad_norm, batches
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        progress = tqdm(loader, desc=f"{args.run_name} epoch {epoch}/{args.epochs}",
                        disable=rank != 0)
        for batch_index, sequences in enumerate(progress):
            if args.max_train_batches and batch_index >= args.max_train_batches:
                break
            base_qp = synchronized_qp("variable", 0, device, rank, world_size)
            sequences = sequences.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss, rate, d_det, d_seg = forward_group(
                model, system, image_model, det_teacher, sequences, base_qp,
                lambda_for_qp(base_qp), args.group_size)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                parameters, args.grad_clip, error_if_nonfinite=True)
            optimizer.step()
            system.video_model.clear_dpb()
            for slot, value in enumerate((loss, rate, d_det, d_seg, grad_norm, 1.0)):
                sums[slot] += nan_to_zero(value)
            count = sums[5]
            progress.set_postfix(loss=sums[0] / count, bpp=sums[1] / count,
                                 d_det=sums[2] / count, d_seg=sums[3] / count, qp=base_qp)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started

        loss, rate, d_det, d_seg, grad_norm, batches = reduce_sums(sums, device, world_size)
        if batches == 0:
            raise RuntimeError("No training batch was produced")
        record = {
            "epoch": epoch,
            "total_loss": loss / batches,
            "bpp": rate / batches,
            "feature_mse_det": d_det / batches if system.det_clone is not None else None,
            "feature_mse_seg": d_seg / batches if system.seg_branch is not None else None,
            "grad_norm": grad_norm / batches,
            "train_seconds_rank0": elapsed,
            "seconds_per_global_batch": elapsed / (batches / world_size),
            "global_batches": int(batches / world_size),
            "full_epoch_batches": len(loader),
        }
        record.update(validate(model, system, image_model, det_teacher,
                               validation_loader, args, device, rank, world_size))
        history.append(record)
        improved = record["val_total_loss"] < best_val_loss
        best_val_loss = min(best_val_loss, record["val_total_loss"])

        rng_states = gather_rng_states(device, world_size)
        if rank == 0:
            history_path.parent.mkdir(parents=True, exist_ok=True)
            history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
            evaluation = evaluation_payload(system, config, epoch)
            atomic_save({
                **evaluation,
                "optimizer": optimizer.state_dict(),
                "training_history": history,
                "rng_states": rng_states,
                "world_size": world_size,
            }, last_path)
            atomic_save(evaluation, save_dir / f"epoch_{epoch:04d}.pth")
            if improved:
                atomic_save(evaluation, save_dir / "best_val_loss.pth")
            print(json.dumps(record))
        if world_size > 1:
            dist.barrier()
        longest_epoch_seconds = max(longest_epoch_seconds, time.time() - epoch_started_wall)

    if rank == 0:
        save_dir.mkdir(parents=True, exist_ok=True)
        best = min(history, key=lambda r: r["val_total_loss"]) if history else None
        (save_dir / "status.json").write_text(json.dumps({
            "run_name": args.run_name,
            "completed_epochs": len(history),
            "target_epochs": args.epochs,
            "finished": len(history) >= args.epochs,
            "stopped_by_deadline": stopped_by_deadline,
            "best_epoch_by_val_loss": best["epoch"] if best else None,
            "best_val_total_loss": best["val_total_loss"] if best else None,
            "longest_epoch_minutes": longest_epoch_seconds / 60.0,
            "multitask": config,
        }, indent=2), encoding="utf-8")


def self_check(args):
    """Runs without Vimeo or DCVC-RT weights (randomly initialised DMC)."""
    for qp, expected in ((0, 1.0), (21, 4.0), (42, 16.0), (63, 64.0)):
        assert math.isclose(lambda_for_qp(qp), expected, rel_tol=1e-9), (qp, lambda_for_qp(qp))

    device = torch.device(args.device)
    seg = FrozenYoloFeature(args.seg_weights, args.seg_layer, device)
    seg.train()
    assert not seg.training and not seg.model.training
    assert not any(p.requires_grad for p in seg.parameters())
    image = torch.rand(1, 3, args.crop_size, args.crop_size, device=device, requires_grad=True)
    seg(image).pow(2).mean().backward()
    assert image.grad is not None and image.grad.abs().sum() > 0
    assert all(p.grad is None for p in seg.parameters())

    frames = torch.rand(1, 3, 3, args.crop_size, args.crop_size, device=device)
    results = {}
    # (alpha_det, det_mode): 0.0/clone=R1 (seg only, det_mode irrelevant), 0.5/clone=R2-style
    # (measured to collapse -- kept only so the self-check still covers that code path),
    # 1.0/frozen=R3 (detection only, no trainable params), 0.5/frozen=R4 (both branches frozen).
    for alpha_det, det_mode in ((0.0, "clone"), (0.5, "clone"), (1.0, "frozen"), (0.5, "frozen")):
        args.alpha_det = alpha_det
        args.det_mode = det_mode
        config = run_config(args, 1.0)
        image_model, det_teacher, system = build_system(args, config, device)
        system.train()
        loss, rate, d_det, d_seg = forward_group(
            system, system, image_model, det_teacher, frames, 42, lambda_for_qp(42), 2)
        assert torch.isfinite(loss), loss
        loss.backward()
        dmc_grad = sum(p.grad.abs().sum().item() for p in system.video_model.parameters()
                       if p.grad is not None)
        assert dmc_grad > 0, "codec received no gradient"
        if system.seg_branch is not None:
            assert all(p.grad is None for p in system.seg_branch.parameters()), "seg branch was updated"
        if system.det_clone is not None:
            assert any(p.grad is not None and p.grad.abs().sum() > 0
                       for p in system.det_clone.parameters())
            check_clone_batchnorm_frozen(system, image_model, det_teacher, frames)
        if system.det_frozen is not None:
            assert not any(p.requires_grad for p in system.det_frozen.parameters())
            assert all(p.grad is None for p in system.det_frozen.parameters()), \
                "frozen detection branch was updated"
            sample = frames[:, 0]
            assert torch.allclose(system.det_frozen(sample), system.det_frozen.target(sample),
                                  atol=1e-5), "frozen detection branch is not deterministic"
        results[f"alpha_det={alpha_det},det_mode={det_mode}"] = {
            "loss": loss.item(), "d_det": nan_to_zero(d_det), "d_seg": d_seg.item(),
            "dmc_grad_abs_sum": dmc_grad,
        }

        payload = evaluation_payload(system, config, epoch=1)
        path = Path(args.save_dir) / "_self_check" / f"alpha_det_{alpha_det}.pth"
        atomic_save(payload, path)
        from evaluate_vcm import load_codec_checkpoint
        reloaded = load_codec_checkpoint(DMC(), path)
        assert reloaded["multitask"] == config
        del system, image_model, det_teacher
        if device.type == "cuda":
            torch.cuda.empty_cache()

    check_warm_start_layouts(args, device)
    print(json.dumps(results, indent=2))
    print("multitask_exp self-check passed: lambda 1-64, detection-clone BatchNorm stays in "
          "eval mode through train(), frozen seg branch teaches the codec "
          "without being updated, checkpoints load with evaluate_vcm.load_codec_checkpoint, "
          "warm start accepts train_base and paper (p_net/student_front) layouts")


def check_clone_batchnorm_frozen(system, image_model, det_teacher, frames):
    batchnorms = [m for m in system.det_clone.modules()
                  if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
    assert batchnorms, "detection clone has no BatchNorm to freeze"
    for mode in (True, False, True):
        system.train(mode)
        assert all(not bn.training for bn in batchnorms), f"BatchNorm left training after train({mode})"
    assert all(not p.requires_grad for bn in batchnorms for p in bn.parameters())
    assert all(bn.weight.grad is None for bn in batchnorms), "BatchNorm affine was updated"
    before = [bn.running_mean.clone() for bn in batchnorms]
    with torch.no_grad():
        forward_group(system, system, image_model, det_teacher, frames, 42, lambda_for_qp(42), 2)
    assert all(torch.equal(a, bn.running_mean) for a, bn in zip(before, batchnorms)), (
        "BatchNorm running statistics changed during a training-mode forward")
    # Same input must now give zero clone loss: no batch-statistics noise floor.
    sample = frames[:, 0]
    assert torch.allclose(system.det_clone(sample), extract_teacher_feature(det_teacher, sample),
                          atol=1e-5), "clone does not match the teacher on identical input"


def check_warm_start_layouts(args, device):
    import tempfile
    from multitask_exp.checkpoints import is_lambda_1_64, lambda_evidence, load_raw

    source_dmc = DMC()
    _, source_clone = make_yolo_teacher_and_clone(args.det_weights, torch.device("cpu"))
    with torch.no_grad():
        for parameter in source_clone.parameters():
            parameter.add_(0.01)
    layouts = {
        "paper_like": ({"p_net": {f"module.{k}": v for k, v in source_dmc.state_dict().items()},
                        "student_front": source_clone.state_dict(), "epoch": 11},
                       {"train": {"lambda_min": 1.0, "lambda_max": 64.0}}, True),
        "train_base": ({"state_dict": source_dmc.state_dict(),
                        "cloned_frontend_state_dict": source_clone.state_dict(),
                        "lambda_range": (1.0, 64.0), "lambda_mapping": "geometric_qp"},
                       None, True),
        "shaped": ({"state_dict": source_dmc.state_dict()},
                   {"lambda_min": 0.25, "lambda_max": 64.0, "lambda_mapping": "geometric_qp_shaped"},
                   False),
    }
    with tempfile.TemporaryDirectory() as root:
        for name, (payload, config, expect_1_64) in layouts.items():
            folder = Path(root) / name
            folder.mkdir()
            path = folder / "best.pth.tar"
            torch.save(payload, path)
            if config is not None:
                (folder / "config.json").write_text(json.dumps(config))
            evidence = lambda_evidence(path, load_raw(path))
            assert is_lambda_1_64(evidence) == expect_1_64, (name, evidence)
            target_dmc = DMC()
            _, target_clone = make_yolo_teacher_and_clone(args.det_weights, torch.device("cpu"))
            _, clone_source = warm_start(target_dmc, target_clone, path)
            for key, value in source_dmc.state_dict().items():
                assert torch.equal(target_dmc.state_dict()[key], value), (name, key)
            if name == "shaped":
                assert clone_source == "pretrained_yolov5s"
            else:
                assert clone_source.startswith("init_checkpoint:"), (name, clone_source)
                for key, value in source_clone.state_dict().items():
                    assert torch.equal(target_clone.state_dict()[key], value), (name, key)


def parse_args():
    parser = argparse.ArgumentParser(description="Multi-task (det + frozen seg) DCVC-RT-VCM training")
    parser.add_argument("--dataset", help="Vimeo-90K Septuplet root")
    parser.add_argument("--model_path_i", default="./checkpoints/dcvc_rt/cvpr2025_image.pth.tar")
    parser.add_argument("--model_path_p", default="./checkpoints/dcvc_rt/cvpr2025_video.pth.tar")
    parser.add_argument("--init_checkpoint",
                        help="Warm start: DMC state_dict and detection clone from a trained checkpoint")
    parser.add_argument("--det_weights", default="./yolov5s.pt")
    parser.add_argument("--seg_weights", default="multitask_exp/weights/yolov5s-seg.pt")
    parser.add_argument("--seg_layer", type=int, default=17)
    parser.add_argument("--det_mode", choices=("clone", "frozen"), default="clone",
                        help="clone = paper's trainable-clone recipe (measured to drift over "
                             "extended training, see docs/theoretical_foundation.md); "
                             "frozen = no trainable detection parameters at all, like the "
                             "segmentation branch")
    parser.add_argument("--alpha_det", type=float, default=0.5,
                        help="alpha_seg = 1 - alpha_det; 1.0 = R0, 0.0 = R1, 0.5 = R2")
    parser.add_argument("--seg_scale", type=float,
                        help="Multiplier bringing D_seg to the D_det scale")
    parser.add_argument("--task_scale_file", help="JSON from measure_task_scales with key seg_scale")
    parser.add_argument("--save_dir", default="multitask_exp/output/runs")
    parser.add_argument("--run_name", default="R2")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--group_size", type=int, choices=(5, 6), default=6)
    parser.add_argument("--crop_size", type=int, choices=(256,), default=256)
    parser.add_argument("--no_hflip", dest="hflip", action="store_false",
                        help="Disable the clip-level horizontal flip (on by default, as in the paper)")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--validation_list", default="sep_testlist.txt")
    parser.add_argument("--max_train_batches", type=int, default=0,
                        help="Stop each epoch early (timing runs); 0 = full epoch")
    parser.add_argument("--max_val_batches", type=int, default=0)
    parser.add_argument("--resume", action="store_true", help="Resume from <save_dir>/<run_name>/last.pth.tar")
    parser.add_argument("--deadline_unix", type=float, default=0.0,
                        help="Do not start an epoch that would end after this Unix time (0 = off)")
    parser.add_argument("--epoch_minutes_estimate", type=float, default=0.0,
                        help="Epoch duration assumed before the first epoch has been timed")
    parser.add_argument("--random_init", action="store_true",
                        help="Testing only: skip loading DCVC-RT weights")
    parser.add_argument("--self_check", action="store_true")
    args = parser.parse_args()
    if not 0.0 <= args.alpha_det <= 1.0:
        raise ValueError("--alpha_det must be in [0, 1]")
    return args


def main():
    args = parse_args()
    if args.self_check:
        self_check(args)
        return
    if not args.dataset:
        raise ValueError("--dataset is required")
    device, rank, world_size, local_rank = setup_distributed(args.device)
    try:
        train_worker(args, device, rank, world_size, local_rank)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
