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
from multitask_exp.exact_rate import forward_train_exact
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
        "rate_mode": args.rate_mode,
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
            # No trainable detection parameters at all -- structurally identical to the
            # segmentation branch. Measured to give the same real detection quality as
            # det_mode="clone" (R3 ~ R0b, R4 ~ R2b), so it is a simplification, not a fix.
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
        det_frozen=det_frozen, rate_mode=config["rate_mode"]).to(device)
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


def has_det_branch(system):
    """True for either detection supervision mode (trainable clone or frozen)."""
    return system.det_clone is not None or system.det_frozen is not None


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
        "val_feature_mse_det": d_det / batches if has_det_branch(system) else None,
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

    # broadcast_buffers=False matches the paper script: every BatchNorm here is frozen in
    # eval mode, so re-broadcasting buffers from rank 0 each forward only costs bandwidth.
    model = DistributedDataParallel(system, device_ids=[local_rank], output_device=local_rank,
                                    broadcast_buffers=False, find_unused_parameters=False) \
        if world_size > 1 else system
    model.train()
    parameters = tuple(p for p in system.parameters() if p.requires_grad)
    optimizer = torch.optim.Adam(parameters, lr=args.learning_rate)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)

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
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=args.amp):
                loss, rate, d_det, d_seg = forward_group(
                    model, system, image_model, det_teacher, sequences, base_qp,
                    lambda_for_qp(base_qp), args.group_size)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)  # clip on true gradients, as in the paper script
            grad_norm = torch.nn.utils.clip_grad_norm_(
                parameters, args.grad_clip, error_if_nonfinite=not args.amp)
            scaler.step(optimizer)
            scaler.update()
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
            "feature_mse_det": d_det / batches if has_det_branch(system) else None,
            "feature_mse_seg": d_seg / batches if system.seg_branch is not None else None,
            "grad_norm": grad_norm / batches,
            "train_seconds_rank0": elapsed,
            "seconds_per_global_batch": elapsed / (batches / world_size),
            "global_batches": int(batches / world_size),
            "full_epoch_batches": len(loader),
            "amp": bool(args.amp),
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


def codec_grad_vector(system):
    """Flattened d(loss)/d(codec parameters), the quantity optimisation actually follows."""
    parts = [p.grad.detach().reshape(-1) for p in system.video_model.parameters()
             if p.grad is not None]
    return torch.cat(parts) if parts else None


def diagnose_rates(args):
    """Price the SAME warm-started model with both rate estimators. Trains nothing.

    Meant for the paper checkpoint: if the unrounded-latent surrogate and the exact
    rounded-symbol cost disagree on it, continued training with the surrogate is
    optimising a different objective from the one the checkpoint came from.

    The two estimators can agree on bpp to a fraction of a percent while pointing the
    gradient in a visibly different direction, so this measures BOTH: the forward bpp
    and, for the same batch, the cosine and norm ratio between the two gradients over
    the codec parameters. Only the gradient enters an optimiser step, so a bpp ratio
    near 1.0 on its own is NOT evidence against the rate hypothesis.
    """
    device = torch.device(args.device)
    config = run_config(args, resolve_seg_scale(args))
    image_model, det_teacher, system = build_system(args, config, device)
    system.train()  # the gradient we care about is the training-time one (BN stays frozen)
    dataset = VimeoSeptupletFlip(
        args.dataset, args.crop_size, args.group_size,
        list_name=args.validation_list, random_crop=False, hflip=False)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers)
    table = {qp: {"surrogate": [], "exact": [], "grad": []} for qp in VALIDATION_QPS}
    for batch_index, sequences in enumerate(tqdm(loader, desc="rate diagnosis")):
        if batch_index >= args.diagnose_batches:
            break
        qp = VALIDATION_QPS[batch_index % len(VALIDATION_QPS)]
        sequences = sequences.to(device)
        grads = {}
        for mode in ("surrogate", "exact"):
            system.rate_mode = mode
            system.zero_grad(set_to_none=True)
            loss, rate, d_det, d_seg = forward_group(
                system, system, image_model, det_teacher, sequences, qp,
                lambda_for_qp(qp), args.group_size)
            loss.backward()
            grads[mode] = codec_grad_vector(system)
            system.video_model.clear_dpb()
            table[qp][mode].append((loss.item(), rate.item(), nan_to_zero(d_det), nan_to_zero(d_seg)))
        system.zero_grad(set_to_none=True)
        g_s, g_e = grads["surrogate"], grads["exact"]
        if g_s is not None and g_e is not None and g_e.norm() > 0:
            table[qp]["grad"].append((
                torch.nn.functional.cosine_similarity(g_s, g_e, dim=0).item(),
                (g_s.norm() / g_e.norm()).item()))
        del grads, g_s, g_e

    def mean(rows, column):
        return sum(row[column] for row in rows) / len(rows)

    summary = {}
    print(f"{'qp':>4} {'batches':>8} {'bpp surrogate':>14} {'bpp exact':>10} {'exact/surr':>11} "
          f"{'loss surr':>10} {'loss exact':>11} {'cos(grad)':>10} {'|gs|/|ge|':>10}")
    for qp, modes in table.items():
        if not modes["exact"]:
            continue
        s_rate, e_rate = mean(modes["surrogate"], 1), mean(modes["exact"], 1)
        cosine = mean(modes["grad"], 0) if modes["grad"] else float("nan")
        norm_ratio = mean(modes["grad"], 1) if modes["grad"] else float("nan")
        summary[qp] = {
            "batches": len(modes["exact"]),
            "bpp_surrogate": s_rate, "bpp_exact": e_rate,
            "loss_surrogate": mean(modes["surrogate"], 0), "loss_exact": mean(modes["exact"], 0),
            "d_det": mean(modes["exact"], 2), "d_seg": mean(modes["exact"], 3),
            "grad_cosine": cosine, "grad_norm_ratio": norm_ratio,
        }
        print(f"{qp:>4} {len(modes['exact']):>8} {s_rate:>14.5f} {e_rate:>10.5f} {e_rate / s_rate:>11.3f} "
              f"{summary[qp]['loss_surrogate']:>10.5f} {summary[qp]['loss_exact']:>11.5f} "
              f"{cosine:>10.5f} {norm_ratio:>10.3f}")
    finite = [v["grad_cosine"] for v in summary.values() if math.isfinite(v["grad_cosine"])]
    overall = {
        "bpp_surrogate": sum(v["bpp_surrogate"] for v in summary.values()) / len(summary),
        "bpp_exact": sum(v["bpp_exact"] for v in summary.values()) / len(summary),
        "grad_cosine": sum(finite) / len(finite) if finite else float("nan"),
    }
    print(f"mean over QPs: surrogate {overall['bpp_surrogate']:.5f}  exact {overall['bpp_exact']:.5f}  "
          f"(paper's own val_bpp at its best epoch: 0.1205)")
    print(f"mean cos(grad) between the two estimators: {overall['grad_cosine']:.5f}  "
          f"(1.0 = same direction; read THIS, not the bpp ratio, to judge the rate hypothesis)")
    out = Path(args.save_dir) / "rate_diagnosis.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"per_qp": summary, "overall": overall, "config": config}, indent=2))
    print("saved", out)


def save_warm_start(args):
    """Round-trip the init checkpoint through the exact path a run uses, training nothing.

    Every run goes warm_start -> build_system -> evaluation_payload -> atomic_save, and every
    run so far came out damaged on both task axes. A silent parameter loss anywhere on that
    path would produce exactly that, whatever the loss function was. Evaluating this file and
    the init checkpoint must give identical numbers; if they differ, the objective is not on
    trial at all -- the plumbing is.
    """
    device = torch.device(args.device)
    if not args.init_checkpoint:
        raise ValueError("--save_warm_start requires --init_checkpoint")
    config = run_config(args, resolve_seg_scale(args))
    _, _, system = build_system(args, config, device)
    payload = evaluation_payload(system, config, epoch=0)
    path = Path(args.save_warm_start)
    atomic_save(payload, path)

    source = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
    source_state = next((source[key] for key in ("dmc_state_dict", "p_net", "model_state_dict",
                                                 "state_dict") if key in source), None)
    saved = system.video_model.state_dict()
    missing = sorted(set(saved) - set(source_state or {}))
    identical = sum(1 for key in saved
                    if key in (source_state or {})
                    and torch.equal(saved[key].cpu(), source_state[key].cpu()))
    print(f"saved {path}")
    print(f"codec tensors: {len(saved)} total, {identical} bit-identical to the init checkpoint, "
          f"{len(missing)} absent from it")
    if missing:
        print("absent keys (these came from the pretrained DCVC-RT weights, not the init "
              f"checkpoint): {missing[:8]}{' ...' if len(missing) > 8 else ''}")


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
    combinations = ((0.0, "clone", "surrogate"), (0.5, "clone", "surrogate"),
                    (1.0, "frozen", "surrogate"), (0.5, "frozen", "surrogate"),
                    (1.0, "frozen", "exact"), (0.5, "frozen", "exact"))
    for alpha_det, det_mode, rate_mode in combinations:
        args.alpha_det = alpha_det
        args.det_mode = det_mode
        args.rate_mode = rate_mode
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
        def reported(value):
            # A branch that is switched off yields NaN; report it as null, not as 0.0,
            # so the output never reads like a branch that trained to zero error.
            value = float(value)
            return value if math.isfinite(value) else None

        results[f"alpha_det={alpha_det},det_mode={det_mode},rate_mode={rate_mode}"] = {
            "loss": loss.item(), "d_det": reported(d_det), "d_seg": reported(d_seg),
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

    check_exact_rate_matches_paper(device)
    check_warm_start_layouts(args, device)
    print(json.dumps(results, indent=2))
    print("multitask_exp self-check passed: exact-rate forward matches the paper script "
          "(values and gradients), lambda 1-64, detection-clone BatchNorm stays in "
          "eval mode through train(), frozen seg branch teaches the codec "
          "without being updated, checkpoints load with evaluate_vcm.load_codec_checkpoint, "
          "warm start accepts train_base and paper (p_net/student_front) layouts")


def check_exact_rate_matches_paper(device):
    """forward_train_exact must reproduce the paper script's p_frame_forward: values and gradients."""
    import copy
    from multitask_exp.paper_reference import p_frame_forward as paper_forward

    torch.manual_seed(3)
    reference_model = DMC().to(device).eval()
    with torch.no_grad():
        for parameter in reference_model.parameters():
            parameter.add_(0.02 * torch.randn_like(parameter))
    ours_model = copy.deepcopy(reference_model)
    first = torch.rand(1, 3, 64, 64, device=device)
    ref = torch.rand(1, 3, 64, 64, device=device)

    def prime(model):
        model.clear_dpb()
        model.set_curr_poc(0)
        model.add_ref_frame(None, ref)

    for qp in (0, 21, 42, 63):
        prime(reference_model); prime(ours_model)
        paper_x, paper_bpp = paper_forward(reference_model, first, qp)
        ours_x, ours_bpp = forward_train_exact(ours_model, first, qp)
        assert torch.allclose(paper_x, ours_x, atol=1e-5), f"x_hat differs at qp {qp}"
        assert math.isclose(paper_bpp.mean().item(), ours_bpp.item(), rel_tol=1e-5), f"bpp differs at qp {qp}"

    prime(reference_model); prime(ours_model)
    reference_model.zero_grad(); ours_model.zero_grad()
    paper_x, paper_bpp = paper_forward(reference_model, first, 21)
    (paper_bpp.mean() + paper_x.pow(2).mean()).backward()
    ours_x, ours_bpp = forward_train_exact(ours_model, first, 21)
    (ours_bpp + ours_x.pow(2).mean()).backward()
    difference = sum((a.grad - b.grad).norm() ** 2 for a, b in
                     zip(reference_model.parameters(), ours_model.parameters())
                     if a.grad is not None) ** 0.5
    scale = sum(a.grad.norm() ** 2 for a in reference_model.parameters() if a.grad is not None) ** 0.5
    assert difference / scale < 1e-4, f"gradients differ: {(difference / scale).item():.2e}"


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
                        help="clone = the paper script's trainable copy of layers 0-4; "
                             "frozen = no trainable detection parameters, like the segmentation "
                             "branch. Measured equivalent in real detection quality.")
    parser.add_argument("--rate_mode", choices=("surrogate", "exact"), default="surrogate",
                        help="surrogate = DMC.forward_train (rate on unrounded latents, used by "
                             "R0-R4); exact = rate on the rounded symbols, as in the script that "
                             "trained the paper checkpoint")
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
    parser.add_argument("--amp", action="store_true",
                        help="Mixed precision (fp16 autocast + GradScaler), as in the paper script. "
                             "Off by default so R3x/R4x keep their behaviour. Needs CUDA and "
                             "--rate_mode exact: only the exact path keeps entropy coding in fp32, "
                             "the surrogate path lives in dcvc_rt/ which this project must not edit.")
    parser.add_argument("--random_init", action="store_true",
                        help="Testing only: skip loading DCVC-RT weights")
    parser.add_argument("--diagnose_rates", action="store_true",
                        help="price the warm-started model with both rate estimators, no training")
    parser.add_argument("--diagnose_batches", type=int, default=120)
    parser.add_argument("--save_warm_start",
                        help="Build the system from --init_checkpoint, train NOTHING, and save "
                             "the epoch-0 checkpoint here. Evaluating this file must reproduce "
                             "the init checkpoint's own numbers exactly; anything else means the "
                             "warm-start/save/load path loses parameters, and every trained run "
                             "that went through it is void.")
    parser.add_argument("--self_check", action="store_true")
    args = parser.parse_args()
    if not 0.0 <= args.alpha_det <= 1.0:
        raise ValueError("--alpha_det must be in [0, 1]")
    if args.amp and args.rate_mode != "exact":
        raise ValueError("--amp requires --rate_mode exact (the surrogate path would price "
                         "entropy in fp16, which the paper script never does)")
    if args.amp and not args.device.startswith("cuda"):
        raise ValueError("--amp requires a CUDA device")
    return args


def main():
    args = parse_args()
    if args.self_check:
        self_check(args)
        return
    if args.save_warm_start:
        save_warm_start(args)
        return
    if not args.dataset:
        raise ValueError("--dataset is required")
    if args.diagnose_rates:
        diagnose_rates(args)
        return
    device, rank, world_size, local_rank = setup_distributed(args.device)
    try:
        train_worker(args, device, rank, world_size, local_rank)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
