import argparse
import gc
import json
import math
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path

import torch
import torch.distributed as dist
from PIL import Image
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torchvision.transforms.functional import to_tensor
from tqdm import tqdm

from dcvc_rt.src.models.image_model import DMCI
from dcvc_rt.src.models.video_model import DMC
from dcvc_rt.src.utils.common import get_state_dict
from dcvc_rt.src.utils.transforms import rgb2ycbcr
from svc_machine.feature_extractor import extract_teacher_feature, make_yolo_teacher_and_clone
from svc_machine.feature_loss import feature_mse_loss, machine_rate_distortion_loss
from svc_machine.system import MachineBaseSystem


LAMBDA_MIN = 2.0
LAMBDA_MAX = 16.0
INDEX_MAP = (0, 1, 0, 2, 0, 2, 0, 2)
VALIDATION_QPS = (0, 21, 42, 63)
PAPER_LAMBDAS = (2, 4, 8, 16)
PROJECT_ROOT = Path(__file__).resolve().parent


def setup_distributed(device_name):
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    rank = int(os.environ.get('RANK', '0'))
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    if world_size == 1:
        if device_name.startswith('cuda') and not torch.cuda.is_available():
            raise RuntimeError('CUDA is required but is not available')
        return torch.device(device_name), rank, world_size, local_rank
    if not torch.cuda.is_available():
        raise RuntimeError('Distributed training requires CUDA/NCCL')
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')
    return torch.device('cuda', local_rank), rank, world_size, local_rank


def capture_rng_state(device):
    return {
        'python': random.getstate(),
        'torch': torch.get_rng_state(),
        'cuda': torch.cuda.get_rng_state(device) if device.type == 'cuda' else None,
    }


def gather_rng_states(device, world_size):
    state = capture_rng_state(device)
    if world_size == 1:
        return [state]
    states = [None] * world_size
    dist.all_gather_object(states, state)
    return states


def restore_rng_state(states, rank, device):
    if len(states) <= rank:
        raise ValueError('Resume checkpoint does not contain RNG state for this rank')
    state = states[rank]
    random.setstate(state['python'])
    torch.set_rng_state(state['torch'])
    if device.type == 'cuda' and state['cuda'] is not None:
        torch.cuda.set_rng_state(state['cuda'], device)


def synchronized_qp(mode, fixed_qp, device, rank, world_size):
    qp = random.randint(0, DMCI.get_qp_num() - 1) if mode == 'variable' else fixed_qp
    if world_size > 1:
        value = torch.tensor(qp if rank == 0 else 0, device=device)
        dist.broadcast(value, 0)
        qp = int(value.item())
    return qp


def lambda_for_qp(qp):
    if not 0 <= qp < DMCI.get_qp_num():
        raise ValueError('base QP must be in the range 0..63')
    position = qp / (DMCI.get_qp_num() - 1)
    return math.exp(math.log(LAMBDA_MIN) + position * math.log(LAMBDA_MAX / LAMBDA_MIN))


def qp_schedule(mode, fixed_qp):
    return ('fixed', fixed_qp) if mode == 'fixed' \
        else ('uniform', 0, DMCI.get_qp_num() - 1)


def training_tag(mode, fixed_qp, lambda_task):
    return f'lambda{lambda_task:g}_random_qp' if mode == 'variable' \
        else f'lambda{lambda_task:g}_qp{fixed_qp}'


class VimeoSeptuplet(Dataset):
    def __init__(self, root, crop_size=256, group_size=5,
                 list_name='sep_trainlist.txt', random_crop=True):
        self.root = Path(root)
        self.sequence_root = self.root / 'sequences'
        list_path = self.root / list_name
        if not self.sequence_root.is_dir() or not list_path.is_file():
            raise FileNotFoundError(
                f'{self.root} must contain sequences/ and {list_name}')
        self.sequences = [line.strip() for line in list_path.read_text().splitlines() if line.strip()]
        if not self.sequences:
            raise ValueError(f'No Vimeo-90K sequences listed in {list_path}')
        self.crop_size = crop_size
        self.group_size = group_size
        self.random_crop = random_crop

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        frame_count = self.group_size + 1
        start = random.randint(1, 8 - frame_count) if self.random_crop else 1
        folder = self.sequence_root / self.sequences[index]
        frames = []
        for frame_index in range(start, start + frame_count):
            path = folder / f'im{frame_index}.png'
            with Image.open(path) as image:
                frames.append(to_tensor(image.convert('RGB')))
        height, width = frames[0].shape[-2:]
        if any(frame.shape[-2:] != (height, width) for frame in frames):
            raise ValueError(f'Frame sizes differ in {folder}')
        if height < self.crop_size or width < self.crop_size:
            raise ValueError(f'{folder} is smaller than {self.crop_size}x{self.crop_size}')
        if self.random_crop:
            top = random.randint(0, height - self.crop_size)
            left = random.randint(0, width - self.crop_size)
        else:
            top = (height - self.crop_size) // 2
            left = (width - self.crop_size) // 2
        return torch.stack([
            frame[:, top:top + self.crop_size, left:left + self.crop_size]
            for frame in frames
        ])


def save_training_outputs(history, json_path, plot_path):
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(history, indent=2), encoding='utf-8')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    epochs = [item['epoch'] for item in history]
    figure, axes = plt.subplots(1, 3, figsize=(15, 4))
    for axis, key, title in zip(
            axes, ('total_loss', 'bpp', 'feature_mse'),
            ('Total Loss', 'BPP', 'Feature MSE')):
        axis.plot(epochs, [item[key] for item in history], marker='o', label='train')
        axis.plot(epochs, [item.get(f'val_{key}', math.nan) for item in history],
                  marker='s', label='validation')
        axis.set(title=title, xlabel='Epoch', ylabel=title)
        axis.grid(True, alpha=0.3)
        axis.legend()
    figure.tight_layout()
    figure.savefig(plot_path, dpi=150)
    plt.close(figure)


def save_checkpoint(path, epoch, mode, lambda_task, system, optimizer,
                    training_history, validation_history, best_bd_rate,
                    rng_states, world_size, fixed_qp=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + '.tmp')
    metadata = {
        'epoch': epoch,
        'mode': mode,
        'lambda_task': lambda_task,
        'lambda_range': (LAMBDA_MIN, LAMBDA_MAX),
        'qp_sampling': qp_schedule(mode, fixed_qp),
        'hierarchical_qp': mode != 'fixed',
    }
    if fixed_qp is not None:
        metadata.update({'fixed_qp': fixed_qp, 'fixed_lambda': lambda_task})
    torch.save({
        **metadata,
        'schema_version': 7,
        'state_dict': system.video_model.state_dict(),
        'cloned_frontend_state_dict': system.cloned_frontend.state_dict(),
        'trainable_components': ['dcvc_rt_dmc', 'yolo_cloned_frontend'],
        'frozen_components': ['dmci', 'yolo_teacher', 'yolo_backend'],
        'optimizer': optimizer.state_dict(),
        'training_history': training_history,
        'validation_history': validation_history,
        'best_bd_rate_map_percent': best_bd_rate,
        'rng_states': rng_states,
        'world_size': world_size,
    }, temporary_path)
    temporary_path.replace(path)


def resume_training(path, mode, fixed_qp, lambda_task, system, optimizer, world_size):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    required = {
        'schema_version', 'epoch', 'mode', 'lambda_task', 'lambda_range', 'qp_sampling',
        'hierarchical_qp', 'state_dict',
        'cloned_frontend_state_dict',
        'optimizer', 'rng_states', 'world_size',
    }
    missing = sorted(required - checkpoint.keys())
    if missing:
        raise ValueError(f'Resume checkpoint is missing: {", ".join(missing)}')
    if checkpoint['schema_version'] != 7:
        raise ValueError('Unsupported resume checkpoint schema')
    if checkpoint['mode'] != mode:
        raise ValueError(f'Resume checkpoint mode is {checkpoint["mode"]}, not {mode}')
    if float(checkpoint['lambda_task']) != lambda_task:
        raise ValueError('Resume checkpoint uses a different fixed task lambda')
    if tuple(checkpoint['lambda_range']) != (LAMBDA_MIN, LAMBDA_MAX):
        raise ValueError('Resume checkpoint uses a different lambda range')
    if tuple(checkpoint['qp_sampling']) != qp_schedule(mode, fixed_qp):
        raise ValueError('Resume checkpoint uses a different QP sampling strategy')
    if checkpoint['hierarchical_qp'] != (mode != 'fixed'):
        raise ValueError('Resume checkpoint uses a different frame-QP schedule')
    if mode == 'fixed' and checkpoint.get('fixed_qp') != fixed_qp:
        raise ValueError('Resume checkpoint uses a different fixed QP')
    if checkpoint['world_size'] != world_size:
        raise ValueError('Resume must use the same number of distributed processes')
    system.video_model.load_state_dict(checkpoint['state_dict'])
    system.cloned_frontend.load_state_dict(checkpoint['cloned_frontend_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return checkpoint


def save_evaluation_checkpoint(path, epoch, mode, lambda_task, system, fixed_qp=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + '.tmp')
    metadata = {
        'epoch': epoch,
        'mode': mode,
        'lambda_task': lambda_task,
        'lambda_range': (LAMBDA_MIN, LAMBDA_MAX),
        'qp_sampling': qp_schedule(mode, fixed_qp),
        'hierarchical_qp': mode != 'fixed',
    }
    if fixed_qp is not None:
        metadata.update({'fixed_qp': fixed_qp, 'fixed_lambda': lambda_task})
    torch.save({
        **metadata,
        'schema_version': 7,
        'state_dict': system.video_model.state_dict(),
        'cloned_frontend_state_dict': system.cloned_frontend.state_dict(),
        'trainable_components': ['dcvc_rt_dmc', 'yolo_cloned_frontend'],
        'frozen_components': ['dmci', 'yolo_teacher', 'yolo_backend'],
    }, temporary_path)
    temporary_path.replace(path)


def move_optimizer_state(optimizer, device):
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def load_validation_config(path):
    config = json.loads(Path(path).read_text(encoding='utf-8'))
    required = {'data_dir', 'dataset_manifest', 'anchor_results'}
    missing = sorted(required - config.keys())
    if missing:
        raise ValueError(f'Missing validation config fields: {", ".join(missing)}')
    config['selection_metric'] = config.get('selection_metric', 'map5095')
    if config['selection_metric'] not in ('map50', 'map5095'):
        raise ValueError('selection_metric must be map50 or map5095')
    qps = tuple(config.get('qps', VALIDATION_QPS))
    if qps != VALIDATION_QPS:
        raise ValueError('Validation QPs must be 0 21 42 63')
    config['qps'] = qps
    for key in required:
        if not Path(config[key]).exists():
            raise FileNotFoundError(f'Validation {key} not found: {config[key]}')
    return config


def validate_checkpoint(args, checkpoint_path, output_dir, config):
    output_dir = Path(output_dir)
    codec_output = output_dir / 'codec'
    comparison_output = output_dir / 'comparison'
    bitstream_dir = output_dir / 'bitstreams'
    method_name = f'validation_epoch_{int(torch.load(checkpoint_path, map_location="cpu", weights_only=True)["epoch"]):04d}'
    candidate_results = codec_output / f'{method_name}_results.json'

    if not candidate_results.is_file():
        command = [
            sys.executable, str(PROJECT_ROOT / 'evaluate_vcm.py'), '--mode', 'codec',
            '--data-dir', str(config['data_dir']),
            '--dataset-manifest', str(config['dataset_manifest']),
            '--image-ckpt', str(args.model_path_i),
            '--video-ckpt', str(checkpoint_path),
            '--qps', *(str(qp) for qp in config['qps']),
            '--reset-interval', str(config.get('reset_interval', 64)),
            '--force-zero-thres', str(config.get('force_zero_thres', 0.12)),
            '--codec-precision', config.get('codec_precision', 'fp16'),
            '--yolov5-weights', str(config.get('yolov5_weights', args.weights)),
            '--detector-size', str(config.get('detector_size', 640)),
            '--confidence-threshold', str(config.get('confidence_threshold', 0.001)),
            '--nms-iou-threshold', str(config.get('nms_iou_threshold', 0.6)),
            '--max-detections', str(config.get('max_detections', 300)),
            '--cuda-index', str(config.get('cuda_index', 0)),
            '--method-name', method_name,
            '--output-dir', str(codec_output),
            '--bitstream-dir', str(bitstream_dir),
        ]
        if config.get('max_sequences') is not None:
            command.extend(['--max-sequences', str(config['max_sequences'])])
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)

    subprocess.run([
        sys.executable, str(PROJECT_ROOT / 'evaluate_vcm.py'), '--mode', 'bdrate',
        '--anchor-results', str(config['anchor_results']),
        '--candidate-results', str(candidate_results),
        '--rate', config.get('rate', 'actual_bpp'),
        '--metric', config['selection_metric'],
        '--output-dir', str(comparison_output),
    ], cwd=PROJECT_ROOT, check=True)
    comparison = json.loads(
        (comparison_output / 'bd_rate_map.json').read_text(encoding='utf-8'))
    if bitstream_dir.is_dir() and not config.get('keep_bitstreams', False):
        shutil.rmtree(bitstream_dir)
    metrics = comparison['metrics']
    return {
        'selection_metric': config['selection_metric'],
        'bd_rate_map_percent': metrics[config['selection_metric']]['bd_rate_percent'],
        'bd_rate_map50_percent': metrics['map50']['bd_rate_percent'],
        'bd_rate_map5095_percent': metrics['map5095']['bd_rate_percent'],
        'candidate_results': str(candidate_results),
        'comparison_results': str(comparison_output / 'bd_rate_map.json'),
    }


def save_best_checkpoint(source_path, destination_path, validation_record):
    checkpoint = torch.load(source_path, map_location='cpu', weights_only=True)
    checkpoint['selected_by'] = validation_record
    checkpoint['best_bd_rate_map_percent'] = validation_record['bd_rate_map_percent']
    checkpoint['validation_history'] = [validation_record]
    temporary_path = destination_path.with_suffix(destination_path.suffix + '.tmp')
    torch.save(checkpoint, temporary_path)
    temporary_path.replace(destination_path)


def update_validation_metadata(path, history, best_bd_rate):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    checkpoint['validation_history'] = history
    checkpoint['best_bd_rate_map_percent'] = best_bd_rate
    temporary_path = path.with_suffix(path.suffix + '.tmp')
    torch.save(checkpoint, temporary_path)
    temporary_path.replace(path)


def select_best_checkpoint(args, tag, validation_config):
    save_dir = Path(args.save_dir)
    snapshots = sorted(save_dir.glob(f'video_{tag}_epoch_*.pth'))
    best_path = save_dir / 'best.pth'
    if not snapshots:
        if best_path.is_file():
            print(f'Best checkpoint already exists: {best_path}')
            return
        raise FileNotFoundError('No validation snapshots were saved')

    history_path = save_dir / f'{tag}_validation_history.json'
    history = json.loads(history_path.read_text(encoding='utf-8')) \
        if history_path.is_file() else []
    completed_epochs = {int(record['epoch']) for record in history}
    for snapshot in snapshots:
        checkpoint = torch.load(snapshot, map_location='cpu', weights_only=True)
        epoch = int(checkpoint['epoch'])
        if epoch in completed_epochs:
            continue
        print(f'Validating epoch {epoch} by actual-bitstream BD-rate-mAP...')
        result = validate_checkpoint(
            args, snapshot, save_dir / f'validation_{tag}' / f'epoch_{epoch:04d}',
            validation_config)
        if not math.isfinite(result['bd_rate_map_percent']):
            raise ValueError('Validation BD-rate-mAP is not finite')
        history.append({'epoch': epoch, 'checkpoint': str(snapshot), **result})
        history.sort(key=lambda record: int(record['epoch']))
        history_path.write_text(json.dumps(history, indent=2), encoding='utf-8')

    metric = validation_config['selection_metric']
    best = min(history, key=lambda record: record['bd_rate_map_percent'])
    best_snapshot = next(
        snapshot for snapshot in snapshots
        if int(torch.load(snapshot, map_location='cpu', weights_only=True)['epoch']) == best['epoch'])
    save_best_checkpoint(best_snapshot, best_path, best)
    last_path = save_dir / f'video_{tag}_last.pth.tar'
    if last_path.is_file():
        update_validation_metadata(
            last_path, history, best['bd_rate_map_percent'])
    (save_dir / 'best_validation.json').write_text(
        json.dumps(best, indent=2), encoding='utf-8')
    print(f'Best epoch {best["epoch"]}: BD-rate-{metric} '
          f'{best["bd_rate_map_percent"]:.3f}% -> {best_path}')


def forward_group(model, system, image_model, teacher, sequences, base_qp,
                  lambda_task, group_size, hierarchical_qp):
    system.video_model.clear_dpb()
    system.video_model.set_curr_poc(0)
    with torch.no_grad():
        reference_ycbcr = image_model.forward_reconstruction(
            rgb2ycbcr(sequences[:, 0]), base_qp)
    system.video_model.add_ref_frame(None, reference_ycbcr)

    rgb_frames = sequences[:, 1:]
    ycbcr_frames = torch.stack([
        rgb2ycbcr(rgb_frames[:, index])
        for index in range(rgb_frames.shape[1])
    ], dim=1)
    target_features = torch.stack([
        extract_teacher_feature(teacher, rgb_frames[:, index])
        for index in range(rgb_frames.shape[1])
    ], dim=1)
    qps = tuple(
        system.video_model.shift_qp(base_qp, INDEX_MAP[frame_index % 8])
        if hierarchical_qp else base_qp
        for frame_index in range(1, group_size + 1))
    result = model(ycbcr_frames, target_features, qps, lambda_task)
    return result, lambda_task


@torch.no_grad()
def validate_training_epoch(model, system, image_model, teacher, loader,
                            group_size, mode, fixed_qp, lambda_task,
                            device, rank, world_size):
    model.eval()
    totals = torch.zeros(4, device=device, dtype=torch.float64)
    progress = tqdm(loader, desc='validation', disable=rank != 0)
    for batch_index, sequences in enumerate(progress):
        base_qp = fixed_qp if mode == 'fixed' \
            else VALIDATION_QPS[batch_index % len(VALIDATION_QPS)]
        sequences = sequences.to(device, non_blocking=True)
        (loss, rate, task_loss), _ = forward_group(
            model, system, image_model, teacher, sequences, base_qp,
            lambda_task, group_size, mode != 'fixed')
        system.video_model.clear_dpb()
        totals += torch.tensor(
            [loss.item(), rate.item(), task_loss.item(), 1],
            device=device, dtype=torch.float64)
    if world_size > 1:
        dist.all_reduce(totals)
    model.train()
    if totals[3].item() == 0:
        raise RuntimeError('No complete validation batch was produced')
    return {
        'val_total_loss': (totals[0] / totals[3]).item(),
        'val_bpp': (totals[1] / totals[3]).item(),
        'val_feature_mse': (totals[2] / totals[3]).item(),
    }


def train_worker(args, device, rank, world_size, local_rank):
    if args.epochs <= 0 or args.learning_rate <= 0 or args.batch_size <= 0 or args.workers < 0:
        raise ValueError('epochs, learning_rate and batch_size must be positive; workers cannot be negative')
    if args.validation_interval <= 0 or args.grad_clip <= 0:
        raise ValueError('validation_interval and grad_clip must be positive')
    if args.batch_size % world_size:
        raise ValueError('Global batch_size must be divisible by the distributed world size')
    if args.mode == 'fixed' and args.validation_config:
        raise ValueError(
            'BD-rate selection requires the four fixed-lambda models together; '
            'train each model without --validation_config')
    random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    if device.type == 'cuda':
        torch.cuda.manual_seed(args.seed + rank)
    lambda_task = float(args.paper_lambda) if args.paper_lambda is not None \
        else lambda_for_qp(args.fixed_qp)
    validation_config = load_validation_config(args.validation_config) \
        if args.validation_config else None
    if validation_config and device.type != 'cuda':
        raise ValueError('Real-bitstream validation requires a CUDA device')

    dataset = VimeoSeptuplet(args.dataset, args.crop_size, args.group_size)
    validation_dataset = VimeoSeptuplet(
        args.dataset, args.crop_size, args.group_size,
        list_name=args.validation_list, random_crop=False)
    if len(dataset) < args.batch_size:
        raise ValueError('Dataset must contain at least batch_size sequences')
    if len(validation_dataset) < args.batch_size:
        raise ValueError('Validation dataset must contain at least batch_size sequences')
    sampler = DistributedSampler(dataset, shuffle=True) if world_size > 1 else None
    validation_sampler = DistributedSampler(
        validation_dataset, shuffle=False, drop_last=True) if world_size > 1 else None
    loader = DataLoader(dataset, batch_size=args.batch_size // world_size,
                        shuffle=sampler is None, sampler=sampler,
                        num_workers=args.workers, pin_memory=device.type == 'cuda', drop_last=True)
    validation_loader = DataLoader(
        validation_dataset, batch_size=args.batch_size // world_size,
        shuffle=False, sampler=validation_sampler, num_workers=args.workers,
        pin_memory=device.type == 'cuda', drop_last=True)

    image_model = DMCI().to(device)
    image_model.load_state_dict(get_state_dict(args.model_path_i))
    video_model = DMC().to(device)
    video_model.load_state_dict(get_state_dict(args.model_path_p))
    image_model.eval()
    for parameter in image_model.parameters():
        parameter.requires_grad_(False)
    teacher, cloned_frontend = make_yolo_teacher_and_clone(args.weights, device)
    system = MachineBaseSystem(video_model, cloned_frontend).to(device)
    model = DistributedDataParallel(
        system, device_ids=[local_rank], output_device=local_rank) \
        if world_size > 1 else system
    model.train()

    trainable_parameters = tuple(
        parameter for parameter in system.parameters() if parameter.requires_grad)
    optimizer = torch.optim.Adam(trainable_parameters, lr=args.learning_rate)
    tag = training_tag(args.mode, args.fixed_qp, lambda_task)
    save_dir = Path(args.save_dir)
    last_checkpoint = save_dir / f'video_{tag}_last.pth.tar'
    history_path = save_dir / f'{tag}_training_history.json'
    plot_path = save_dir / f'{tag}_training_curves.png'
    start_epoch = 1
    training_history = []
    validation_history = []
    best_bd_rate = math.inf
    best_val_loss = math.inf

    if args.resume:
        resume_path = last_checkpoint if args.resume == 'auto' else Path(args.resume)
        if not resume_path.is_file():
            raise FileNotFoundError(f'Resume checkpoint not found: {resume_path}')
        checkpoint = resume_training(
            resume_path, args.mode, args.fixed_qp, lambda_task,
            system, optimizer, world_size)
        move_optimizer_state(optimizer, device)
        restore_rng_state(checkpoint['rng_states'], rank, device)
        start_epoch = checkpoint['epoch'] + 1
        training_history = checkpoint.get('training_history', [])
        validation_history = checkpoint.get('validation_history', [])
        best_bd_rate = checkpoint.get('best_bd_rate_map_percent', math.inf)
        best_val_loss = min(
            (item.get('val_total_loss', math.inf) for item in training_history),
            default=math.inf)
        if rank == 0:
            print(f'Resumed {resume_path} from epoch {checkpoint["epoch"]}')

    if start_epoch > args.epochs:
        if training_history:
            if rank == 0:
                save_training_outputs(training_history, history_path, plot_path)
        if rank == 0:
            print(f'Training already completed through epoch {start_epoch - 1}')
        return

    for epoch in range(start_epoch, args.epochs + 1):
        if sampler is not None:
            sampler.set_epoch(epoch)
        totals = {'loss': 0.0, 'rate': 0.0, 'task': 0.0, 'grad_norm': 0.0}
        batch_count = 0
        progress = tqdm(loader, desc=f'epoch {epoch}/{args.epochs}', disable=rank != 0)
        for sequences in progress:
            base_qp = synchronized_qp(
                args.mode, args.fixed_qp, device, rank, world_size)
            sequences = sequences.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            (loss, rate, task_loss), _ = forward_group(
                model, system, image_model, teacher, sequences,
                base_qp, lambda_task, args.group_size, args.mode != 'fixed')
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                trainable_parameters, args.grad_clip, error_if_nonfinite=True)
            optimizer.step()
            system.video_model.clear_dpb()

            totals['loss'] += loss.item()
            totals['rate'] += rate.item()
            totals['task'] += task_loss.item()
            totals['grad_norm'] += float(grad_norm)
            batch_count += 1
            progress.set_postfix(loss=totals['loss'] / batch_count,
                                 bpp=totals['rate'] / batch_count,
                                 feature_mse=totals['task'] / batch_count,
                                 grad_norm=totals['grad_norm'] / batch_count,
                                 qp=base_qp, lambda_task=f'{lambda_task:.3f}')

        reduced = torch.tensor([
            totals['loss'], totals['rate'], totals['task'], totals['grad_norm'], batch_count,
        ], device=device, dtype=torch.float64)
        if world_size > 1:
            dist.all_reduce(reduced)
        global_batches = int(reduced[4].item())
        if global_batches == 0:
            raise RuntimeError('No complete training batch was produced')
        record = {
            'epoch': epoch,
            'total_loss': reduced[0].item() / global_batches,
            'bpp': reduced[1].item() / global_batches,
            'feature_mse': reduced[2].item() / global_batches,
            'grad_norm': reduced[3].item() / global_batches,
        }
        record.update(validate_training_epoch(
            model, system, image_model, teacher, validation_loader,
            args.group_size, args.mode, args.fixed_qp, lambda_task,
            device, rank, world_size))
        training_history.append(record)
        improved_val_loss = record['val_total_loss'] < best_val_loss
        best_val_loss = min(best_val_loss, record['val_total_loss'])
        if rank == 0:
            save_training_outputs(training_history, history_path, plot_path)
            shutil.copy2(
                plot_path,
                save_dir / f'{tag}_training_curves_epoch_{epoch:04d}.png')
        fixed_qp = args.fixed_qp if args.mode == 'fixed' else None
        validation_due = (
            epoch % args.validation_interval == 0 or epoch == args.epochs)

        rng_states = gather_rng_states(device, world_size)
        if rank == 0:
            save_checkpoint(last_checkpoint, epoch, args.mode, lambda_task,
                            system, optimizer,
                            training_history, validation_history, best_bd_rate,
                            rng_states, world_size, fixed_qp)
            if validation_due:
                save_evaluation_checkpoint(
                    save_dir / f'video_{tag}_epoch_{epoch:04d}.pth', epoch,
                    args.mode, lambda_task, system, fixed_qp)
            if improved_val_loss:
                save_evaluation_checkpoint(
                    save_dir / 'best_val_loss.pth', epoch, args.mode,
                    lambda_task, system, fixed_qp)
        if world_size > 1:
            dist.barrier()


def train(args):
    device, rank, world_size, local_rank = setup_distributed(args.device)
    completed = False
    try:
        train_worker(args, device, rank, world_size, local_rank)
        completed = True
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
    if completed and rank == 0 and args.validation_config:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        lambda_task = float(args.paper_lambda) if args.paper_lambda is not None \
            else lambda_for_qp(args.fixed_qp)
        tag = training_tag(args.mode, args.fixed_qp, lambda_task)
        select_best_checkpoint(args, tag, load_validation_config(args.validation_config))


def parse_args():
    parser = argparse.ArgumentParser(description='Train the DCVC-RT machine base layer')
    parser.add_argument('--dataset', help='Vimeo-90K Septuplet root')
    parser.add_argument('--mode', choices=('variable', 'fixed'), default='variable',
                        help='variable samples base QP uniformly; fixed is a legacy fixed-QP baseline')
    parser.add_argument('--paper_lambda', type=int, choices=PAPER_LAMBDAS,
                        help='Fixed machine-task lambda; with this option QP is sampled uniformly from 0..63')
    parser.add_argument('--fixed_qp', type=int, default=42,
                        help='Base QP for fixed-rate baseline (default: 42, lambda=8)')
    parser.add_argument('--model_path_i', default='./checkpoints/dcvc_rt/cvpr2025_image.pth.tar')
    parser.add_argument('--model_path_p', default='./checkpoints/dcvc_rt/cvpr2025_video.pth.tar')
    parser.add_argument('--weights', default='./yolov5s.pt')
    parser.add_argument('--save_dir', default='./checkpoints/base_task')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=1e-6)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--group_size', type=int, choices=(5,), default=5,
                        help='Loss-bearing P-frames; one frozen-DMCI reference is loaded in addition')
    parser.add_argument('--crop_size', type=int, choices=(256,), default=256)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--validation_list', default='sep_testlist.txt',
                        help='Held-out Vimeo list used for deterministic validation')
    parser.add_argument('--validation_config',
                        help='Optional JSON config for post-DDP BD-rate-mAP best selection')
    parser.add_argument('--validation_interval', type=int, default=1)
    parser.add_argument('--resume', nargs='?', const='auto',
                        help='Resume at the next epoch; omit PATH to use this mode last checkpoint')
    parser.add_argument('--check_dataset', action='store_true',
                        help='Load one reference plus five cropped P-frames and exit')
    parser.add_argument('--self_check', action='store_true',
                        help='Check QP/lambda interpolation and exit')
    return parser.parse_args()


if __name__ == '__main__':
    arguments = parse_args()
    if arguments.paper_lambda is not None:
        arguments.mode = 'variable'
    elif arguments.mode == 'variable' and not arguments.self_check and not arguments.check_dataset:
        raise ValueError('--mode variable requires --paper_lambda 2, 4, 8, or 16')
    if arguments.self_check:
        from tempfile import TemporaryDirectory

        assert PAPER_LAMBDAS == (2, 4, 8, 16)
        assert qp_schedule('fixed', 42) == ('fixed', 42)
        assert training_tag('variable', 42, 8.0) == 'lambda8_random_qp'
        random_state = random.getstate()
        random.seed(0)
        sampled_qps = [
            synchronized_qp('variable', 42, torch.device('cpu'), 0, 1)
            for _ in range(4096)
        ]
        random.setstate(random_state)
        assert min(sampled_qps) == 0 and max(sampled_qps) == 63
        qp_shift = (0, 8, 4)
        assert [qp_shift[INDEX_MAP[index]] for index in range(5)] == [0, 8, 0, 4, 0]
        target = torch.tensor([0.0, 1.0, 2.0])
        reconstructed = torch.tensor([1.0, 0.0, 0.0])
        distortion = feature_mse_loss(reconstructed, target)
        assert distortion.item() == 2.0
        assert machine_rate_distortion_loss(
            [torch.tensor(1.0)], [distortion], 2.0).item() == 5.0
        trainable_clone = torch.nn.Sequential(torch.nn.BatchNorm1d(1))
        joint_system = MachineBaseSystem(torch.nn.Linear(1, 1), trainable_clone)
        joint_system.train()
        assert joint_system.cloned_frontend.training
        assert all(parameter.requires_grad
                   for parameter in joint_system.cloned_frontend.parameters())
        assert all(parameter.requires_grad
                   for parameter in joint_system.video_model.parameters())
        with TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            history = [{
                'epoch': 1,
                'total_loss': 1.0,
                'bpp': 0.5,
                'feature_mse': 0.25,
                'val_total_loss': 1.1,
                'val_bpp': 0.55,
                'val_feature_mse': 0.3,
            }]
            system = torch.nn.Module()
            system.video_model = torch.nn.Linear(1, 1)
            system.cloned_frontend = torch.nn.Linear(1, 1)
            optimizer = torch.optim.Adam(system.parameters())
            checkpoint_path = directory / 'last.pth.tar'
            snapshot_path = directory / 'epoch_0001.pth'
            best_path = directory / 'best.pth'
            save_training_outputs(history, directory / 'history.json', directory / 'curves.png')
            save_checkpoint(checkpoint_path, 1, 'variable', 8.0, system, optimizer,
                            history, [], math.inf, [capture_rng_state(torch.device('cpu'))], 1)
            save_evaluation_checkpoint(snapshot_path, 1, 'variable', 8.0, system)
            save_best_checkpoint(snapshot_path, best_path, {
                'epoch': 1, 'selection_metric': 'map5095',
                'bd_rate_map_percent': -1.0,
            })
            resumed_system = torch.nn.Module()
            resumed_system.video_model = torch.nn.Linear(1, 1)
            resumed_system.cloned_frontend = torch.nn.Linear(1, 1)
            resumed_optimizer = torch.optim.Adam(resumed_system.parameters())
            checkpoint = resume_training(
                checkpoint_path, 'variable', 42, 8.0,
                resumed_system, resumed_optimizer, 1)
            assert checkpoint['epoch'] == 1
            assert checkpoint['lambda_task'] == 8.0
            assert checkpoint['trainable_components'] == [
                'dcvc_rt_dmc', 'yolo_cloned_frontend']
            assert torch.load(best_path, map_location='cpu', weights_only=True)[
                'selected_by']['selection_metric'] == 'map5095'
            assert (directory / 'history.json').is_file()
            assert (directory / 'curves.png').is_file()
        print('Joint DMC/clone, checkpoint/best, resume and plot self-check passed')
    elif not arguments.dataset:
        raise ValueError('--dataset is required for training or --check_dataset')
    elif arguments.check_dataset:
        sample = VimeoSeptuplet(arguments.dataset, arguments.crop_size, arguments.group_size)[0]
        assert sample.shape == (6, 3, 256, 256), sample.shape
        print(f'Dataset check passed: {tuple(sample.shape)}')
    else:
        train(arguments)
