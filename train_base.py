import argparse
import json
import math
import os
import random
import shutil
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


class VimeoSeptuplet(Dataset):
    def __init__(self, root, crop_size=256, group_size=5):
        self.root = Path(root)
        self.sequence_root = self.root / 'sequences'
        list_path = self.root / 'sep_trainlist.txt'
        if not self.sequence_root.is_dir() or not list_path.is_file():
            raise FileNotFoundError(
                f'{self.root} must contain sequences/ and sep_trainlist.txt')
        self.sequences = [line.strip() for line in list_path.read_text().splitlines() if line.strip()]
        if not self.sequences:
            raise ValueError(f'No Vimeo-90K sequences listed in {list_path}')
        self.crop_size = crop_size
        self.group_size = group_size

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        start = random.randint(1, 8 - self.group_size)
        folder = self.sequence_root / self.sequences[index]
        frames = []
        for frame_index in range(start, start + self.group_size):
            path = folder / f'im{frame_index}.png'
            with Image.open(path) as image:
                frames.append(to_tensor(image.convert('RGB')))
        height, width = frames[0].shape[-2:]
        if any(frame.shape[-2:] != (height, width) for frame in frames):
            raise ValueError(f'Frame sizes differ in {folder}')
        if height < self.crop_size or width < self.crop_size:
            raise ValueError(f'{folder} is smaller than {self.crop_size}x{self.crop_size}')
        top = random.randint(0, height - self.crop_size)
        left = random.randint(0, width - self.crop_size)
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
        axis.plot(epochs, [item[key] for item in history], marker='o')
        axis.set(title=title, xlabel='Epoch', ylabel=title)
        axis.grid(True, alpha=0.3)
    figure.tight_layout()
    figure.savefig(plot_path, dpi=150)
    plt.close(figure)


def save_checkpoint(path, epoch, mode, system, optimizer, training_history,
                    validation_history, best_bd_rate, rng_states, world_size,
                    fixed_qp=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + '.tmp')
    metadata = {'epoch': epoch, 'mode': mode, 'lambda_range': (LAMBDA_MIN, LAMBDA_MAX)}
    if fixed_qp is not None:
        metadata.update({'fixed_qp': fixed_qp, 'fixed_lambda': lambda_for_qp(fixed_qp)})
    torch.save({
        **metadata,
        'schema_version': 2,
        'state_dict': system.video_model.state_dict(),
        'cloned_frontend_state_dict': system.cloned_frontend.state_dict(),
        'optimizer': optimizer.state_dict(),
        'training_history': training_history,
        'validation_history': validation_history,
        'best_bd_rate_map_percent': best_bd_rate,
        'rng_states': rng_states,
        'world_size': world_size,
    }, temporary_path)
    temporary_path.replace(path)


def resume_training(path, mode, fixed_qp, system, optimizer, world_size):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    required = {
        'schema_version', 'epoch', 'mode', 'lambda_range', 'state_dict',
        'cloned_frontend_state_dict',
        'optimizer', 'rng_states', 'world_size',
    }
    missing = sorted(required - checkpoint.keys())
    if missing:
        raise ValueError(f'Resume checkpoint is missing: {", ".join(missing)}')
    if checkpoint['schema_version'] != 2:
        raise ValueError('Unsupported resume checkpoint schema')
    if checkpoint['mode'] != mode:
        raise ValueError(f'Resume checkpoint mode is {checkpoint["mode"]}, not {mode}')
    if tuple(checkpoint['lambda_range']) != (LAMBDA_MIN, LAMBDA_MAX):
        raise ValueError('Resume checkpoint uses a different lambda range')
    if mode == 'fixed' and checkpoint.get('fixed_qp') != fixed_qp:
        raise ValueError('Resume checkpoint uses a different fixed QP')
    if checkpoint['world_size'] != world_size:
        raise ValueError('Resume must use the same number of distributed processes')
    system.video_model.load_state_dict(checkpoint['state_dict'])
    system.cloned_frontend.load_state_dict(checkpoint['cloned_frontend_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return checkpoint


def move_optimizer_state(optimizer, device):
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def load_validation_config(path):
    config = json.loads(Path(path).read_text(encoding='utf-8'))
    required = {'inp_path', 'labels_path', 'prefix', 'fps', 'no_frames', 'gop', 'anchor_path'}
    missing = sorted(required - config.keys())
    if missing:
        raise ValueError(f'Missing validation config fields: {", ".join(missing)}')
    return config


def validate_checkpoint(args, checkpoint_path, output_dir, config):
    from test_base import encode_the_base_layer
    return encode_the_base_layer(
        (args.model_path_i,), (str(checkpoint_path),), VALIDATION_QPS,
        config['no_frames'], config['inp_path'], config['labels_path'], config['prefix'],
        output_dir, config['gop'], config['fps'], config.get('img_size', 640), args.weights,
        config.get('map_metric', 'map50_95'), config['anchor_path'],
        str(Path(output_dir) / 'latest_metrics.json'),
        config.get('force_zero_thres', 0.12), config.get('reset_interval', 64),
        save_frames=False)


def train_worker(args, device, rank, world_size, local_rank):
    if args.epochs <= 0 or args.learning_rate <= 0 or args.batch_size <= 0 or args.workers < 0:
        raise ValueError('epochs, learning_rate and batch_size must be positive; workers cannot be negative')
    if args.validation_interval <= 0 or args.grad_clip <= 0:
        raise ValueError('validation_interval and grad_clip must be positive')
    if args.batch_size % world_size:
        raise ValueError('Global batch_size must be divisible by the distributed world size')
    if world_size > 1 and args.validation_config:
        raise ValueError(
            'Periodic actual-bitstream validation is single-process; run test_base.py '
            'on saved DDP checkpoints')
    random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    if device.type == 'cuda':
        torch.cuda.manual_seed(args.seed + rank)
    lambda_for_qp(args.fixed_qp)
    validation_config = load_validation_config(args.validation_config) \
        if args.validation_config else None
    if validation_config and device.type != 'cuda':
        raise ValueError('Real-bitstream validation requires a CUDA device')

    dataset = VimeoSeptuplet(args.dataset, args.crop_size, args.group_size)
    if len(dataset) < args.batch_size:
        raise ValueError('Dataset must contain at least batch_size sequences')
    sampler = DistributedSampler(dataset, shuffle=True) if world_size > 1 else None
    loader = DataLoader(dataset, batch_size=args.batch_size // world_size,
                        shuffle=sampler is None, sampler=sampler,
                        num_workers=args.workers, pin_memory=device.type == 'cuda', drop_last=True)

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

    optimizer = torch.optim.Adam(system.parameters(), lr=args.learning_rate)
    tag = 'variable_rate' if args.mode == 'variable' else f'fixed_qp{args.fixed_qp}'
    save_dir = Path(args.save_dir)
    last_checkpoint = save_dir / f'video_{tag}_last.pth.tar'
    best_checkpoint = save_dir / f'video_{tag}_best.pth.tar'
    history_path = save_dir / f'{tag}_training_history.json'
    plot_path = save_dir / f'{tag}_training_curves.png'
    start_epoch = 1
    training_history = []
    validation_history = []
    best_bd_rate = math.inf

    if args.resume:
        resume_path = last_checkpoint if args.resume == 'auto' else Path(args.resume)
        if not resume_path.is_file():
            raise FileNotFoundError(f'Resume checkpoint not found: {resume_path}')
        checkpoint = resume_training(
            resume_path, args.mode, args.fixed_qp, system, optimizer, world_size)
        move_optimizer_state(optimizer, device)
        restore_rng_state(checkpoint['rng_states'], rank, device)
        start_epoch = checkpoint['epoch'] + 1
        training_history = checkpoint.get('training_history', [])
        validation_history = checkpoint.get('validation_history', [])
        best_bd_rate = checkpoint.get('best_bd_rate_map_percent', math.inf)
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
            lambda_task = lambda_for_qp(base_qp)
            sequences = sequences.to(device, non_blocking=True)
            system.video_model.clear_dpb()
            system.video_model.set_curr_poc(0)
            optimizer.zero_grad(set_to_none=True)

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
            qps = tuple(system.video_model.shift_qp(
                base_qp, INDEX_MAP[frame_index % 8])
                for frame_index in range(1, args.group_size))
            loss, rate, task_loss = model(
                ycbcr_frames, target_features, qps, lambda_task)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                system.parameters(), args.grad_clip, error_if_nonfinite=True)
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
        training_history.append({
            'epoch': epoch,
            'total_loss': reduced[0].item() / global_batches,
            'bpp': reduced[1].item() / global_batches,
            'feature_mse': reduced[2].item() / global_batches,
            'grad_norm': reduced[3].item() / global_batches,
        })
        if rank == 0:
            save_training_outputs(training_history, history_path, plot_path)
        fixed_qp = args.fixed_qp if args.mode == 'fixed' else None
        validation_due = validation_config and (
            epoch % args.validation_interval == 0 or epoch == args.epochs)
        if validation_due and rank == 0:
            rng_states = [capture_rng_state(device)]
            save_checkpoint(last_checkpoint, epoch, args.mode, system, optimizer,
                            training_history, validation_history, best_bd_rate,
                            rng_states, world_size, fixed_qp)
        improved = False
        if validation_due:
            del sequences, rgb_frames, ycbcr_frames, reference_ycbcr, target_features, qps
            del loss, rate, task_loss, grad_norm
            for current_model in (image_model, system, teacher):
                current_model.cpu()
            move_optimizer_state(optimizer, 'cpu')
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            try:
                result = validate_checkpoint(
                    args, last_checkpoint, str(save_dir / f'validation_{tag}'),
                    validation_config)
            finally:
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                for current_model in (image_model, system, teacher):
                    current_model.to(device)
                move_optimizer_state(optimizer, device)
                image_model.eval()
                system.train()
                teacher.eval()

            bd_rate = result['bd_rate_map_percent']
            if not math.isfinite(bd_rate):
                raise ValueError('Validation BD-rate-mAP is not finite')
            validation_history.append({'epoch': epoch, **result})
            (save_dir / f'{tag}_validation_history.json').write_text(
                json.dumps(validation_history, indent=2), encoding='utf-8')
            if bd_rate < best_bd_rate:
                best_bd_rate = bd_rate
                improved = True

        rng_states = gather_rng_states(device, world_size)
        if rank == 0:
            save_checkpoint(last_checkpoint, epoch, args.mode, system, optimizer,
                            training_history, validation_history, best_bd_rate,
                            rng_states, world_size, fixed_qp)
            if improved:
                shutil.copy2(last_checkpoint, best_checkpoint)
                print(f'New best validation BD-rate-mAP: {best_bd_rate:.3f}%')
        if world_size > 1:
            dist.barrier()


def train(args):
    device, rank, world_size, local_rank = setup_distributed(args.device)
    try:
        train_worker(args, device, rank, world_size, local_rank)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def parse_args():
    parser = argparse.ArgumentParser(description='Train the variable-rate machine-task DMC')
    parser.add_argument('--dataset', help='Vimeo-90K Septuplet root')
    parser.add_argument('--mode', choices=('variable', 'fixed'), default='variable')
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
    parser.add_argument('--group_size', type=int, choices=(5,), default=5)
    parser.add_argument('--crop_size', type=int, choices=(256,), default=256)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--validation_config',
                        help='JSON config for periodic real-bitstream BD-rate-mAP validation')
    parser.add_argument('--validation_interval', type=int, default=1)
    parser.add_argument('--resume', nargs='?', const='auto',
                        help='Resume at the next epoch; omit PATH to use this mode last checkpoint')
    parser.add_argument('--check_dataset', action='store_true',
                        help='Load one cropped five-frame group, report its shape, and exit')
    parser.add_argument('--self_check', action='store_true',
                        help='Check QP/lambda interpolation and exit')
    return parser.parse_args()


if __name__ == '__main__':
    arguments = parse_args()
    if arguments.self_check:
        from tempfile import TemporaryDirectory

        expected = {0: 2.0, 21: 4.0, 42: 8.0, 63: 16.0}
        assert all(abs(lambda_for_qp(qp) - value) < 1e-9
                   for qp, value in expected.items())
        qp_shift = (0, 8, 4)
        assert [qp_shift[INDEX_MAP[index]] for index in range(5)] == [0, 8, 0, 4, 0]
        target = torch.tensor([0.0, 1.0, 2.0])
        reconstructed = torch.tensor([1.0, 0.0, 0.0])
        distortion = feature_mse_loss(reconstructed, target)
        assert distortion.item() == 2.0
        assert machine_rate_distortion_loss(
            [torch.tensor(1.0)], [distortion], 2.0).item() == 5.0
        with TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            history = [{'epoch': 1, 'total_loss': 1.0, 'bpp': 0.5, 'feature_mse': 0.25}]
            system = torch.nn.Module()
            system.video_model = torch.nn.Linear(1, 1)
            system.cloned_frontend = torch.nn.Linear(1, 1)
            optimizer = torch.optim.Adam(system.parameters())
            checkpoint_path = directory / 'last.pth.tar'
            save_training_outputs(history, directory / 'history.json', directory / 'curves.png')
            save_checkpoint(checkpoint_path, 1, 'variable', system, optimizer,
                            history, [], math.inf, [capture_rng_state(torch.device('cpu'))], 1)
            resumed_system = torch.nn.Module()
            resumed_system.video_model = torch.nn.Linear(1, 1)
            resumed_system.cloned_frontend = torch.nn.Linear(1, 1)
            resumed_optimizer = torch.optim.Adam(resumed_system.parameters())
            checkpoint = resume_training(
                checkpoint_path, 'variable', 42, resumed_system, resumed_optimizer, 1)
            assert checkpoint['epoch'] == 1
            assert (directory / 'history.json').is_file()
            assert (directory / 'curves.png').is_file()
        print('Variable-rate, checkpoint/resume and plot self-check passed')
    elif not arguments.dataset:
        raise ValueError('--dataset is required for training or --check_dataset')
    elif arguments.check_dataset:
        sample = VimeoSeptuplet(arguments.dataset, arguments.crop_size, arguments.group_size)[0]
        assert sample.shape == (5, 3, 256, 256), sample.shape
        print(f'Dataset check passed: {tuple(sample.shape)}')
    else:
        train(arguments)
