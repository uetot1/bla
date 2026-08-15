import argparse
import json
import math
import random
import shutil
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.functional import to_tensor
from tqdm import tqdm

from dcvc_rt.src.models.image_model import DMCI
from dcvc_rt.src.models.video_model import DMC
from dcvc_rt.src.utils.common import get_state_dict
from dcvc_rt.src.utils.transforms import rgb2ycbcr, ycbcr2rgb
from models.experimental import attempt_load


LAMBDA_MIN = 2.0
LAMBDA_MAX = 16.0
INDEX_MAP = (0, 1, 0, 2, 0, 2, 0, 2)
YOLO_FEATURE_LAYERS = (17, 20, 23)
VALIDATION_QPS = (0, 21, 42, 63)


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


def make_yolo_feature_model(weights, device):
    yolo = attempt_load(weights, device=device, fuse=False)
    if len(yolo.model) <= max(YOLO_FEATURE_LAYERS):
        raise ValueError('The selected YOLO model does not expose layers 17, 20 and 23')
    yolo = yolo.to(device).eval()
    for parameter in yolo.parameters():
        parameter.requires_grad_(False)
    features = {}
    for index in YOLO_FEATURE_LAYERS:
        yolo.model[index].register_forward_hook(
            lambda _module, _inputs, output, index=index: features.__setitem__(index, output))
    return yolo, features


def save_checkpoint(path, epoch, mode, video_model, fixed_qp=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {'epoch': epoch, 'mode': mode, 'lambda_range': (LAMBDA_MIN, LAMBDA_MAX)}
    if fixed_qp is not None:
        metadata.update({'fixed_qp': fixed_qp, 'fixed_lambda': lambda_for_qp(fixed_qp)})
    torch.save({**metadata, 'state_dict': video_model.state_dict()}, path)


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


def train(args):
    if args.epochs <= 0 or args.learning_rate <= 0 or args.batch_size <= 0 or args.workers < 0:
        raise ValueError('epochs, learning_rate and batch_size must be positive; workers cannot be negative')
    if args.validation_interval <= 0:
        raise ValueError('validation_interval must be positive')
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA is required but is not available')
    device = torch.device(args.device)
    lambda_for_qp(args.fixed_qp)
    validation_config = load_validation_config(args.validation_config) \
        if args.validation_config else None
    if validation_config and device.type != 'cuda':
        raise ValueError('Real-bitstream validation requires a CUDA device')

    dataset = VimeoSeptuplet(args.dataset, args.crop_size, args.group_size)
    if len(dataset) < args.batch_size:
        raise ValueError('Dataset must contain at least batch_size sequences')
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.workers, pin_memory=device.type == 'cuda', drop_last=True)

    image_model = DMCI().to(device)
    image_model.load_state_dict(get_state_dict(args.model_path_i))
    video_model = DMC().to(device)
    video_model.load_state_dict(get_state_dict(args.model_path_p))
    image_model.eval()
    for parameter in image_model.parameters():
        parameter.requires_grad_(False)
    video_model.train()
    task_model, task_features = make_yolo_feature_model(args.weights, device)

    optimizer = torch.optim.Adam(video_model.parameters(), lr=args.learning_rate)
    tag = 'variable_rate' if args.mode == 'variable' else f'fixed_qp{args.fixed_qp}'
    save_dir = Path(args.save_dir)
    last_checkpoint = save_dir / f'video_{tag}_last.pth.tar'
    best_checkpoint = save_dir / f'video_{tag}_best.pth.tar'
    best_bd_rate = math.inf
    validation_history = []

    for epoch in range(1, args.epochs + 1):
        totals = {'loss': 0.0, 'rate': 0.0, 'task': 0.0}
        progress = tqdm(loader, desc=f'epoch {epoch}/{args.epochs}')
        for sequences in progress:
            base_qp = random.randint(0, DMCI.get_qp_num() - 1) \
                if args.mode == 'variable' else args.fixed_qp
            lambda_task = lambda_for_qp(base_qp)
            sequences = sequences.to(device, non_blocking=True)
            video_model.clear_dpb()
            video_model.set_curr_poc(0)
            optimizer.zero_grad(set_to_none=True)
            frame_losses = []
            frame_rates = []
            frame_tasks = []

            with torch.no_grad():
                reference_ycbcr = image_model.forward_reconstruction(
                    rgb2ycbcr(sequences[:, 0]), base_qp)
            video_model.add_ref_frame(None, reference_ycbcr)

            for frame_index in range(1, args.group_size):
                rgb = sequences[:, frame_index]
                ycbcr = rgb2ycbcr(rgb)
                current_qp = video_model.shift_qp(base_qp, INDEX_MAP[frame_index % 8])
                reconstructed_ycbcr, rate = video_model.forward_train(ycbcr, current_qp)

                reconstructed_rgb = ycbcr2rgb(reconstructed_ycbcr)
                with torch.no_grad():
                    task_model(rgb, cut_model=1, cutting_layer=23)
                    target_features = tuple(task_features[index].detach()
                                            for index in YOLO_FEATURE_LAYERS)
                task_model(reconstructed_rgb, cut_model=1, cutting_layer=23)
                reconstructed_features = tuple(task_features[index]
                                               for index in YOLO_FEATURE_LAYERS)
                task_features.clear()
                task_loss = sum(F.mse_loss(reconstructed, target)
                                for reconstructed, target in zip(
                                    reconstructed_features, target_features)) / 3
                frame_rates.append(rate)
                frame_tasks.append(task_loss)
                frame_losses.append(rate + lambda_task * task_loss)

            loss = torch.stack(frame_losses).mean()
            rate = torch.stack(frame_rates).mean()
            task_loss = torch.stack(frame_tasks).mean()
            loss.backward()
            optimizer.step()
            video_model.clear_dpb()

            totals['loss'] += loss.item()
            totals['rate'] += rate.item()
            totals['task'] += task_loss.item()
            batches = progress.n + 1
            progress.set_postfix(loss=totals['loss'] / batches,
                                 bpp=totals['rate'] / batches,
                                 feature_mse=totals['task'] / batches,
                                 qp=base_qp, lambda_task=f'{lambda_task:.3f}')

        save_checkpoint(last_checkpoint, epoch, args.mode, video_model,
                        args.fixed_qp if args.mode == 'fixed' else None)
        if validation_config and (epoch % args.validation_interval == 0 or epoch == args.epochs):
            del sequences, rgb, ycbcr, reference_ycbcr, reconstructed_ycbcr, reconstructed_rgb
            del target_features, reconstructed_features, frame_losses, frame_rates, frame_tasks
            del loss, rate, task_loss
            task_features.clear()
            for model in (image_model, video_model, task_model):
                model.cpu()
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
                for model in (image_model, video_model, task_model):
                    model.to(device)
                move_optimizer_state(optimizer, device)
                image_model.eval()
                video_model.train()
                task_model.eval()

            bd_rate = result['bd_rate_map_percent']
            if not math.isfinite(bd_rate):
                raise ValueError('Validation BD-rate-mAP is not finite')
            validation_history.append({'epoch': epoch, **result})
            (save_dir / f'{tag}_validation_history.json').write_text(
                json.dumps(validation_history, indent=2), encoding='utf-8')
            if bd_rate < best_bd_rate:
                best_bd_rate = bd_rate
                shutil.copy2(last_checkpoint, best_checkpoint)
                print(f'New best validation BD-rate-mAP: {bd_rate:.3f}%')


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
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--group_size', type=int, choices=(5,), default=5)
    parser.add_argument('--crop_size', type=int, choices=(256,), default=256)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--validation_config',
                        help='JSON config for periodic real-bitstream BD-rate-mAP validation')
    parser.add_argument('--validation_interval', type=int, default=1)
    parser.add_argument('--check_dataset', action='store_true',
                        help='Load one cropped five-frame group, report its shape, and exit')
    parser.add_argument('--self_check', action='store_true',
                        help='Check QP/lambda interpolation and exit')
    return parser.parse_args()


if __name__ == '__main__':
    random.seed(0)
    torch.manual_seed(0)
    arguments = parse_args()
    if arguments.self_check:
        expected = {0: 2.0, 21: 4.0, 42: 8.0, 63: 16.0}
        assert all(abs(lambda_for_qp(qp) - value) < 1e-9
                   for qp, value in expected.items())
        qp_shift = (0, 8, 4)
        assert [qp_shift[INDEX_MAP[index]] for index in range(5)] == [0, 8, 0, 4, 0]
        print('Variable-rate QP/lambda self-check passed')
    elif not arguments.dataset:
        raise ValueError('--dataset is required for training or --check_dataset')
    elif arguments.check_dataset:
        sample = VimeoSeptuplet(arguments.dataset, arguments.crop_size, arguments.group_size)[0]
        assert sample.shape == (5, 3, 256, 256), sample.shape
        print(f'Dataset check passed: {tuple(sample.shape)}')
    else:
        train(arguments)
