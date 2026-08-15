import argparse
import io
import json
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm
from PIL import Image
from torchvision.transforms import ToPILImage, ToTensor
from torch import nn
from models.common import AutoShape, DetectMultiBackend
from utils.general import xywh2xyxy
from utils.metrics import ap_per_class, box_iou
from utils.torch_utils import select_device
from dcvc_rt.src.layers.cuda_inference import replicate_pad
from dcvc_rt.src.models.image_model import DMCI
from dcvc_rt.src.models.video_model import DMC
from dcvc_rt.src.utils.common import get_state_dict
from dcvc_rt.src.utils.stream_helper import SPSHelper, write_ip, write_sps
from dcvc_rt.src.utils.transforms import rgb2ycbcr, ycbcr2rgb
from machine_metrics import bd_rate_map, load_points
from svc_machine.feature_extractor import install_cloned_frontend
#=======================================================================================================================
def torch2img(x: torch.Tensor) -> Image.Image:
    return ToPILImage()(x.clamp_(0, 1).squeeze())


def load_labels(path, height, width, device):
    rows = [line.split() for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]
    if any(len(row) != 5 for row in rows):
        raise ValueError(f'Expected YOLO labels "class x y width height" in {path}')
    labels = torch.tensor([[float(value) for value in row] for row in rows], device=device)
    if not rows:
        return torch.empty((0, 5), device=device)
    labels[:, 1:] = xywh2xyxy(labels[:, 1:]) * torch.tensor(
        (width, height, width, height), device=device)
    return labels


def match_detections(detections, labels, iou_thresholds):
    correct = np.zeros((detections.shape[0], iou_thresholds.numel()), dtype=bool)
    if detections.shape[0] == 0 or labels.shape[0] == 0:
        return torch.tensor(correct, device=detections.device)
    iou = box_iou(labels[:, 1:], detections[:, :4])
    correct_class = labels[:, 0:1] == detections[:, 5]
    for index, threshold in enumerate(iou_thresholds):
        matches = torch.where((iou >= threshold) & correct_class)
        if matches[0].numel() == 0:
            continue
        matches = torch.cat((torch.stack(matches, 1), iou[matches[0], matches[1]][:, None]), 1).cpu().numpy()
        if matches.shape[0] > 1:
            matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
            matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        correct[matches[:, 1].astype(int), index] = True
    return torch.tensor(correct, device=detections.device)
#=======================================================================================================================
# Creating our proposed model
class CompressModel(nn.Module):#nn.Module
    """Basic Compress Model"""

    def __init__(self):
        super(CompressModel, self).__init__()
#=======================================================================================================================
class Pframe(CompressModel):
    INDEX_MAP = [0, 1, 0, 2, 0, 2, 0, 2]

    def __init__(self, model_path_i, model_path_p, qp, force_zero_thres=0.12, reset_interval=64):
        super(Pframe, self).__init__()
        self.if_model = DMCI().cuda()
        self.if_model.load_state_dict(get_state_dict(model_path_i))
        self.if_model.eval()
        self.if_model.update(force_zero_thres)
        self.if_model.half()

        self.p_model = DMC().cuda()
        self.p_model.load_state_dict(get_state_dict(model_path_p))
        self.p_model.eval()
        self.p_model.update(force_zero_thres)
        self.p_model.half()

        self.qp = qp
        self.last_qp = qp
        self.reset_interval = reset_interval

    @staticmethod
    def _prepare_frame(frame):
        height, width = frame.shape[-2:]
        padding_r, padding_b = DMCI.get_padding_size(height, width, 16)
        return replicate_pad(rgb2ycbcr(frame).half(), padding_b, padding_r)

    @staticmethod
    def _restore_frame(frame, height, width):
        return ycbcr2rgb(frame[:, :, :height, :width]).float().clamp(0, 1)

    #forward function for testing
    def forward_pair(self, ref_frame, coding_frame, p_order=1):
        del ref_frame  # DCVC-RT keeps the decoded reference in its DPB.
        use_ada_i = self.reset_interval > 0 and p_order % self.reset_interval == 1
        if use_ada_i:
            self.p_model.prepare_feature_adaptor_i(self.last_qp)
        curr_qp = self.p_model.shift_qp(self.qp, self.INDEX_MAP[p_order % 8])
        encoded = self.p_model.compress(self._prepare_frame(coding_frame), curr_qp)
        self.last_qp = curr_qp
        height, width = coding_frame.shape[-2:]
        reconstructed = self._restore_frame(encoded['x_hat'], height, width)
        return reconstructed, encoded['bit_stream'], curr_qp, int(use_ada_i)

    @torch.no_grad()
    def test(self, no_frames, inp_path, labels_path, prefix, out_path, gop, fps, img_size,
             save_frames=True):
        stats = []
        output_buffer = io.BytesIO()
        sps_helper = SPSHelper()
        iou_thresholds = torch.linspace(0.5, 0.95, 10, device='cuda')
        self.p_model.clear_dpb()
        self.p_model.set_curr_poc(0)
        self.last_qp = self.qp
        for idx in tqdm(range(0,no_frames)):
            filename = '%s/%s%03d.png' % (inp_path, prefix, idx)
            coding_frame = ToTensor()(Image.open(filename)).unsqueeze(0).cuda()
            height, width = coding_frame.shape[-2:]
            use_two_entropy_coders = height * width > 1280 * 720
            self.if_model.set_use_two_entropy_coders(use_two_entropy_coders)
            self.p_model.set_use_two_entropy_coders(use_two_entropy_coders)

            if idx%gop==0: # I-frames
                encoded = self.if_model.compress(self._prepare_frame(coding_frame), self.qp)
                bit_stream, curr_qp, use_ada_i = encoded['bit_stream'], self.qp, 0
                rec_ycbcr = encoded['x_hat']
                rec_frame = self._restore_frame(rec_ycbcr, height, width)
                self.p_model.clear_dpb()
                self.p_model.add_ref_frame(None, rec_ycbcr)
                self.last_qp = self.qp
                frame_idx = 1
            else: #P-frames
                rec_frame, bit_stream, curr_qp, use_ada_i = self.forward_pair(
                    ref_frame, coding_frame, frame_idx)
                rec_frame = rec_frame.clamp(0, 1)
                frame_idx = frame_idx + 1

            sps = {'height': height, 'width': width, 'ec_part': int(use_two_entropy_coders),
                   'use_ada_i': use_ada_i}
            sps_id, is_new_sps = sps_helper.get_sps_id(sps)
            sps['sps_id'] = sps_id
            if is_new_sps:
                write_sps(output_buffer, sps)
            write_ip(output_buffer, idx % gop == 0, sps_id, curr_qp, bit_stream)

            img = torch2img(rec_frame.detach().cpu())
            if save_frames:
                img.save('%s/%s%03d.png' % (out_path, prefix, idx))
            detections = self.feature_extractor(img, size=img_size).pred[0]
            labels = load_labels(Path(labels_path) / f'{prefix}{idx:03d}.txt', height, width,
                                 detections.device)
            correct = match_detections(detections, labels, iou_thresholds)
            stats.append((correct.cpu(), detections[:, 4].cpu(), detections[:, 5].cpu(),
                          labels[:, 0].cpu()))

            ref_frame = rec_frame

        stats = [torch.cat(values, 0).numpy() for values in zip(*stats)]
        if stats[3].size == 0:
            raise ValueError(f'No ground-truth objects found in {labels_path}')
        _, _, _, _, _, average_precision, _ = ap_per_class(
            *stats, names=self.feature_extractor.names)
        bitstream = output_buffer.getvalue()
        (Path(out_path) / f'{prefix}dcvc_rt.bin').write_bytes(bitstream)
        output_buffer.close()
        return {
            'qp': self.qp,
            'bitrate_kbps': len(bitstream) * 8 * fps / (1000 * no_frames),
            'map50': float(average_precision[:, 0].mean()),
            'map50_95': float(average_precision.mean()),
        }
#=======================================================================================================================
@torch.no_grad()
def encode_the_base_layer(model_path_i, model_path_p, qps, no_frames, inp_path, labels_path,
                          prefix, out_path, gop, fps, img_size, weights, map_metric,
                          anchor_path, result_path, force_zero_thres, reset_interval,
                          save_frames=True):
    if no_frames <= 0 or gop <= 0 or fps <= 0:
        raise ValueError('no_frames, gop and fps must be positive')
    if tuple(qps) != (0, 21, 42, 63):
        raise ValueError('BD-rate-mAP evaluation requires QPs 0 21 42 63 in this order')

    def paths_for_points(paths, name):
        if len(paths) == 1:
            return paths * len(qps)
        if len(paths) != len(qps):
            raise ValueError(f'{name} must contain either one path or one path per QP')
        return paths

    image_paths = paths_for_points(model_path_i, 'model_path_i')
    video_paths = paths_for_points(model_path_p, 'model_path_p')
    device = select_device('', batch_size=1)
    detector = DetectMultiBackend(
        weights=weights, device=device, dnn=False, fp16=False, fuse=False)
    original_frontend = {
        key: value.detach().clone()
        for key, value in detector.model.model[:5].state_dict().items()
    }
    feature_extractor = AutoShape(detector, verbose=False)
    feature_extractor.conf = 0.001
    feature_extractor.iou = 0.6

    points = []
    for qp, image_path, video_path in zip(qps, image_paths, video_paths):
        checkpoint = torch.load(video_path, map_location='cpu', weights_only=True)
        cloned_frontend = checkpoint.get('cloned_frontend_state_dict') \
            if isinstance(checkpoint, dict) else None
        install_cloned_frontend(detector, cloned_frontend or original_frontend)
        net = Pframe(image_path, video_path, qp, force_zero_thres, reset_interval).cuda()
        net.feature_extractor = feature_extractor
        net.eval()
        qp_out_path = Path(out_path) / f'qp_{qp}'
        qp_out_path.mkdir(parents=True, exist_ok=True)
        print(f'Generating and evaluating base frames at QP {qp}...')
        point = net.test(no_frames, inp_path, labels_path, prefix, qp_out_path, gop, fps,
                         img_size, save_frames)
        point['trained_cloned_frontend'] = cloned_frontend is not None
        points.append(point)
        print(f"QP {qp}: {point['bitrate_kbps']:.3f} kbps, "
              f"mAP@0.5={point['map50']:.6f}, mAP@0.5:0.95={point['map50_95']:.6f}")

    clone_flags = {point['trained_cloned_frontend'] for point in points}
    if len(clone_flags) != 1:
        raise ValueError('All four evaluation points must use the same cloned front-end policy')

    result = {'metric': map_metric, 'points': points}
    result['bd_rate_map_percent'] = bd_rate_map(load_points(anchor_path), points, map_metric)
    print(f"BD-rate-{map_metric}: {result['bd_rate_map_percent']:.3f}%")
    result_path = Path(result_path) if result_path else Path(out_path) / 'machine_metrics.json'
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2), encoding='utf-8')

    return result

#=======================================================================================================================
def parse_arguments():
    # Create an argument parser
    parser = argparse.ArgumentParser(description="Parse command-line arguments.")

    parser.add_argument(
        '--inp_path',
        type=str,
        default='./input',
    help = 'Path to the input directory (default: ./input)'
    )

    parser.add_argument(
        '--out_path',
        type=str,
        default='./out/base',
    help = 'Directory for reconstructed base frames and machine metrics'
    )

    parser.add_argument(
        '--prefix',
        type=str,
        required=True,
    help = 'prefix of the video frame names (e.g. Parkscene_)'
    )

    parser.add_argument(
        '--labels_path',
        type=str,
        required=True,
        help='Directory containing YOLO ground-truth labels matching the input frame names'
    )

    parser.add_argument(
        '--qps',
        type=int,
        nargs='+',
        default=(0, 21, 42, 63),
        help='DCVC-RT QPs used to form the machine-task rate-mAP curve'
    )

    parser.add_argument(
        '--model_path_i',
        type=str,
        nargs='+',
        default=('./checkpoints/dcvc_rt/cvpr2025_image.pth.tar',),
        help='One image checkpoint for all QPs, or one per QP'
    )

    parser.add_argument(
        '--model_path_p',
        type=str,
        nargs='+',
        default=('./checkpoints/dcvc_rt/cvpr2025_video.pth.tar',),
        help='One video checkpoint for all QPs, or one per QP'
    )

    parser.add_argument(
        '--force_zero_thres',
        type=float,
        default=0.12,
        help='DCVC-RT entropy threshold (default: 0.12)'
    )

    parser.add_argument(
        '--reset_interval',
        type=int,
        default=64,
        help='DCVC-RT feature reset interval (default: 64)'
    )

    parser.add_argument(
        '--gop',
        type=int,
        default=32,
        help='GOP or intra period size (default: 32)'
    )

    parser.add_argument(
        '--no_frames',
        type=int,
        default=100,
        help='Number of frames (default: 100)'
    )

    parser.add_argument(
        '--fps',
        type=float,
        required=True,
        help='Source frame rate used by the CTC bitrate formula'
    )

    parser.add_argument(
        '--weights',
        type=str,
        default='./yolov5s.pt',
        help='Machine-task detector checkpoint'
    )

    parser.add_argument(
        '--img_size',
        type=int,
        default=640,
        help='Detector inference size (default: 640)'
    )

    parser.add_argument(
        '--map_metric',
        choices=('map50', 'map50_95'),
        default='map50_95',
        help='mAP variant used for BD-rate-mAP (default: map50_95)'
    )

    parser.add_argument(
        '--anchor_path',
        type=str,
        required=True,
        help='JSON containing the VTM anchor bitrate-mAP points'
    )

    parser.add_argument(
        '--result_path',
        type=str,
        help='Output JSON path (default: OUT_PATH/machine_metrics.json)'
    )

    # Parse the arguments
    args = parser.parse_args()

    return args
#=======================================================================================================================
if __name__ == "__main__":
    args = parse_arguments()

    encode_the_base_layer(args.model_path_i, args.model_path_p, args.qps, args.no_frames,
                          args.inp_path, args.labels_path, args.prefix, args.out_path, args.gop,
                          args.fps, args.img_size, args.weights, args.map_metric, args.anchor_path,
                          args.result_path, args.force_zero_thres, args.reset_interval)


