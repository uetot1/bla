import torch
from torch import nn

from dcvc_rt.src.utils.transforms import ycbcr2rgb
from svc_machine.feature_loss import feature_mse_loss


def freeze_batchnorm(module):
    """Keep every BatchNorm in eval mode with frozen affine parameters.

    Mirrors freeze_bn() in the script that trained the paper checkpoint
    (kaggle_train_random_qp_lambda_1_64.ipynb). In train mode, BatchNorm
    normalises with the statistics of a 2-clip batch, which makes the feature
    loss nearly blind to brightness, contrast and colour shifts that the frozen
    evaluation detector does see.
    """
    for layer in module.modules():
        if isinstance(layer, nn.modules.batchnorm._BatchNorm):
            layer.eval()
            for parameter in layer.parameters():
                parameter.requires_grad_(False)


class MultiTaskMachineSystem(nn.Module):
    """DCVC-RT DMC trained against a detection clone and a frozen segmentation model.

    L_t = R_t + lambda(q) * w_t * (alpha_det * D_det + alpha_seg * seg_scale * D_seg)
    A branch whose alpha is 0 is passed as None and skipped entirely. The
    clone's convolutions train; its BatchNorm layers stay frozen in eval mode.
    """

    def __init__(self, video_model, det_clone, seg_branch,
                 alpha_det, alpha_seg, seg_scale):
        super().__init__()
        if (alpha_det > 0) != (det_clone is not None):
            raise ValueError("det_clone must be given exactly when alpha_det > 0")
        if (alpha_seg > 0) != (seg_branch is not None):
            raise ValueError("seg_branch must be given exactly when alpha_seg > 0")
        self.video_model = video_model
        self.det_clone = det_clone
        self.seg_branch = seg_branch
        self.alpha_det = float(alpha_det)
        self.alpha_seg = float(alpha_seg)
        self.seg_scale = float(seg_scale)
        if self.det_clone is not None:
            freeze_batchnorm(self.det_clone)

    def train(self, mode=True):
        # nn.Module.train() would flip the clone's BatchNorm back to batch statistics.
        super().train(mode)
        if self.det_clone is not None:
            freeze_batchnorm(self.det_clone)
        return self

    def forward(self, ycbcr_frames, det_targets, seg_targets, qps, lambda_task,
                distortion_weights):
        terms, rates, det_distortions, seg_distortions = [], [], [], []
        for index, qp in enumerate(qps):
            reconstructed_ycbcr, rate = self.video_model.forward_train(
                ycbcr_frames[:, index], qp)
            reconstructed = ycbcr2rgb(reconstructed_ycbcr)
            task_term = reconstructed.new_zeros(())
            if self.det_clone is not None:
                det_distortion = feature_mse_loss(
                    self.det_clone(reconstructed), det_targets[:, index])
                det_distortions.append(det_distortion)
                task_term = task_term + self.alpha_det * det_distortion
            if self.seg_branch is not None:
                seg_distortion = feature_mse_loss(
                    self.seg_branch(reconstructed), seg_targets[:, index])
                seg_distortions.append(seg_distortion)
                task_term = task_term + self.alpha_seg * self.seg_scale * seg_distortion
            terms.append(rate + lambda_task * distortion_weights[index] * task_term)
            rates.append(rate)

        def mean_or_nan(values):
            return torch.stack(values).mean() if values else torch.tensor(float("nan"))

        return (
            torch.stack(terms).mean(),
            torch.stack(rates).mean(),
            mean_or_nan(det_distortions),
            mean_or_nan(seg_distortions),
        )
