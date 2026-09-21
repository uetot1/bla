import torch
from torch import nn

from dcvc_rt.src.utils.transforms import ycbcr2rgb
from svc_machine.feature_loss import feature_mse_loss

from multitask_exp.exact_rate import forward_train_exact


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
    """DCVC-RT DMC trained against a detection branch and a segmentation branch.

    L_t = R_t + lambda(q) * w_t * (alpha_det * D_det + alpha_seg * seg_scale * D_seg)
    A branch whose alpha is 0 is passed as None and skipped entirely.

    Detection can be supervised two ways, mutually exclusive:
    - det_clone: a trainable copy of the frontend (BatchNorm frozen in eval mode,
      convolutions trainable), the recipe of the paper's training script.
    - det_frozen: a FrozenYoloFeature, identical in kind to the segmentation
      branch -- no trainable parameters at all.

    Measured on SFU Class C/D with real bitstreams, the two are indistinguishable
    in detection quality (R0b ~ R3, R2b ~ R4): a drifting clone is NOT what breaks
    detection in the continued-training runs. What predicts the damage is the
    weight on the detection feature loss, and the most concrete recipe difference
    left from the paper's script is the rate term -- see rate_mode below and
    multitask_exp/exact_rate.py.
    """

    def __init__(self, video_model, det_clone, seg_branch,
                 alpha_det, alpha_seg, seg_scale, det_frozen=None, rate_mode="surrogate"):
        super().__init__()
        if rate_mode not in ("surrogate", "exact"):
            raise ValueError(f"unknown rate_mode {rate_mode!r}")
        if det_clone is not None and det_frozen is not None:
            raise ValueError("det_clone and det_frozen are mutually exclusive")
        if (alpha_det > 0) != (det_clone is not None or det_frozen is not None):
            raise ValueError("exactly one of det_clone/det_frozen must be given when alpha_det > 0")
        if (alpha_seg > 0) != (seg_branch is not None):
            raise ValueError("seg_branch must be given exactly when alpha_seg > 0")
        self.video_model = video_model
        self.det_clone = det_clone
        self.det_frozen = det_frozen
        self.seg_branch = seg_branch
        self.alpha_det = float(alpha_det)
        self.alpha_seg = float(alpha_seg)
        self.seg_scale = float(seg_scale)
        # "surrogate": DMC.forward_train prices the unrounded latents (what train_base.py and
        # R0-R4 used). "exact": prices the rounded symbols like the paper's training script.
        self.rate_mode = rate_mode
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
            if self.rate_mode == "exact":
                reconstructed_ycbcr, rate = forward_train_exact(
                    self.video_model, ycbcr_frames[:, index], qp)
            else:
                reconstructed_ycbcr, rate = self.video_model.forward_train(
                    ycbcr_frames[:, index], qp)
            reconstructed = ycbcr2rgb(reconstructed_ycbcr)
            task_term = reconstructed.new_zeros(())
            det_module = self.det_clone if self.det_clone is not None else self.det_frozen
            if det_module is not None:
                det_distortion = feature_mse_loss(
                    det_module(reconstructed), det_targets[:, index])
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
