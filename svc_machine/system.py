import torch
from torch import nn

from dcvc_rt.src.utils.transforms import ycbcr2rgb
from svc_machine.feature_loss import feature_mse_loss, machine_rate_distortion_loss


class MachineBaseSystem(nn.Module):
    """Jointly trainable DCVC-RT DMC and SVC cloned YOLO front-end."""

    def __init__(self, video_model, cloned_frontend):
        super().__init__()
        self.video_model = video_model
        self.cloned_frontend = cloned_frontend

    def forward(self, ycbcr_frames, target_features, qps, lambda_task):
        rates = []
        distortions = []
        for frame_index, qp in enumerate(qps):
            reconstructed_ycbcr, rate = self.video_model.forward_train(
                ycbcr_frames[:, frame_index], qp)
            reconstructed_feature = self.cloned_frontend(ycbcr2rgb(reconstructed_ycbcr))
            distortions.append(feature_mse_loss(
                reconstructed_feature, target_features[:, frame_index]))
            rates.append(rate)
        return (
            machine_rate_distortion_loss(rates, distortions, lambda_task),
            torch.stack(rates).mean(),
            torch.stack(distortions).mean(),
        )
