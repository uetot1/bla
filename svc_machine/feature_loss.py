import torch
import torch.nn.functional as F


def feature_mse_loss(reconstructed_feature, target_feature):
    if reconstructed_feature.shape != target_feature.shape:
        raise ValueError('Reconstructed and target features must have the same shape')
    return F.mse_loss(reconstructed_feature, target_feature)


def machine_rate_distortion_loss(rates, feature_distortions, lambda_task,
                                 distortion_weights=None):
    if not rates or len(rates) != len(feature_distortions):
        raise ValueError('Rate and feature-distortion lists must have the same non-zero length')
    if distortion_weights is None:
        distortion_weights = [1.0] * len(rates)
    if len(distortion_weights) != len(rates):
        raise ValueError('Distortion weights must match the number of frames')
    return torch.stack([
        rate + lambda_task * weight * distortion
        for rate, distortion, weight in zip(
            rates, feature_distortions, distortion_weights)
    ]).mean()
