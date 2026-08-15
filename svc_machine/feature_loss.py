import torch
import torch.nn.functional as F


def feature_mse_loss(reconstructed_feature, target_feature):
    if reconstructed_feature.shape != target_feature.shape:
        raise ValueError('Reconstructed and target features must have the same shape')
    return F.mse_loss(reconstructed_feature, target_feature)


def machine_rate_distortion_loss(rates, feature_distortions, lambda_task):
    if not rates or len(rates) != len(feature_distortions):
        raise ValueError('Rate and feature-distortion lists must have the same non-zero length')
    return torch.stack([
        rate + lambda_task * distortion
        for rate, distortion in zip(rates, feature_distortions)
    ]).mean()
