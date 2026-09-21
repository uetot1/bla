"""Verbatim copy of the P-frame training forward from the paper's training script.

Source: the script embedded in kaggle_train_random_qp_lambda_1_64.ipynb (class
VCMSystem.p_frame_forward and its helpers), the one that produced the paper
checkpoint. Kept unchanged, as an oracle: multitask_exp.exact_rate must agree with
it to floating-point precision (checked in train_multitask --self_check).
Do not "clean this up"; its only value is being the original.
"""

import math

import torch


def ste_round(x):
    return x + (torch.round(x) - x).detach()


def normal_cdf(x):
    return 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def gaussian_bits(symbols, scales, mask):
    # Match DCVC-RT GaussianEncoder coding range: scale_min=.11, scale_max=16.
    with torch.autocast(device_type='cuda', enabled=False):
        s = scales.float().clamp(0.11, 16.0)
        y = symbols.float()
        upper = normal_cdf((y + 0.5) / s)
        lower = normal_cdf((y - 0.5) / s)
        prob = (upper - lower).clamp_min(1e-9)
        bits = -torch.log2(prob)
        return bits * mask.float()


def z_bits(model, z_hat, qp):
    B = z_hat.shape[0]
    idx = torch.full((B,), int(qp), dtype=torch.long, device=z_hat.device)
    with torch.autocast(device_type='cuda', enabled=False):
        z = z_hat.float()
        upper = model.bit_estimator_z(z + 0.5, idx)
        lower = model.bit_estimator_z(z - 0.5, idx)
        prob = (upper - lower).clamp_min(1e-9)
        return -torch.log2(prob)


def masked_quant_train(y, scales, means, mask):
    scales_hat = scales * mask
    means_hat = means * mask
    y_res = (y - means_hat) * mask
    y_q = ste_round(y_res).clamp(-128.0, 127.0)
    y_hat = y_q + means_hat
    return y_q, y_hat, scales_hat


def p_frame_forward(m, x, qp):
    q_encoder = m.q_encoder[qp:qp+1]
    q_decoder = m.q_decoder[qp:qp+1]
    q_feature = m.q_feature[qp:qp+1]
    q_recon = m.q_recon[qp:qp+1]

    ref_feature = m.apply_feature_adaptor()
    ctx, ctx_t = m.feature_extractor(ref_feature, q_feature)
    y = m.encoder(x, ctx, q_encoder)

    hyper_inp = m.pad_for_y(y)
    z = m.hyper_encoder(hyper_inp)
    z_hat = ste_round(z).clamp(-128.0, 127.0)

    common_params = m.res_prior_param_decoder(z_hat, ctx_t)
    y_scaled, q_dec, scales, means = m.separate_prior_for_video_encoding(common_params, y)
    B, C, H_y, W_y = y_scaled.shape
    mask0, mask1 = m.get_mask_2x(B, C, H_y, W_y, y_scaled.dtype, y_scaled.device)

    yq0, yh0, s0 = masked_quant_train(y_scaled, scales, means, mask0)
    cat_params = torch.cat((yh0, common_params), dim=1)
    scales1, means1 = m.y_spatial_prior(cat_params).chunk(2, 1)
    yq1, yh1, s1 = masked_quant_train(y_scaled, scales1, means1, mask1)

    y_hat = (yh0 + yh1) * q_dec
    x_hat, feature = m.get_recon_and_feature(y_hat, ctx, q_decoder, q_recon)

    by0 = gaussian_bits(yq0, s0, mask0).flatten(1).sum(1)
    by1 = gaussian_bits(yq1, s1, mask1).flatten(1).sum(1)
    bz = z_bits(m, z_hat, qp).flatten(1).sum(1)
    pixel_num = x.shape[-2] * x.shape[-1]
    bpp = (by0 + by1 + bz) / pixel_num

    # Keep temporal graph across P frames (BPTT over this short clip).
    m.add_ref_frame(feature=feature, frame=x_hat)
    return x_hat, bpp
