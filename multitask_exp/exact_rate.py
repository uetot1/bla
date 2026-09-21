"""DMC training forward whose rate is the exact cost of the rounded symbols.

DMC.forward_train (used by train_base.py and, until now, by every multitask_exp
run) prices the latents y and z on their *unrounded* values: for a residual r it
charges -log2 P(r - 0.5 < X < r + 0.5), a smooth surrogate. The coder, however,
transmits the integer round(r). For the small scales that dominate a low-rate
DCVC-RT latent the surrogate's gradient towards r = 0 is several times larger
than the real cost's (measured ~4x at scale 0.2, ~1.8x at 0.3), so it keeps
squeezing residuals that cost almost nothing once rounded.

The script that trained the paper checkpoint (the one embedded in
kaggle_train_random_qp_lambda_1_64.ipynb) instead prices the rounded symbols
(`gaussian_bits(yq, ...)`, `z_bits(z_hat, ...)`). This module reproduces that
forward using the model's own layers, so a warm start from the paper checkpoint
keeps optimising the objective it was trained with.
"""

import torch


def forward_train_exact(model, x, qp):
    """Same signature and DPB behaviour as DMC.forward_train; returns (x_hat, bpp)."""
    q_encoder = model.q_encoder[qp:qp + 1]
    q_decoder = model.q_decoder[qp:qp + 1]
    q_feature = model.q_feature[qp:qp + 1]
    q_recon = model.q_recon[qp:qp + 1]

    ref_feature = model.apply_feature_adaptor()
    ctx, ctx_t = model.feature_extractor(ref_feature, q_feature)
    y = model.encoder(x, ctx, q_encoder)
    z = model.hyper_encoder(model.pad_for_y(y))

    z_hat = model.quantize_ste(z).clamp(-128.0, 127.0)
    index = torch.full((z_hat.shape[0],), int(qp), dtype=torch.long, device=z_hat.device)
    z_prob = model.bit_estimator_z(z_hat + 0.5, index) - model.bit_estimator_z(z_hat - 0.5, index)
    z_bits = -torch.log2(z_prob.clamp_min(1e-9)).sum()

    common_params = model.res_prior_param_decoder(z_hat, ctx_t)
    y_scaled, q_dec, scales, means = model.separate_prior_for_video_encoding(common_params, y)
    batch, channel, height, width = y_scaled.shape
    mask0, mask1 = model.get_mask_2x(batch, channel, height, width, y_scaled.dtype, y_scaled.device)

    y_bits = 0
    y_hat = 0
    for stage, mask in enumerate((mask0, mask1)):
        residual = (y_scaled - means) * mask
        symbols = model.quantize_ste(residual).clamp(-128.0, 127.0)
        y_bits = y_bits + model.get_gaussian_bits(symbols, scales, mask)
        y_hat = y_hat + (symbols + means) * mask
        if stage == 0:
            scales, means = model.y_spatial_prior(torch.cat((y_hat, common_params), dim=1)).chunk(2, 1)

    x_hat, feature = model.get_recon_and_feature(y_hat * q_dec, ctx, q_decoder, q_recon)
    model.add_ref_frame(feature, x_hat)
    bpp = (y_bits + z_bits) / (x.shape[0] * x.shape[2] * x.shape[3])
    return x_hat, bpp
