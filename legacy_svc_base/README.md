# Original SVC base-codec reference

This directory preserves the original base-layer codec source from Git commit
`c32dd98` before DCVC-RT replaced it. It contains the original `Pframe`, CANF
motion/residual coders, PWCNet motion estimation, SDCNet motion extrapolation,
refinement network, entropy/context models, utility kernels and base configs.

It is reference-only and is deliberately not imported by the active project.
The active codec is `dcvc_rt/`; the SVC machine-task feature supervision and
rate-task objective are implemented in `svc_machine/`.

The upstream repository exposed base-layer evaluation in `test_base.py`, but
did not include the paper's complete machine-task training loop. Its `loss.py`
contains PSNR/YUV420 helpers, not the feature-rate training objective. The
active objective therefore lives in `svc_machine/feature_loss.py`.

Do not add this directory to `PYTHONPATH` unless intentionally reproducing the
legacy codec. Some original optional CUDA operators and checkpoint weights are
not included.
