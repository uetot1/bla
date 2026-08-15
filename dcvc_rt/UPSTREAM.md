# Upstream source

- Repository: https://github.com/microsoft/DCVC
- Path: `DCVC-family/DCVC-RT`
- Commit: `819c219`
- License: MIT (`LICENSE.txt` and `NOTICE.txt`)

Local integration changes: `DMC.compress()` returns its reconstructed frame for SVC evaluation. `DMCI` and `DMC` also expose differentiable training forwards using straight-through quantization and entropy-model rate estimates; training selects the native PyTorch layer paths so gradients are preserved. The original entropy-coded inference path remains unchanged.
