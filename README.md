# SVC machine base layer with DCVC-RT

This repository keeps the machine base-layer pipeline from *Learned Scalable Video Coding for Humans and Machines* and replaces its image/video codec with Microsoft's pretrained [DCVC-RT](https://github.com/microsoft/DCVC/tree/main/DCVC-family/DCVC-RT).

```text
input PNG -> DCVC-RT bitstream -> reconstructed PNG -> YOLOv5 -> mAP
          -> bitrate (kbps) + anchor VTM curve -> BD-rate-mAP
```

The enhancement layer and the PSNR, SSIM, MS-SSIM and reconstruction-MSE evaluation paths are removed. The remaining metrics follow `wg2n00231_r1.docx`: video rate is measured in kbps, object detection is measured by mAP, and the final comparison against the VTM anchor is BD-rate-mAP.

## Models

- `DMCI` with `cvpr2025_image.pth.tar` encodes each GOP's I-frame.
- `DMC` with `cvpr2025_video.pth.tar` encodes the following P-frames.
- `yolov5s.pt` supplies the frozen teacher and initializes the trainable cloned front-end (layers `0..4`). Its back-end (layers `5..23`) stays frozen for detection evaluation.

The official DCVC-RT source is vendored under `dcvc_rt/` and pinned in `dcvc_rt/UPSTREAM.md` with its license and notice.

The active project is split by responsibility:

```text
svc_machine/          SVC feature supervision and rate-task loss
dcvc_rt/              active DMCI/DMC codec implementation
train_base.py         five-frame SVC training schedule and orchestration
test_base.py          base-layer coding and BD-rate-mAP evaluation
legacy_svc_base/      original CANF/PWC/SDC base codec, reference only
models/ + utils/      required frozen YOLOv5 code
```

The enhancement layer, human-viewing path and unrelated YOLO utilities remain removed. The original SVC base-codec source is preserved under `legacy_svc_base/` for traceability but is not imported at runtime.

## Current training pipeline

![DCVC-RT machine base-layer training pipeline](training_pipeline.svg)

## Installation and checkpoints

DCVC-RT recommends Python 3.12, PyTorch 2.6 and CUDA 12.6.

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt

cd dcvc_rt/src/cpp
pip install .
cd ../layers/extensions/inference
pip install .
cd ../../../..
```

Download the two official checkpoints from the [Microsoft checkpoint folder](https://1drv.ms/f/c/2866592d5c55df8c/Esu0KJ-I2kxCjEP565ARx_YB88i0UnR6XnODqFcvZs4LcA?e=by8CO8):

```text
checkpoints/dcvc_rt/cvpr2025_image.pth.tar
checkpoints/dcvc_rt/cvpr2025_video.pth.tar
```

## Train the machine base layer

Prepare Vimeo-90K Septuplet in its standard layout:

```text
vimeo_septuplet/
  sep_trainlist.txt
  sequences/00001/0001/im1.png ... im7.png
```

The paper protocol trains four separate base-layer models, one for each fixed
`lambda_base`. Random QP, lambda interpolation and hierarchical QP offsets are
disabled in this mode. DCVC-RT needs a QP index, so this project maps the four
paper operating points to its four canonical pretrained-rate indices:

```text
QP       0    21    42    63
lambda   2     4     8    16
```

Each sample loads six consecutive Vimeo frames. The first frame initializes the
DCVC-RT DPB through frozen pretrained DMCI and is outside the loss. The following
five P-frames use the same fixed QP and form the paper's `N=5` sequence loss:

```text
L_base* = (1/5) * sum[t=1..5](R_DMC(t) + lambda_base * MSE(F4(x_t), Fclone4(xhat_t)))
```

Adam jointly optimizes pretrained `DMC` and cloned YOLO layers `0..4`. Pretrained
DMCI, the original YOLO teacher and YOLO backend `5..23` remain frozen. Defaults
match the paper: random `256x256` crop, `N=5`, global batch size 4, Adam at
`1e-6`, and 10 epochs. There is no pixel MSE, detection loss, PSNR, MS-SSIM or
enhancement-layer loss. Gradient clipping and DDP/resume remain reliability
features inherited from the current project.

First check the schedule and dataset (`6` tensors = one reference + five P-frames):

```bash
python train_base.py --self_check
python train_base.py --dataset /path/to/vimeo_septuplet --check_dataset
```

Train one model per paper lambda. Use a different output directory for every
operating point:

```bash
python train_base.py --dataset /path/to/vimeo_septuplet --paper_lambda 2  --save_dir checkpoints/paper_lambda2
python train_base.py --dataset /path/to/vimeo_septuplet --paper_lambda 4  --save_dir checkpoints/paper_lambda4
python train_base.py --dataset /path/to/vimeo_septuplet --paper_lambda 8  --save_dir checkpoints/paper_lambda8
python train_base.py --dataset /path/to/vimeo_septuplet --paper_lambda 16 --save_dir checkpoints/paper_lambda16
```

For DDP, keep `--batch_size 4` as the global batch size and choose a process count that divides four:

```bash
torchrun --standalone --nproc_per_node=2 train_base.py \
  --dataset /path/to/vimeo_septuplet \
  --paper_lambda 2 \
  --save_dir checkpoints/paper_lambda2
```

For the lambda-2 model the output names are:

```text
checkpoints/paper_lambda2/video_lambda2_qp0_last.pth.tar
checkpoints/paper_lambda2/video_lambda2_qp0_epoch_0001.pth
checkpoints/paper_lambda2/best_val_loss.pth
checkpoints/paper_lambda2/lambda2_qp0_training_history.json
checkpoints/paper_lambda2/lambda2_qp0_training_curves.png
checkpoints/paper_lambda2/lambda2_qp0_training_curves_epoch_0001.png
```

The history and Total Loss/BPP/Feature MSE chart are refreshed after every epoch;
an epoch-numbered chart is also retained. Every epoch evaluates the deterministic
centre crop from `sep_testlist.txt` at that model's fixed QP. The lowest held-out
loss is saved as `best_val_loss.pth`.

If training is interrupted, resume the same lambda and output directory.
`--epochs` is the final total, not the number of additional epochs:

```bash
python train_base.py \
  --dataset /path/to/vimeo_septuplet \
  --paper_lambda 2 \
  --save_dir checkpoints/paper_lambda2 \
  --epochs 10 \
  --resume
```

Use `--resume /path/to/checkpoint.pth.tar` for an explicit checkpoint. The checkpoint restores DMC, the trainable cloned front-end, joint Adam, Python/Torch/CUDA RNG state, histories, and the next epoch. DDP resume requires the same process count. An interruption inside an epoch repeats that incomplete epoch because checkpoints are committed only after complete epochs.

Older checkpoints cannot resume into schema 6 because the frame/QP schedule and
optimizer state differ. Start these four runs from the supplied pretrained
DCVC-RT and YOLO checkpoints. `--validation_config` is deliberately rejected in
paper mode: BD-rate selection must combine the four separately trained models,
not evaluate one fixed-rate model at four QPs.

## VTM anchor input

The CTC document supplies anchor QPs but not measured bitrate-mAP values. Run the VTM anchor and provide at least four measured points as JSON. mAP values must use the same `0..1` scale and the same variant as the proposal.

```json
{
  "points": [
    {"bitrate_kbps": 100.0, "map50_95": 0.20},
    {"bitrate_kbps": 200.0, "map50_95": 0.30},
    {"bitrate_kbps": 400.0, "map50_95": 0.40},
    {"bitrate_kbps": 800.0, "map50_95": 0.50}
  ]
}
```

Replace these example values with the VTM results for the same sequence, frames, FPS, detector and mAP definition.

## Run

Input images and YOLO labels must have matching names such as `Parkscene_000.png` and `Parkscene_000.txt`. Labels use normalized `class x_center y_center width height` format.

```bash
python test_base.py \
  --qps 0 21 42 63 \
  --model_path_i ./checkpoints/dcvc_rt/cvpr2025_image.pth.tar \
  --model_path_p \
    ./checkpoints/paper_lambda2/best_val_loss.pth \
    ./checkpoints/paper_lambda4/best_val_loss.pth \
    ./checkpoints/paper_lambda8/best_val_loss.pth \
    ./checkpoints/paper_lambda16/best_val_loss.pth \
  --inp_path ./input/ParkScene \
  --labels_path ./labels/ParkScene \
  --out_path ./out/ParkScene \
  --prefix Parkscene_ \
  --fps 24 \
  --no_frames 100 \
  --gop 32 \
  --anchor_path ./anchors/ParkScene.json
```

The four checkpoint paths correspond positionally to QPs `0,21,42,63`. Each
checkpoint supplies its own trained DMC and cloned front-end; frozen YOLO layers
`5..23` produce detections. Reconstructed PNG files and actual DCVC-RT `.bin`
streams are written under `out/ParkScene/qp_<QP>/`. The proposal curve and
`bd_rate_map_percent` are written to `machine_metrics.json`; a negative value
means bitrate saving over the anchor at equal mAP.

Use `--map_metric map50` when the anchor contains `map50`; the default is `map50_95`, corresponding to the average over IoU thresholds `0.50:0.05:0.95` defined by the CTC document.

This reproduces the paper's base-layer training schedule while retaining the
requested DCVC-RT core and pretrained weights. It cannot reproduce the paper's
published numbers exactly because the original paper uses LCCM-VC/CANF-VC, not
DCVC-RT; the lambda-to-QP mapping and frozen DMCI reference are the necessary
codec adaptation.

## HEVC x265 comparison using the reference evaluator

`evaluate_hevc.py` and the manifest/evaluation utilities are ported from
[`uetot1/DCVC-RT`](https://github.com/uetot1/DCVC-RT) at commit
`6cb7bcf6b30c3c51f712fc14541302740c603a3c`. The evaluator reads the repository's
native manifest schema (`sequences`, not a top-level `prefix`), encodes an RGB
source as BT.709 YUV444 10-bit Low-Delay P, evaluates every decoded frame with
the frozen YOLO detector, counts complete HEVC bitstreams, and resumes after
each completed sequence.

```bash
python evaluate_hevc.py \
  --data-dir /kaggle/input/class-d/vcm_eval \
  --dataset-manifest /kaggle/input/class-d/vcm_eval/manifest.json \
  --x265-encoder x265 \
  --qps 22 27 32 37 42 47 \
  --chroma-format 444 \
  --bit-depth 10 \
  --preset medium \
  --yolov5-weights ./yolov5s.pt \
  --resume \
  --keep-progress-checkpoint
```

Evaluate paper mode with the four checkpoints in QP order `0,21,42,63`:

```bash
python evaluate_vcm.py --mode codec \
  --data-dir /kaggle/input/class-d/vcm_eval \
  --dataset-manifest /kaggle/input/class-d/vcm_eval/manifest.json \
  --image-ckpt /kaggle/input/pre-trained-model/cvpr2025_image.pth.tar \
  --video-ckpt \
    /kaggle/working/checkpoints/paper_lambda2/best_val_loss.pth \
    /kaggle/working/checkpoints/paper_lambda4/best_val_loss.pth \
    /kaggle/working/checkpoints/paper_lambda8/best_val_loss.pth \
    /kaggle/working/checkpoints/paper_lambda16/best_val_loss.pth \
  --qps 0 21 42 63 \
  --reset-interval 64 \
  --force-zero-thres 0.12 \
  --yolov5-weights ./yolov5s.pt \
  --method-name svc_dcvc_rt_paper
```

Finally compute both BD-rate-mAP values and create two separate plots:

```bash
python evaluate_vcm.py --mode bdrate \
  --anchor-results output/hevc_evaluation/hevc_x265_ldp_rgb444_10bit_results.json \
  --candidate-results output/evaluation/dcvc_rt_svc_base_epoch20_results.json \
  --rate actual_bpp \
  --metric map5095 \
  --output-dir output/comparison_x265
```

The plots are `rd_curve_actual_bpp_map50.png` and
`rd_curve_actual_bpp_map5095.png`; `bd_rate_map.json` contains numeric results
for both metrics.
