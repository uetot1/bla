# SVC cho machine task với DCVC-RT

Đây là phiên bản machine base-layer của bài *Learned Scalable Video Coding for
Humans and Machines*. Codec base gốc của bài báo đã được thay bằng
[DCVC-RT](https://github.com/microsoft/DCVC/tree/main/DCVC-family/DCVC-RT) và
pretrained chính thức của Microsoft.

Mục tiêu là nén video để YOLOv5 vẫn phát hiện đối tượng tốt với bitrate thấp.
Phần enhancement layer, human-viewing bitstream, PSNR, SSIM và MS-SSIM không
còn được dùng.

```text
Frame RGB
  │
  ├─ DMCI (frame tham chiếu đầu, frozen)
  └─ DMC / DCVC-RT (P-frame, trainable) ──> bitstream
                                             │
Frame tái tạo ──> YOLO clone 0..4 ──> feature tái tạo
                                      │
Frame gốc ─────> YOLO teacher 0..4 ──> feature mục tiêu
                                      │
                    Rate + λ × Feature MSE
                                      │
                       cập nhật DMC và YOLO clone 0..4
```

Ở lúc đánh giá, frame tái tạo đi qua YOLOv5 để lấy `mAP50` và `mAP50:95`.
Bitrate thực tế được lấy từ bitstream. Kết quả cuối là đường rate–mAP và
BD-rate-mAP so với HEVC/x265.

## Thành phần chính

| Thành phần | Trạng thái | Vai trò |
|---|---|---|
| `DMCI` | Frozen | Tạo frame tham chiếu đầu từ pretrained image codec. |
| `DMC` | Trainable | Nén/giải nén các P-frame của DCVC-RT. |
| YOLO teacher | Frozen | Trích feature mục tiêu ở layer 4 từ ảnh gốc. |
| YOLO clone layer 0..4 | Trainable | Trích feature từ ảnh tái tạo; được tối ưu cùng DMC. |
| YOLO backend layer 5..23 | Frozen | Tạo detection để tính mAP khi đánh giá. |

> Trạng thái trên mô tả **code hiện tại**. Teacher không nhận gradient; clone
> nhận gradient cùng DMC.

Cấu trúc thư mục quan trọng:

```text
train_base.py          train, validation loss, resume, biểu đồ
evaluate_vcm.py        mã hóa bitstream DCVC-RT, mAP và BD-rate
evaluate_hevc.py       HEVC/x265 anchor, mAP và bitrate thực
svc_machine/           feature loss và hệ thống train machine task
dcvc_rt/               mã nguồn DCVC-RT
models/, utils/        YOLOv5 cần cho teacher/detector
legacy_svc_base/       codec SVC cũ, chỉ để tham khảo; không chạy runtime
```

## Hàm loss và lịch train

Một sample gồm sáu frame Vimeo-90K liên tiếp:

- Frame đầu: chỉ tạo reference trong DPB bằng DMCI, không tính loss.
- Năm frame sau: là P-frame có loss (`N = 5`).

Với từng P-frame `t`:

```text
L(t) = R(t) + λtask × MSE(Fteacher(x_t), Fclone(xhat_t))
Lgroup = (1/5) × Σ L(t)
```

- `R(t)`: estimated rate/BPP từ entropy model của DMC khi train.
- `Fteacher(x_t)`: feature YOLO teacher ở layer 4 của frame gốc.
- `Fclone(xhat_t)`: feature YOLO clone ở layer 4 của frame tái tạo.
- Không có pixel MSE, PSNR, MS-SSIM hoặc detection loss.

Mỗi lần train tạo **một model với λ cố định**. Có bốn model riêng:
`λtask ∈ {2, 4, 8, 16}`. Trong mọi iteration, base QP được lấy ngẫu nhiên đều
từ `0..63`; λ không được nội suy từ QP. Lịch QP phân cấp nội bộ của DCVC-RT vẫn
được bật.

## Cài môi trường

Khuyến nghị: Python 3.12, PyTorch 2.6, CUDA 12.6.

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt

# Bắt buộc để mã hóa bitstream DCVC-RT thực tế khi evaluate_vcm.py
cd dcvc_rt/src/cpp
pip install .
cd ../../..

# Tùy chọn: tăng tốc inference; nếu không build được sẽ tự fallback PyTorch
cd dcvc_rt/src/layers/extensions/inference
pip install .
cd ../../../../..
```

Tải các checkpoint DCVC-RT chính thức vào `checkpoints/dcvc_rt/`:

```text
cvpr2025_image.pth.tar
cvpr2025_video.pth.tar
```

Nguồn checkpoint: [Microsoft DCVC-RT checkpoint folder](https://1drv.ms/f/c/2866592d5c55df8c/Esu0KJ-I2kxCjEP565ARx_YB88i0UnR6XnODqFcvZs4LcA?e=by8CO8).

Đặt `yolov5s.pt` ở thư mục gốc repository hoặc truyền đường dẫn bằng `--weights`.

## Chuẩn bị dữ liệu train

Vimeo-90K Septuplet cần có layout sau:

```text
vimeo_septuplet/
├── sep_trainlist.txt
├── sep_testlist.txt
└── sequences/00001/0001/im1.png ... im7.png
```

Kiểm tra dataset trước khi train:

```bash
python train_base.py --self_check
python train_base.py --dataset /path/to/vimeo_septuplet --check_dataset
```

## Train

Mặc định: crop `256×256`, batch global `4`, năm P-frame, Adam `lr=1e-6`,
gradient clipping và validation loss sau mỗi epoch.

Train bốn model riêng, mỗi model dùng một thư mục output riêng:

```bash
python train_base.py --dataset /path/to/vimeo_septuplet --paper_lambda 2  --save_dir checkpoints/paper_lambda2  --epochs 20
python train_base.py --dataset /path/to/vimeo_septuplet --paper_lambda 4  --save_dir checkpoints/paper_lambda4  --epochs 20
python train_base.py --dataset /path/to/vimeo_septuplet --paper_lambda 8  --save_dir checkpoints/paper_lambda8  --epochs 20
python train_base.py --dataset /path/to/vimeo_septuplet --paper_lambda 16 --save_dir checkpoints/paper_lambda16 --epochs 20
```

Train bằng hai GPU:

```bash
torchrun --standalone --nproc_per_node=2 train_base.py \
  --dataset /path/to/vimeo_septuplet \
  --paper_lambda 8 \
  --save_dir checkpoints/paper_lambda8 \
  --epochs 20
```

`--batch_size 4` là batch tổng, do đó dùng 2 GPU thì mỗi GPU nhận batch 2.

### File sau khi train

Ví dụ với `λ=8`:

```text
checkpoints/paper_lambda8/
├── video_lambda8_random_qp_last.pth.tar  # checkpoint để resume
├── video_lambda8_random_qp_epoch_0020.pth # snapshot cuối epoch
├── best_val_loss.pth                      # validation loss thấp nhất
├── lambda8_random_qp_training_history.json
├── lambda8_random_qp_training_curves.png
└── lambda8_random_qp_training_curves_epoch_0020.png
```

Biểu đồ được lưu lại **sau mỗi epoch** và chứa train/validation cho Total Loss,
BPP và Feature MSE.

### Resume

`--epochs` là epoch tổng cần đạt, không phải số epoch cộng thêm:

```bash
python train_base.py \
  --dataset /path/to/vimeo_septuplet \
  --paper_lambda 8 \
  --save_dir checkpoints/paper_lambda8 \
  --epochs 20 \
  --resume
```

Resume khôi phục DMC, clone, Adam, RNG Python/Torch/CUDA, lịch sử và bắt đầu ở
epoch kế tiếp. Checkpoint của cấu hình QP cố định cũ không resume được vào cấu
hình QP ngẫu nhiên mới.

`best_val_loss.pth` tốt nhất theo validation loss, **không tự động là tốt nhất
theo BD-rate-mAP**. Muốn chọn checkpoint theo BD-rate, cung cấp
`--validation_config`; việc này tốn thời gian vì phải mã hóa bitstream thật.

## Đánh giá SVC/DCVC-RT

Dataset đánh giá cần `frames/`, `labels/` và `manifest.json` theo schema của
repository. Một random-QP model được đánh giá tại bốn QP `0, 21, 42, 63` để tạo
một đường rate–mAP đầy đủ.

```bash
python evaluate_vcm.py --mode codec \
  --data-dir /path/to/vcm_eval \
  --dataset-manifest /path/to/vcm_eval/manifest.json \
  --image-ckpt checkpoints/dcvc_rt/cvpr2025_image.pth.tar \
  --video-ckpt checkpoints/paper_lambda8/best_val_loss.pth \
  --qps 0 21 42 63 \
  --reset-interval 32 \
  --force-zero-thres 0.12 \
  --codec-precision fp16 \
  --yolov5-weights yolov5s.pt \
  --detector-size 640 \
  --confidence-threshold 0.001 \
  --nms-iou-threshold 0.6 \
  --max-detections 300 \
  --cuda-index 0 \
  --method-name svc_lambda8_random_qp \
  --output-dir output/svc_lambda8 \
  --bitstream-dir output/svc_lambda8_bitstreams
```

Kết quả chính:

```text
output/svc_lambda8/svc_lambda8_random_qp_results.json
```

Lặp lại cho λ = 2, 4, 16 rồi chọn model có BD-rate-mAP validation tốt nhất.

## Đánh giá HEVC/x265

`evaluate_hevc.py` nén bằng x265, giải mã bằng FFmpeg, chạy YOLO trên toàn bộ
frame tái tạo và đo số byte bitstream thật. Ví dụ cấu hình YUV420 8-bit gần với
cấu hình HEVC trong bài SVC (x265 không phải HM 18.0):

```bash
python evaluate_hevc.py \
  --data-dir /path/to/vcm_eval \
  --dataset-manifest /path/to/vcm_eval/manifest.json \
  --x265-encoder x265 \
  --ffmpeg ffmpeg \
  --qps 27 29 32 38 42 47 \
  --bit-depth 8 \
  --chroma-format 420 \
  --preset medium \
  --x265-extra-arg=--ref=4 \
  --x265-extra-arg=--keyint=32 \
  --x265-extra-arg=--min-keyint=32 \
  --yolov5-weights yolov5s.pt \
  --detector-size 640 \
  --detector-batch-size 4 \
  --confidence-threshold 0.001 \
  --nms-iou-threshold 0.6 \
  --max-detections 300 \
  --cuda-index 0 \
  --method-name hevc_x265_ldp_yuv420_8bit \
  --output-dir output/hevc \
  --bitstream-dir output/hevc_bitstreams \
  --encoder-log-dir output/hevc_logs \
  --resume \
  --keep-progress-checkpoint
```

Kết quả chính:

```text
output/hevc/hevc_x265_ldp_yuv420_8bit_results.json
```

## So sánh BD-rate-mAP

Hai file JSON phải dùng cùng dataset, manifest, YOLO weights, detector threshold
và toàn bộ frame đánh giá. QP không cần giống nhau, nhưng hai đường mAP phải có
vùng chất lượng giao nhau.

```bash
python evaluate_vcm.py --mode bdrate \
  --anchor-results output/hevc/hevc_x265_ldp_yuv420_8bit_results.json \
  --candidate-results output/svc_lambda8/svc_lambda8_random_qp_results.json \
  --rate actual_bpp \
  --metric map5095 \
  --output-dir output/comparison_lambda8
```

Các file tạo ra:

```text
output/comparison_lambda8/
├── bd_rate_map.json
├── rd_points.csv
├── rd_curve_actual_bpp_map50.png
└── rd_curve_actual_bpp_map5095.png
```

- BD-rate âm: SVC cần ít bitrate hơn HEVC tại cùng mAP.
- BD-rate dương: SVC cần nhiều bitrate hơn HEVC tại cùng mAP.
- Nếu báo `do not overlap in quality`, hãy chạy thêm QP HEVC cao hơn hoặc QP
  SVC phù hợp hơn; không nên ngoại suy BD-rate ngoài vùng giao nhau.

## Giới hạn cần biết

- Đây không phải tái lập chính xác kết quả bài SVC gốc: bài gốc dùng
  LCCM-VC/CANF-VC, còn dự án này dùng DCVC-RT.
- x265 là baseline thực nghiệm tiện dụng; nếu cần tái lập paper tuyệt đối phải
  dùng HM 18.0 theo cấu hình của bài báo.
- Cảnh báo `cannot import cuda implementation for inference, fallback to
  pytorch` chỉ có nghĩa extension tăng tốc tùy chọn chưa build; không làm sai
  kết quả.
