# Báo cáo rà soát mã — DCVC-RT-VCM đa tác vụ

*Bắt đầu 2026-09-21. **Trạng thái: đã xong Pha 1 và Pha 2.** Pha 3 (chất lượng mã), Pha 4 (đề xuất) và Pha 5 (sửa) chưa làm, chờ duyệt.*
*Chuẩn đối chiếu: script train thật của bài, trích từ `git show HEAD:kaggle_train_random_qp_lambda_1_64.ipynb` → 794 dòng, lưu tại `$TEMP/audit/paper_train.py`. Mọi số dòng "bài" bên dưới là của tệp đó.*
*Mọi khẳng định kèm `file:dòng` hoặc output lệnh. Phần suy đoán được gắn nhãn GIẢ THUYẾT.*

---

## 1. Tóm tắt

1. **Không tìm thấy lỗi nào làm sai hàm mục tiêu của R3/R4 so với công thức bài.** Các mục cốt lõi đều khớp: λ dùng `q` cơ sở, lịch `q_eff`, trọng số thời gian, chuẩn hóa bpp, cắt teacher, đóng băng DMCI, BPTT qua DPB, Adam lr 1e-6, clip 1.0, batch 4.
2. **`exact_rate.py` khớp script bài ở mức bit**: `bpp` lệch 0.00e+00, `x_hat` lệch 0.00e+00, gradient lệch tương đối **6.38e-07** (đo lại, mục 3 A-00).
3. **AUDIT-01 (CAO):** `diagnose_rates` chạy dưới `@torch.no_grad()` nên **chỉ đo giá trị bpp, không đo gradient**. Đo được trên CPU: hai ước lượng cho giá trị **gần như bằng nhau** (tỉ lệ 0.996–1.002) nhưng **hướng gradient lệch** (cosine 0.68–0.77) và chuẩn gradient surrogate chỉ bằng **56–71 %** exact. Notebook đang hướng dẫn "hai số bpp gần bằng nhau thì nên dừng" → **có thể khiến bỏ H1 một cách sai**.
4. **AUDIT-02 (CAO, đúng trọng tâm H2):** bất đối xứng chuỗi tham chiếu. Train: 6 khung P. Đánh giá: **một khung I duy nhất cho cả chuỗi**, rồi tới 599 khung P liên tiếp; `--reset-interval 32` **không chèn khung I mới**, chỉ đổi sang `feature_adaptor_i` còn khung vẫn lấy từ lần giải mã trước. Sai lệch mỗi khung rất nhỏ có thể khuếch đại ~100 lần ở đánh giá mà 6 khung lúc train không thấy.
5. **AUDIT-03 (TRUNG):** thiếu AMP. Bài bật `autocast(fp16)` + `GradScaler` + `unscale_` trước clip; dự án chạy fp32 thuần.
6. **AUDIT-04 (TRUNG):** BD-rate của dự án gộp **theo lớp** (gộp mọi chuỗi rồi mới dựng đường cong), còn Bảng III của bài tính **theo từng chuỗi rồi trung bình**. Xác nhận bằng mã.
7. **AUDIT-06 (THẤP nhưng cần biết):** `rate_mode` đổi **hai** thứ cùng lúc, không phải một: ngoài cách tính rate, `DMC.forward_train` còn lưu `frame=None` vào DPB trong khi `exact_rate`/bài lưu `frame=x_hat`. Không ảnh hưởng lúc train (đã lập luận và đo), nhưng R3x/R4x vì thế không phải thí nghiệm đổi-một-biến tuyệt đối.
8. Tám phát hiện mức THẤP còn lại: DDP thiếu `broadcast_buffers=False`, import chết, cờ `training` của wrapper đóng băng, khác seed, khác kích thước tập validation, phụ thuộc `train_base.py` đã bị sửa cục bộ.
9. **Ảnh hưởng tới H1:** chưa bị bác bỏ, nhưng **phép thử đang thiết kế sai** (AUDIT-01). Sửa trước khi chạy Kaggle.
10. **Ảnh hưởng tới H2:** AUDIT-02 cho H2 một cơ chế khuếch đại cụ thể, định lượng được và đo được mà không cần train lại.

---

## 2. Bảng đối chiếu với recipe của bài

Ký hiệu: **K** = khớp, **L** = lệch, **?** = không kiểm được trên máy local.

### 2.1 Rate và loss

| Mục | Script bài | Dự án | KQ |
|---|---|---|---|
| Rate tính trên ký hiệu đã làm tròn | `paper_train.py:254-257` `masked_quant_train` → `ste_round`, `:262-264` `gaussian_bits(yq0,…)` | `exact_rate.py:44-49` | **K** |
| Rate của chế độ mặc định R0–R4 | — | `common_model.py:155-171` `forward_prior_train_2x` tính bits trên `residual` **chưa làm tròn** | **L** (có chủ đích, là H1) |
| Rate trên cả y và z | `:262-266` `by0+by1+bz` | `exact_rate.py:33-36, 46-48` | **K** |
| Chuẩn hóa bpp | `:265-266` chia `x.shape[-2]*x.shape[-1]` (crop 256²), rồi `.mean()` theo batch `:292` | `exact_rate.py:54` chia `B*H*W` sau khi `get_gaussian_bits` `.sum()` toàn bộ | **K** (tương đương toán học) |
| log2 | `:201` `-torch.log2` | `common_model.py:140` `-torch.log2` | **K** |
| Clamp scale [0.11, 16] | `:196` | `common_model.py:139` | **K** |
| `exact_rate` vs bài: giá trị | — | bpp lệch `0.00e+00`, `x_hat` lệch `0.00e+00` | **K** |
| `exact_rate` vs bài: gradient | — | `‖g_bài − g_ta‖ / ‖g_bài‖ = 6.38e-07` | **K** |
| λ dùng `q` cơ sở, không dùng `q_eff` | `:593` `lambda_for_qp(base_qp, args)`, `:287` `curr_qp` chỉ dùng cho `shift_qp` | `train_multitask.py:363` `lambda_for_qp(base_qp)` | **K** |
| λ(q) = 1·64^(q/63) | `:96-100` + CFG `lambda_min 1.0, lambda_max 64.0` | `lambda_schedule.py:9-13` | **K** |
| Lịch `q_eff` | `:286-287` `INDEX_MAP[(t+1)%8]` + `shift_qp` | `train_multitask.py:171-173` cùng công thức | **K** |
| `q_eff` có bị clip không | `video_model.py:400` `shift_qp` **không clip**; `extra_qp = max(qp_shift) = 8` → chỉ số hợp lệ 0..71 | như trên (dùng chung hàm) | **K** |
| Trọng số thời gian | `:295` `DISTORTION_WEIGHTS[t % 4]` → `[0.5,1.2,0.5,0.9,0.5,1.2]` | `train_multitask.py:174-176` cùng công thức | **K** |
| Loss clip = trung bình | `:302` `torch.stack(loss_terms).mean()` | `multitask_system.py:105` | **K** |
| Distortion = MSE trung bình toàn bộ | `:291` `F.mse_loss(..., reduction='mean')` | `svc_machine/feature_loss.py` `feature_mse_loss` | **K** |
| Teacher nhận RGB [0,1], không letterbox, không chuẩn hóa | `:368` `teacher(p_rgb[:, t])` | `train_multitask.py:160-165` | **K** |
| Cắt teacher đúng 5 lớp (0..4) | `:170` `nn.Sequential(*[seq[i] for i in range(5)])` | `feature_extractor.py:27` `cut_model=1, cutting_layer=4` | **K** — đo: lệch `0.000e+00`, shape `(2,128,32,32)` |
| Ảnh vào teacher lúc train là RGB đã clamp | `:289` `ycbcr2rgb(x_hat_yuv, clamp=True)` | `multitask_system.py:85` (`clamp=True` mặc định) | **K** |

### 2.2 Chuỗi tham chiếu (H2)

| Mục | Script bài | Dự án | KQ |
|---|---|---|---|
| Gradient chảy qua DPB giữa các khung P | `:268-269` "Keep temporal graph (BPTT)", không detach | `exact_rate.py:52`, `video_model.py:315` — không detach | **K** |
| DMCI đóng băng + eval | `:457-460` | `train_multitask.py:116-118` | **K** |
| DPB khởi tạo | `:276` `add_ref_frame(feature=None, frame=x0_hat_yuv)` | `train_multitask.py:154` `add_ref_frame(None, reference)` | **K** |
| Nội dung lưu vào DPB mỗi khung P | `:269` `add_ref_frame(feature=feature, frame=x_hat)` | `exact_rate.py:52` giống; **`video_model.py:315` (surrogate) lưu `frame=None`** | **L** — AUDIT-06 |
| Số khung P lúc train | `:283-284` `T = p_yuv.shape[1]` = 6 | `train_multitask.py:171-176`, `--group_size 6` | **K** |
| Số khung P lúc **đánh giá** | — | `evaluate_vcm.py:145-147` một khung I duy nhất; `:163` vòng `range(1, frame_count)` tới 599 khung P | **L** — AUDIT-02 |
| `--reset-interval 32` có chèn khung I mới không | — | **Không**: `evaluate_vcm.py:169-171` chỉ gọi `prepare_feature_adaptor_i`; `video_model.py:395-400` dựng frame từ feature rồi đặt `feature=None` | **L** — AUDIT-02 |

### 2.3 Tối ưu và độ chính xác số

| Mục | Script bài | Dự án | KQ |
|---|---|---|---|
| Optimizer | `:549-552` Adam, 2 nhóm (`lr_video`, `lr_yolo`), `weight_decay=0.0` | `train_multitask.py:307` Adam 1 nhóm, `weight_decay` mặc định 0 | **K** (hai lr đều 1e-6 → tương đương) |
| Learning rate | CFG `lr_video = lr_yolo = 1e-6` | `--learning_rate 1e-6` (notebook) | **K** |
| betas | mặc định `(0.9, 0.999)` | mặc định | **K** |
| Grad clip | `:610` `clip_grad_norm_(system.parameters(), 1.0)` | `train_multitask.py:365-366` `clip_grad_norm_(parameters, 1.0)` | **K** |
| Scheduler | không có | không có | **K** |
| Gradient accumulation | `:596-600`, CFG `grad_accum = 1` | không có (bước mỗi batch) | **K** khi `grad_accum=1` |
| **AMP** | `:603` `autocast(fp16, enabled=amp)`, `:553` `GradScaler`, `:606-612` `scale/unscale_/step`; CFG `amp = True` | **không có autocast, không có scaler**; `train_multitask.py:364` `loss.backward()` | **L** — AUDIT-03 |
| Tính entropy ép fp32 | `:195, 208` `autocast(enabled=False)` bên trong `gaussian_bits`/`z_bits` | không cần vì toàn bộ là fp32 | **K** (hệ quả của AUDIT-03) |
| Batch toàn cục | CFG `batch_per_gpu 2` × 2 GPU = 4 | `--batch_size 4`, `per_rank_batch = 4 // 2 = 2` | **K** |
| DDP | `:498-499` `broadcast_buffers=False, find_unused_parameters=False` | `train_multitask.py:303` **mặc định** (`broadcast_buffers=True`) | **L** — AUDIT-05 |
| Tập tham số cập nhật | `:547-548` `p_net` + `student_front` | `train_multitask.py:306` lọc `requires_grad` | **K** |
| Seed | `:76-79` `random`, `numpy`, `torch`, `cuda` với seed 1234 | `train_multitask.py:280-281` chỉ `random` + `torch`, seed 0 | **L** — AUDIT-11 |

### 2.4 Mạng đóng băng

| Mục | Script bài | Dự án | KQ |
|---|---|---|---|
| Teacher `requires_grad=False` + `eval()` | `:170-173` | `feature_extractor.py:13-17`; `frozen_feature.py:20-22` | **K** |
| `freeze_bn` áp lại sau mỗi `train()` | `:583-584` `system.train(); freeze_bn(student_front)` | `multitask_system.py:68-73` — override `train()` | **K** |
| BN thật sự đóng băng | — | self_check `check_clone_batchnorm_frozen` PASS: `running_mean` không đổi, khớp teacher trên ảnh giống hệt (sai số < 1e-5) | **K** |
| Nhánh đóng băng không nhận gradient | — | đo: `co grad tren tham so dong bang = False` cho cả det4 và seg17 | **K** |
| Gradient chảy về ảnh đầu vào | — | đo: `grad toi anh = 2.50e+00` (det4), `3.17e+02` (seg17) | **K** |
| Tầng cắt | `n_layers=5` | det tầng 4 → `(2,128,32,32)`; seg tầng 17 → `(2,128,32,32)` | **K** |
| Cờ `training` của wrapper đóng băng | — | `frozen_feature.py:16-22` đặt `self.model.eval()` nhưng không đặt cờ wrapper → `training=True` tới khi có ai gọi `.train()` | **L** — AUDIT-07 |

### 2.5 Dữ liệu

| Mục | Script bài | Dự án | KQ |
|---|---|---|---|
| Crop 256² cùng vị trí cho cả 7 khung | `:145-157` | `data.py:37-46` | **K** |
| Lật ngang cùng lúc cả clip, trục W | `:147,155-156` `torch.flip(x, dims=[2])` trên `[3,H,W]` | `data.py:47-48` `torch.flip(clip, dims=(-1,))` | **K** |
| Không lật khi validate | `:151` | `data.py:47` (`and self.random_crop`) + `hflip=False` | **K** |
| Số khung nạp | `:125` `1 + p_frames = 7`, `im1..im7` | `data.py:25-29`, `group_size=6` → `randint(1,1)` = luôn `im1..im7` | **K** |
| Dải giá trị, dtype | `:135` `/255.0` float32 | `data.py:31` `to_tensor` | **K** |
| Split | `sep_trainlist` / `sep_testlist` | `--validation_list sep_testlist.txt` | **K** |
| Số clip validation | CFG `max_val_samples = 100` | toàn bộ 917 clip (`--max_val_batches 250` không ràng buộc) | **L** — AUDIT-10 |
| QP validation | CFG `val_qps '0,21,42,63'` | `train_base.VALIDATION_QPS = (0,21,42,63)` | **K** |

### 2.6 Checkpoint và resume

| Mục | Dự án | KQ |
|---|---|---|
| Checkpoint nạp được bằng `evaluate_vcm.load_codec_checkpoint` | self_check khẳng định, PASS | **K** |
| Không dính tiền tố `module.` | `train_multitask.py:403` truyền `system` (thô), không phải `model` (DDP) | **K** |
| Resume khôi phục optimizer, RNG, epoch, history | `train_multitask.py:314-330` | **K** |
| Resume khôi phục **scaler** | không có scaler (AUDIT-03) | n/a |
| Resume từ chối cấu hình khác | `:316-319` so `config` | **K** |
| Warm start nạp đúng checkpoint bài | `checkpoints.py:40-49`; self_check `check_warm_start_layouts` PASS | **K** |

### 2.7 Đánh giá (chỉ đọc, không sửa)

| Mục | Dự án | KQ |
|---|---|---|
| Intra period khi đánh giá | 1 khung I / chuỗi; 6 khung P khi train | **L** — AUDIT-02 |
| `--force-pretrained-frontend` luôn bật | notebook eval truyền cờ và **assert** `machine_frontend.type == 'pretrained_yolov5_frontend'` sau mỗi method | **K** |
| Anchor 8-bit 4:2:0 cả hai lớp | notebook anchor đối chiếu `codec_config` (qps, bit_depth, chroma, preset) trước khi dùng lại | **K** |
| `evaluation_id` trùng anchor và ứng viên | notebook eval so `evaluation_id` + `detector_config` trước khi giải mã | **K** (không kiểm được local) |
| `prepare_eval_dataset` xác định, idempotent | đã test trước đó: hai thư mục đích khác nhau cho cùng `evaluation_id`; chạy đè cũng vậy | **K** |
| Phân lớp theo tên chuỗi | notebook eval `sfu_class_of` | **K** |
| BD-rate gộp theo lớp hay theo chuỗi | `evaluate_vcm.py:803-806` `curve_arrays` dùng `point[metric]` cấp cao nhất = **gộp mọi chuỗi** | **L** — AUDIT-04 |
| Số điểm rate | 6 điểm `q = {0,12,24,36,48,63}` (bài dùng cùng bộ) | **K** |
| Điểm không đơn điệu | `pareto_front` loại điểm bị trội trước khi nội suy | **K** |

---

## 3. Danh sách phát hiện

### A-00 — Xác minh lại `exact_rate` (không phải lỗi)

Phương pháp đo: `$TEMP/audit/measure_audit.py`. DMC khởi tạo ngẫu nhiên, nhiễu `0.02·N(0,1)` cộng vào mọi tham số (vì DMC khởi tạo sạch cho scale gần suy biến), DPB mồi bằng một khung ngẫu nhiên, cùng đầu vào, qp = 21, loss `bpp + mean(x_hat²)`.

```
bpp paper=7.38269997  ours=7.38269997  |d|=0.00e+00
|x_hat| max diff = 0.00e+00
grad: ||ga-gb||/||ga|| = 6.380e-07
```

Con số `6.38e-07` là **chuẩn tương đối của hiệu vector gradient trên toàn bộ tham số DMC**, không phải sai số từng phần tử.

### AUDIT-01 — CAO — Phép chẩn đoán H1 đo sai đại lượng

**Vị trí:** `multitask_exp/train_multitask.py:435` (`@torch.no_grad()`), `:456-459`.
**Bằng chứng** (`$TEMP/audit/measure_audit.py`, mục B):

```
  qp   bpp surr  bpp exact   ti le  cos(grad) ||gs||/||ge||
   0    7.32488    7.35170   0.996    0.68479         0.709
  21    7.36313    7.38270   0.997    0.76779         0.593
  42    7.32905    7.35785   0.996    0.77425         0.561
  63    7.32639    7.30961   1.002    0.74049         0.641
```

Giá trị bpp gần như trùng (lệch dưới 0,5 %), nhưng **hướng gradient lệch khoảng 40°** và độ lớn chỉ còn 56–71 %. Huấn luyện chỉ phụ thuộc gradient.

Notebook `kaggle_multitask_train_R3x_R4x_exact.ipynb` in ra tỉ lệ bpp và markdown viết *"gần bằng nhau thì giả thuyết yếu đi và nên dừng lại trước khi tốn 6 giờ GPU"*. Với kết quả trên, **nhiều khả năng cell đó sẽ báo tỉ lệ ≈ 1 và dẫn tới kết luận sai**.

**Lưu ý trung thực:** đo trên mô hình khởi tạo ngẫu nhiên có nhiễu, không phải checkpoint bài với dữ liệu Vimeo thật. Tỉ lệ *giá trị* phụ thuộc mô hình (một lần đo khác trên DMC ngẫu nhiên thuần cho tỉ lệ 1,745). Điều ổn định giữa các lần đo là **hai ước lượng cho gradient khác nhau**.

**Đề xuất sửa:** bỏ `@torch.no_grad()`, thêm đo cosine và tỉ lệ chuẩn gradient của `d(loss)/d(tham số DMC)` giữa hai chế độ, trên vài batch. Ghi cả hai vào `rate_diagnosis.json`. Chi phí GPU không đổi đáng kể.
**Trạng thái:** ĐÃ SỬA — `6c400d3`. Notebook R3x/R4x cũng đã đổi tiêu chí quyết định sang `cos(grad)` và ghim lại commit.

### AUDIT-02 — CAO — Bất đối xứng chuỗi tham chiếu train và đánh giá (H2)

**Vị trí:** `evaluate_vcm.py:145-147` (một khung I duy nhất), `:163-172` (vòng khung P), `:169-171` (reset), `dcvc_rt/src/models/video_model.py:395-400` (`prepare_feature_adaptor_i`).

Khi đánh giá, mỗi chuỗi mã hóa **đúng một** khung I bằng DMCI rồi toàn bộ phần còn lại là khung P nối tiếp (BQSquare: 599 khung P). `--reset-interval 32` **không** chèn khung I mới: nó dựng lại `frame` từ `feature` rồi đặt `feature = None`, nên khung tham chiếu vẫn là kết quả giải mã trước đó và sai số vẫn truyền tiếp.

Huấn luyện chỉ thấy 6 khung P. Tỉ lệ khoảng **100 lần**.

**Đây không phải lỗi trong mã dự án** (mã đánh giá là của bài, và checkpoint bài chịu đúng giao thức này). Nó là **cơ chế khuếch đại** khiến một sai lệch mỗi khung rất nhỏ trở thành sụp đổ ở đánh giá, và giải thích vì sao val loss trên clip 7 khung không dự báo được mAP.

**GIẢ THUYẾT** (chưa kiểm): nếu H2 đúng, khoảng cách mAP giữa R3 và checkpoint bài phải **tăng dần theo chỉ số khung P**. Nếu chênh lệch phẳng ngay từ khung đầu thì H2 yếu.
**Trạng thái:** mở; đề xuất phép đo ở Pha 4.

### AUDIT-03 — TRUNG — Thiếu mixed precision

**Vị trí:** bài `paper_train.py:553, 603-612`; dự án `train_multitask.py:364` (`loss.backward()` trần).
Bài bật AMP mặc định (`--amp` default 1, CFG `'amp': True`), gồm `GradScaler`, `scaler.unscale_(optimizer)` trước clip, và ép fp32 cho phần entropy.
fp32 chính xác hơn fp16 nên **khó là nguyên nhân gây sụp**, nhưng đây là khác biệt recipe còn lại và làm thay đổi nhiễu gradient cùng tương tác với clip 1.0.
**Trạng thái:** ĐÃ SỬA — `facf00d`. Cờ `--amp` mặc định tắt, chỉ chạy cùng `--rate_mode exact` (đường surrogate nằm trong `dcvc_rt/`, không được sửa, nên sẽ tính entropy ở fp16). `exact_rate.py` nay ép fp32 cho phần entropy đúng như `paper_train.py:195-213`; guard là no-op khi không bật AMP.

### AUDIT-04 — TRUNG — BD-rate gộp khác cách của bài

**Vị trí:** `evaluate_vcm.py:803-806`.
`curve_arrays` dựng đường cong từ `point[metric]` cấp cao nhất, tức mAP **gộp mọi chuỗi trong lớp**. Bảng III của bài tính BD-rate **từng chuỗi rồi trung bình 8 chuỗi**. Hai cách cho số khác nhau.
Không phải lỗi, nhưng nghĩa là mọi con số trong dự án **không so trực tiếp được** với Bảng III. Đã ghi trong `00_TONG_QUAN_DU_AN.md` mục 9; nay có xác nhận bằng mã. Mã đánh giá là mã cũ → **chỉ báo cáo, không sửa**.
**Trạng thái:** mở (chỉ ghi nhận).

### AUDIT-05 — THẤP — DDP thiếu `broadcast_buffers=False`

**Vị trí:** `train_multitask.py:303`; bài `paper_train.py:498-499`.
Mặc định `broadcast_buffers=True` khiến DDP phát lại buffer (gồm running stats của teacher và nhánh seg đóng băng) từ rank 0 mỗi forward. Vì BN ở eval nên giá trị không đổi → **không sai ngữ nghĩa**, chỉ tốn băng thông.
**Trạng thái:** ĐÃ SỬA — `f5941c1`.

### AUDIT-06 — THẤP — `rate_mode` đổi hai thứ cùng lúc

**Vị trí:** `dcvc_rt/src/models/video_model.py:315` (`add_ref_frame(feature, None)`) so với `exact_rate.py:52` và `paper_train.py:269` (`frame=x_hat`).
**Bằng chứng:**
```
sau forward_train : dpb[0].frame is None -> True
sau exact         : dpb[0].frame is None -> False
```
Không ảnh hưởng khi train: `apply_feature_adaptor` (`video_model.py:376-379`) chỉ đọc `frame` khi `feature is None`, mà `feature` không bao giờ `None` sau khung P đầu tiên, và `prepare_feature_adaptor_i` không được gọi lúc train.
Nhưng phải ghi rõ: R3x/R4x khác R3/R4 ở **hai** điểm, nên nếu R3x tốt lên thì chưa quy được 100 % cho cách tính rate.
**Trạng thái:** mở (ghi nhận, không cần sửa).

### AUDIT-07 — THẤP — Cờ `training` của `FrozenYoloFeature` không nhất quán

**Vị trí:** `multitask_exp/frozen_feature.py:16-22`.
Hàm khởi tạo gọi `self.model.eval()` nhưng không đặt cờ của wrapper, nên `FrozenYoloFeature(...).training == True` cho tới khi có ai gọi `.train()`.
**Bằng chứng:** `C. det4: … training=True` (chỉ tạo rồi forward, chưa gọi `.train()`).
Đường chạy thật an toàn vì `system.train()` gọi tới override. self_check che lỗi này vì nó gọi `seg.train()` trước khi assert (`train_multitask.py:511-513`).
**Trạng thái:** ĐÃ SỬA — `1a19138`.

### AUDIT-08 — THẤP — Import chết

**Vị trí:** `train_multitask.py:50` `from multitask_exp.exact_rate import forward_train_exact`. Việc chọn ước lượng nằm trong `multitask_system.py:79-84`; tệp này chỉ dùng tên đó trong `check_exact_rate_matches_paper` — vốn nằm trong cùng tệp nên import vẫn cần. **Cần xác minh lại ở Pha 3** trước khi kết luận là chết.
**Trạng thái:** mở, chờ Pha 3.

### AUDIT-09 — THẤP — Phụ thuộc `train_base.py` đã bị sửa cục bộ

`train_multitask.py:37-47` nhập `DISTORTION_WEIGHTS, INDEX_MAP, QP_OFFSETS, VALIDATION_QPS, synchronized_qp, …` từ `train_base.py`, mà tệp này đang ở trạng thái **M** (sửa chưa commit).
**Đã kiểm:** bốn hằng số dùng thật **khớp** giữa HEAD và bản local; chỉ `LAMBDA_MIN` (1.0 → 0.25), `LAMBDA_MAPPING` và `LAMBDA_SHAPE` khác. Vì `train_multitask` lấy λ từ `lambda_schedule.py` nên không bị ảnh hưởng, và self_check local vẫn hợp lệ.
**Rủi ro còn lại:** nếu sau này có ai nhập `lambda_for_qp` từ `train_base` thì sẽ lấy đúng công thức sai.
**Trạng thái:** mở (ghi nhận).

### AUDIT-10 — THẤP — Tập validation khác bài

Bài giới hạn 100 clip (`CFG max_val_samples`); dự án dùng cả 917 clip của `sep_testlist.txt`. Không sai, nhưng val loss hai bên **không so trực tiếp được** — điều này liên quan tới nhận định trong tài liệu tổng quan rằng "val loss của các run thấp hơn bài".
**Trạng thái:** mở (ghi nhận).

### AUDIT-11 — THẤP — Seed khác

Bài `paper_train.py:76-79` seed `random`, `numpy`, `torch`, `torch.cuda` bằng 1234 + rank. Dự án `train_multitask.py:280-281` chỉ seed `random` và `torch`, mặc định 0. Không seed `numpy` (không dùng) và không `cuda.manual_seed_all`.
**Trạng thái:** ĐÓNG, không sửa. `torch.manual_seed` đã seed mọi thiết bị CUDA nên `cuda.manual_seed_all` là thừa, và `numpy` không được dùng trong đường chạy này. Chỉ còn khác **giá trị** seed (0 so với 1234), là lựa chọn có chủ ý, đổi thì phá tính tái lập của R0–R4.

---

## 4. Việc không kiểm được trên máy local

| Cần kiểm | Vì sao không làm được local | Lệnh trên Kaggle |
|---|---|---|
| Tỉ lệ bpp và **lệch gradient** giữa hai ước lượng trên **checkpoint bài + Vimeo thật** | Cần checkpoint `u-30epoch` và dữ liệu Vimeo | `--diagnose_rates` sau khi sửa AUDIT-01 |
| mAP theo chỉ số khung P (kiểm H2) | Cần bitstream thật, extension CUDA | script mới, đề xuất ở Pha 4 |
| `u-30epoch` có lưu checkpoint các epoch khác không | Dataset chỉ có trên Kaggle | liệt kê tệp trong dataset |
| `evaluation_id` trùng giữa anchor và ứng viên | Cần dữ liệu SFU | notebook eval đã assert sẵn |
| Thời gian mỗi epoch, hành vi DDP 2 GPU | Máy local 1 GPU, không có extension | notebook train |

---

## 5. Đối chiếu với `CODE_BUGS_REPORT.md` (bên thứ hai)

Báo cáo đó rà cùng phạm vi. Bảng dưới ghi chỗ khớp, chỗ lệch và chỗ sai, mỗi khẳng định
kèm vị trí kiểm được.

| Bên kia | Ở đây | Kết luận |
|---|---|---|
| BUG-01 surrogate/exact, xếp 🔴 "gốc rễ" | H1, AUDIT-06 | **Cùng nghi vấn, nhưng bên kia kết luận không kèm số đo.** Ở đây có đo: bpp lệch <0,5 %, cos(grad) 0,68–0,77. Khuyến nghị "mặc định `rate_mode=exact`" bị bác: đổi mặc định làm mọi lệnh R0–R4 cũ đổi nghĩa, và `rate_mode` còn đổi cả `add_ref_frame` (AUDIT-06), nên không quy được kết quả cho một nguyên nhân |
| BUG-02 thiếu AMP 🔴 | AUDIT-03 🟡 | **Khớp về sự thiếu, lệch về lý do.** Lập luận "BN statistics ở FP16" của bên kia sai: `GradScaler` giữ master weight fp32 và `autocast` vẫn chạy BatchNorm ở fp32. fp32 chính xác hơn fp16 nên khó là nguyên nhân gây sụp; đã sửa thành cờ tùy chọn (`facf00d`) chứ không bật mặc định |
| BUG-03 `@torch.no_grad()` 🟡 | AUDIT-01 🔴 | **Khớp về lỗi, lệch về mức độ.** Đây là thứ duy nhất chặn việc kiểm H1, nên xếp CAO. Đã sửa (`6c400d3`) |
| BUG-04 `FrozenYoloFeature` 🟡 | AUDIT-07 🟢 | **Trích dẫn sai.** `multitask_exp/frozen_feature.py` chỉ có 33 dòng, không có `freeze()`, không có dòng 50–60. Lỗi thật khác hẳn: `__init__` gọi `self.model.eval()` chứ không `self.eval()`. Đã sửa (`1a19138`) |
| BUG-05 NaN `d_seg` 🟢 | — | **Đúng, ở đây bỏ sót.** Đã sửa (`e6d7803`) |
| BUG-07 thiếu `find_unused_parameters` 🟡 | AUDIT-05 | **Sai.** Chính báo cáo đó trích bài dùng `find_unused_parameters=False`, mà `False` là mặc định của PyTorch → không thiếu gì. Đề nghị đặt `True` sẽ làm chậm mà không sửa gì. Khoảng trống DDP thật là `broadcast_buffers` (đã sửa `f5941c1`) |
| BUG-06, BUG-08, BUG-09 | — | **Đúng**, đều là nợ kỹ thuật/tài liệu, không đổi hành vi. BUG-09 trùng phần ghi nhận ở mục 2.7 |
| — | **AUDIT-02** (một khung I rồi tới 599 khung P; `--reset-interval` không chèn khung I) | **Bên kia bỏ sót hoàn toàn.** Đây là cơ chế khuếch đại ~100× giữa huấn luyện và đánh giá |
| — | AUDIT-04, AUDIT-09, AUDIT-10 | Bên kia bỏ sót |

## 6. Pha 3, 4

Chưa làm. Chờ duyệt.
