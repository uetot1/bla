# Kết quả đo — nguồn của các con số trong khóa luận

Mọi tệp `*_results.json` / `*_mask_results.json` ở đây là đầu ra nguyên văn (schema v7) của các
notebook Kaggle, tải về từ tab Output. Tệp được xếp vào thư mục theo **nội dung** (trường
`evaluation_id`, `task`, `ground_truth`, số khung), không theo tên tệp gốc. BD-rate tính lại được bằng
`multitask_exp/plot_rd.py` (`bd_rate`, `incompatibility`).

## `kitti_mots/` — trục segmentation, nhãn người (đủ)

- Nguồn: `train_kaggle/kaggle_multitask_kitti_mots.ipynb`, commit `87b0cb4`, lần chạy thứ 2
  (lần 1: mốc HEVC xong, 5 codec nơ-ron hỏng trên GPU1; tệp HEVC ở đây là của lần 1, được lần 2 dùng lại
  vì `evaluation_id` trùng).
- Tập: KITTI-MOTS val, 9 sequence, 2 981 khung, `evaluation_id` `sha256:8e2cce28…`.
- `HEVC_x265_mask_results.json` là mốc; `paper`, `dcvcrt`, `R4f`, `R3f`, `R1f` là codec nơ-ron (fp16).
- `summary_kitti.json`: bảng BD-rate. **Đã tính lại trên máy từ các tệp JSON ở đây: trùng tới 2 chữ số
  thập phân** với summary (bài −69,88 / −67,47; DCVC-RT gốc −70,19 / −64,99; R4f −70,80 / −67,60;
  R3f −69,89 / −67,58; R1f −75,22 / −72,00).
- Dùng cho: Bảng 4.3 và Hình 4.2 của khóa luận; slide "Chấm bằng mask người gán".

## `sfu_class_c/` — SFU Class C (một phần)

- `evaluation_id` `sha256:6bc62f4b…`, 4 sequence, 1 900 khung.
- `HEVC_x265_mask_results.json`: mốc HEVC trục mask (nhãn proxy) — `kaggle_multitask_mask_anchor.ipynb`, commit `154ee75`.
- `paper_results.json`, `R4f_results.json`, `R3f_results.json`, `R1f_results.json`: trục box (fp16) —
  `kaggle_multitask_classc_rq2.ipynb`, commit `85a0f15`.

## `sfu_class_d/` — SFU Class D (một phần)

- `evaluation_id` `sha256:415914dc…`, 4 sequence, 1 900 khung.
- `HEVC_x265_mask_results.json`: mốc HEVC trục mask (nhãn proxy) — commit `154ee75`.
- `paper_results.json`: trục box của bài (fp16).
- `R4f_flt_e03_fp16_mask_results.json`, `R3f_flt_e03_fp16_mask_results.json`: cặp RQ2 trên dữ liệu đã lọc
  (hiệu −9,3) — `kaggle_multitask_classc_rq2.ipynb`, commit `85a0f15`.

## `legacy_runs/` — các run trước khi sửa cách tính rate

- `R0_results.json`, `R1_results.json`, `R2_results.json`: trục box, SFU Class D, checkpoint
  `multitask_runs/R*/best_val_loss.pth`. Tên gốc khi tải về là `ro.txt`, `r1.txt`, `r2.txt` (nội dung là JSON).
- Không dùng cho bảng kết quả chính; giữ lại làm bằng chứng cho phần chẩn đoán (mục 3.6).

## Còn thiếu so với khóa luận

Các bảng SFU của khóa luận (Bảng 4.1, 4.2, 4.4, 4.5) lấy từ `summary_final.json` của
`kaggle_multitask_final.ipynb` (commit `503ddf8`, lần chạy thứ 3), chưa có trong thư mục này.
Cần tải thêm từ output của phiên đó:

- `summary_final.json`
- `thesis_figures/rd_map50.png`, `thesis_figures/rd_map5095.png` (Hình 4.1)
- nếu được: các tệp `.json` trong `eval_box/classd`, `eval_box/classc`, `eval_mask/class_d`,
  `eval_mask/class_c` (AVC, DCVC-RT gốc, các mask fp16 của bài / R4f / R3f / R1f, cặp seed 4321)

## `rq2_seeds/` — RQ2 với 5 seed (đủ)

- Nguồn: `train_kaggle/kaggle_multitask_seeds.ipynb`, commit `33557b8`, một phiên 6,6 giờ, không lỗi;
  6 run mới đều đủ 3 epoch (`seed_runs/*/status.json`).
- Tiêu chí đặt trước (trong notebook): seed mới 1111, 2222, 3333; gộp 5 cặp R4f − R3f trên dữ liệu gốc;
  RQ2 xác lập chỉ khi cận trên KTC 95 % < 0.
- `eval_mask_class_d/`: toàn bộ kết quả mask SFU Class D (fp16 và fp32 cũ) có cùng `evaluation_id`
  `sha256:415914dc…`, gồm cả tệp của hai cặp cũ (seed 1234, 4321) và mốc `paper_fp16`, `HEVC_x265`.
- Kết quả (tính lại trên máy, trùng `summary_seeds.json`): hiệu cặp −12,55 / −1,20 / −11,28 / +11,78 / +2,64;
  trung bình −2,12, độ lệch chuẩn 10,12, KTC 95 % [−14,68; +10,44] → **RQ2 không xác lập**.

## `bosung/` và `sfu_final/` — phép đo bổ sung (đủ)

- `bosung/`: `train_kaggle/kaggle_multitask_bosung.ipynb`, commit `83a01fe`, 8,6 giờ, không lỗi. Gồm **mọi** tệp kết quả SFU
  (`eval_box/`, `eval_mask/` cho Class C và D, cũ và mới), `summary_bosung.json`, hình vẽ lại.
- `sfu_final/` (cũng có trong `bosung/tu_phien_final/`): `summary_final.json` và `rd_map50.png` / `rd_map5095.png` của phiên final
  (commit `503ddf8`) — hình có đủ mốc HEVC ở cả bốn panel, dùng làm Hình 4.1.
- RQ1 (tiêu chí đặt trước: KTC 95 % của BD box D R4f so với bài nằm trong ±5 %): +1,18 / +7,95 / +8,46 / −0,23 / +8,30;
  trung bình +5,13, KTC [−0,19; +10,45] → **không đạt tiêu chí**; so với HEVC R4f trung bình −71,9 % so với bài −73,2 %.
- RQ2 trên Class C (4 cặp): +4,05 / −2,37 / −6,69 / −15,36; trung bình −5,09, KTC [−18,05; +7,86].
- RQ3: Class D −51,7 %, Class C −49,7 %.
- Thiếu: mốc HEVC trục box trong lần này (không attach output "eval box cũ"); R3f seed 4321 trên trục box (không tìm thấy checkpoint).
