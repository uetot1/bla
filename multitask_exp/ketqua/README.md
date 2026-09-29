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
