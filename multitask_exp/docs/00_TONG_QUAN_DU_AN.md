# Tổng quan dự án: DCVC-RT-VCM và hướng mở rộng đa tác vụ

*Tài liệu dành cho người mới vào dự án. Không cần biết trước về nén video, mạng nơ-ron nén, hay bài báo gốc. Trạng thái cập nhật: **2026-09-21**. Đọc từ trên xuống; mục 0 là bản tóm tắt nếu chỉ có 5 phút.*

---

## 0. Tóm tắt trong một trang

**Dự án là gì.** Nén video sao cho **máy** (mạng nhận diện đối tượng) nhìn vào video đã nén vẫn nhận diện tốt, thay vì để **người** nhìn đẹp. Làm điều đó bằng cách tinh chỉnh một codec video nơ-ron có sẵn (DCVC-RT) với một hàm mục tiêu mới.

**Đã làm xong (bài báo RIVF).** Tinh chỉnh DCVC-RT cho bài toán **phát hiện đối tượng** (object detection). Trên bộ dữ liệu SFU-HW-Objects, tiết kiệm bitrate trung bình **−62,7 %** so với codec HEVC ở cùng độ chính xác (mAP@0.5), tốt hơn DCVC-RT gốc (−54,9 %), tốc độ mã hóa gần như không đổi. Đây là **nền tảng** của mọi thứ còn lại.

**Đang làm (mở rộng đa tác vụ, phục vụ khóa luận).** Câu hỏi: *cùng một bitstream có thể phục vụ đồng thời phát hiện đối tượng và phân đoạn thể hiện (instance segmentation) không, và nếu có thì mất bao nhiêu?*

**Tình trạng thật, nói thẳng.**
- Hạ tầng đã dựng xong: code huấn luyện, đường đánh giá bằng bitstream thật, anchor HEVC, các notebook Kaggle.
- **Chưa có kết quả nào trả lời được câu hỏi nghiên cứu.** Mọi run huấn luyện tiếp từ checkpoint của bài đều làm **chất lượng detection kém đi** so với checkpoint gốc, từ +23 điểm BD-rate (run chỉ segmentation, ít hỏng nhất) đến +76 điểm (run chỉ detection, hỏng nhất). Run chỉ có detection, tức đơn tác vụ, lại hỏng nặng nhất. Nghĩa là lỗi nằm ở **cách huấn luyện tiếp**, không phải ở việc thêm segmentation; đa tác vụ chưa được thử công bằng.
- Đã loại bỏ hai giả thuyết về nguyên nhân (BatchNorm, clone trôi). Giả thuyết đang mở: **cách tính rate** trong huấn luyện khác với script gốc của bài. Run kiểm tra (R3x) đang chờ chạy.
- Đã đọc tài liệu: hầu hết người khác **đóng băng codec và chỉ huấn luyện adapter nhỏ** khi làm đa tác vụ. Đây là hướng dự phòng chính (mục 8).

**Việc tiếp theo gần nhất:** chạy R3x, xây đường đo mask-mAP cho segmentation (chưa có), rồi quyết định giữa tiếp tục sửa công thức huấn luyện hay chuyển sang hướng adapter.

---

## 1. Kiến thức nền cho người mới

### 1.1 Nén video và vì sao có "nén cho máy"

Một video thô rất lớn, nên phải nén. Đo hiệu quả bằng **bitrate** (bit trên giây) hoặc **bpp** (bit trên pixel). Nén càng mạnh, hình càng mất chi tiết.

Truyền thống, codec được thiết kế cho **mắt người**: giữ những chi tiết người nhìn thấy. Nhưng ngày càng nhiều video được gửi thẳng cho **máy phân tích** (camera giám sát gửi lên đám mây để đếm người, nhận diện xe). Máy không cần mọi chi tiết mà người thấy; nó cần đúng những thông tin phục vụ tác vụ. Bỏ phần thừa đi sẽ tiết kiệm bit. Hướng nghiên cứu này gọi là **Video Coding for Machines (VCM)**.

### 1.2 Codec nơ-ron

Codec truyền thống (H.264, HEVC, VVC) là thuật toán thiết kế tay. **Codec nơ-ron** thay các khối bằng mạng học được, huấn luyện đầu-cuối để cực tiểu

> `L = R + λ · D`

trong đó `R` là bitrate ước lượng, `D` là độ méo (distortion), `λ` là hệ số cân bằng giữa hai thứ. Đổi `λ` là đổi điểm làm việc trên đường cong rate–distortion. Với codec cho người, `D` là sai khác pixel (MSE, SSIM). **Với VCM, `D` được đo bằng thứ liên quan đến tác vụ của máy.**

### 1.3 DCVC-RT, codec nền của dự án

**DCVC-RT** (Jia et al., CVPR 2025) là codec video nơ-ron chạy thời gian thực (hơn 100 FPS ở 1080p trên GPU mạnh). Những gì cần biết:

- Có hai mạng: **DMCI** nén khung đầu tiên (I-frame, mã hóa độc lập), **DMC** nén các khung sau (P-frame, mã hóa dựa vào khung trước).
- **DPB** (decoded picture buffer): bộ đệm giữ *đặc trưng* (feature) của khung vừa giải mã để dự đoán khung kế tiếp. Điều quan trọng cho dự án: DPB lưu **feature, không lưu ảnh**. Ảnh giải mã `x̂` chỉ là đầu ra của một khối cuối (`recon_generation_net`).
- Không tính chuyển động tường minh; thay vào đó dùng mô hình thời gian ngầm. Một **latent** độ phân giải thấp duy nhất được lượng tử hóa rồi mã hóa entropy thành bitstream.
- **Điều khiển tốc độ bằng module bank**: một mô hình duy nhất, nhiều mức chất lượng. Chỉ số chất lượng `q` chọn ra các bộ tham số tương ứng (`q_encoder`, `q_decoder`, `q_feature`, `q_recon`). Nhờ vậy không cần huấn luyện một mô hình cho mỗi mức bitrate.

### 1.4 Các thước đo dùng trong dự án

| Thuật ngữ | Ý nghĩa |
|---|---|
| **mAP@0.5** | Độ chính xác trung bình của detector khi một hộp dự đoán được tính đúng nếu chồng lấp ≥ 50 % với nhãn. Cao hơn là tốt |
| **mAP@0.5:0.95** | Trung bình mAP qua nhiều ngưỡng chồng lấp (0.5 đến 0.95), khắt khe hơn |
| **BD-rate** | Chênh lệch bitrate trung bình giữa hai codec **tại cùng độ chính xác**, tính từ hai đường cong rate–accuracy. **Âm = tiết kiệm bit hơn codec neo (anchor)**. Ví dụ −70 % nghĩa là cần ít hơn 70 % số bit so với anchor để đạt cùng mAP |
| **Anchor** | Codec dùng làm mốc so sánh. Ở đây là **HEVC** (x265) |
| **Đánh giá bằng bitstream thật** | Mã hóa thật thành tệp bit, giải mã lại, rồi mới chấm mAP; bitrate là số byte thực. Khác với dùng bitrate *ước lượng* của mạng |

Trong dự án, khi so hai run ta thường nhìn **hiệu `BD(run) − BD(paper)`** (đơn vị "điểm"), cả hai tính so với cùng anchor trong cùng phiên. Dương nghĩa là run đó kém checkpoint bài; số này chỉ so được trong cùng một anchor.

---

## 2. Bài báo RIVF, nền tảng của dự án

*"Task-Oriented Variable-Rate DCVC-RT for Object Detection"*, bài hội nghị 6 trang. Nguồn LaTeX: `rivf_paper/`.

### 2.1 Vấn đề và ý tưởng

DCVC-RT gốc tối ưu cho chất lượng ảnh, không cho máy. Bài **tinh chỉnh DMC** để ảnh giải mã giữ được thông tin mà detector cần, mà **không đổi kiến trúc hay bitstream** của DCVC-RT.

### 2.2 Phương pháp

```
                    HUẤN LUYỆN (chỉ để tạo tín hiệu học)
   khung gốc x_t ──► [YOLOv5s tầng 0–4, ĐÓNG BĂNG "teacher"] ──► đặc trưng F_t ─┐
        │                                                                          ├─► MSE = D_t
        ▼                                                                          │
   DCVC-RT (DMC huấn luyện được) ──► ảnh giải mã x̂_t ──► [bản sao YOLOv5s tầng 0–4,
        │                                                  HUẤN LUYỆN ĐƯỢC "clone"] ──► F̂_t ─┘
        └─► rate R_t (ước lượng từ mô hình entropy)

   Loss mỗi khung:  L_t = R_t + λ(q) · w_t · D_t        Loss clip = trung bình 6 khung P
```

- **Teacher/clone**: hai bản của 5 lớp đầu (tầng 0–4) mạng phát hiện YOLOv5s. Teacher đóng băng nhìn khung gốc; clone nhìn khung đã giải mã. `D_t` là MSE giữa hai đặc trưng. Không cần nhãn đối tượng.
- **Đóng băng**: DMCI, teacher, phần còn lại của detector. **Huấn luyện**: DMC và clone.
- **Điều kiện hóa tốc độ**: mỗi clip lấy ngẫu nhiên chỉ số cơ sở `q ∈ {0…63}`. Chỉ số hiệu dụng từng khung P: `[q+8, q, q+4, q, q+4, q]` (giữ nguyên cách phân bổ chất lượng theo thời gian của DCVC-RT). Hệ số cân bằng: **`λ(q) = 64^(q/63)`**, tức đi từ 1 đến 64.
- **Trọng số thời gian** `w = [0.5, 1.2, 0.5, 0.9, 0.5, 1.2]`, đặt theo kinh nghiệm.

### 2.3 Huấn luyện

Vimeo-90K Septuplet (mỗi mẫu 7 khung: 1 khung I + 6 khung P), cắt ngẫu nhiên 256×256, lật ngang. Adam, **learning rate 1e-6**, mixed precision, batch toàn cục 4, **30 epoch**. Checkpoint chọn theo **val loss trên Vimeo** (kết quả là epoch 11), không phải theo BD-rate.

### 2.4 Đánh giá

Bộ **SFU-HW-Objects-v1**, hai lớp: **Class D** (416×240) và **Class C** (832×480), mỗi lớp 4 chuỗi, tổng 3.800 khung. Mọi phương pháp so sánh được chấm bằng **cùng một YOLOv5s gốc đóng băng** (không phải clone), nên chênh lệch chỉ đến từ chất lượng nén. Bitrate là bitstream thật. Anchor: HEVC (x265, low-delay-P).

### 2.5 Kết quả (Bảng III của bài, trung bình 8 chuỗi, BD-rate so với HEVC, mAP@0.5 / mAP@0.5:0.95)

| Phương pháp | BD-rate |
|---|---|
| SVC Base (thăm dò) | −42,9 % / −44,6 % |
| DCVC-RT gốc | −54,9 % / −58,1 % |
| **Bài RIVF** | **−62,7 % / −64,6 %** |

Tốc độ (Tesla T4, FP16): 55,4 / 66,1 FPS (mã hóa/giải mã) so với 56,5 / 66,6 của DCVC-RT gốc. Kết quả phụ thuộc nội dung: tốt nhất trên BQSquare, RaceHorses, PartyScene; chưa tốt trên BasketballPass, BlowingBubbles, BQMall, BasketballDrill.

### 2.6 Hạn chế mà chính bài tự nêu

Checkpoint chọn theo val loss chứ không theo BD-rate nên không đảm bảo tối ưu; chưa có ablation từng thành phần; chưa so với các baseline VCM khác.

---

## 3. Hướng mở rộng: đa tác vụ

### 3.1 Mục tiêu và câu hỏi nghiên cứu

Mở rộng từ một tác vụ (detection) sang **hai tác vụ (detection + instance segmentation) từ một bitstream chung**. Ba câu hỏi:

| | Câu hỏi | Cách đo |
|---|---|---|
| **RQ1** | Thêm segmentation vào hàm mục tiêu, detection mất bao nhiêu so với chỉ train detection? | `BD(đa tác vụ) − BD(chỉ detection)` trên trục mAP |
| **RQ2** | Codec chỉ train cho segmentation thì detection còn lại bao nhiêu? | `BD(chỉ seg) − BD(chỉ detection)` |
| **RQ3** | Một bitstream chung có rẻ hơn gửi **hai bitstream riêng** (simulcast, mỗi tác vụ một codec) không? | So tổng bitrate tại cùng cặp (mAP, mask-mAP). **Cần trục mask-mAP, chưa có** |

### 3.2 "Kiến trúc A" (thiết kế đã chọn ban đầu)

Một bitstream chung, encoder không đổi, **đánh giá luôn dùng mạng gốc đóng băng** (YOLOv5s và YOLOv5s-seg). Loss:

> `L = R + λ(q) · w_t · ( α_det · D_det + α_seg · s_seg · D_seg )`,   `α_det + α_seg = 1`

- `D_det`: MSE feature ở **tầng 4** của YOLOv5s (như bài gốc).
- `D_seg`: MSE feature ở **tầng 17** của YOLOv5s-seg, mạng đóng băng hoàn toàn.
- `s_seg = 0,0535`: hệ số đưa `D_seg` về cùng bậc với `D_det` (đo ở chế độ eval trên Vimeo). `D_seg` lớn hơn `D_det` khoảng 19 lần và ổn định theo QP.
- Tên các run: `alpha_det = 1` chỉ detection, `0` chỉ segmentation, `0,5` đa tác vụ.

### 3.3 Vì sao chọn tầng 4 cho detection và tầng 17 cho segmentation

Đo **CKA** (độ tương đồng biểu diễn, không phụ thuộc hoán vị kênh) giữa teacher detection và teacher segmentation ở từng tầng, trên Vimeo:

| Tầng | Khoảng cách CKA so với mốc làm mờ | Diễn giải |
|---|---|---|
| 4 (đầu mạng) | **0,048** (nhỏ nhất) | Hai tác vụ gần như nhìn cùng một biểu diễn |
| 17 (cổ mạng, neck) | **0,203** (lớn nhất) | Hai tác vụ đã phân kỳ rõ, cần giám sát riêng |

Segmentation được giám sát ở tầng 17 vì đó là nơi nó khác detection nhiều nhất. Nhánh segmentation là mạng **đóng băng hoàn toàn**, không có bản sao huấn luyện, vì bản sao ở độ sâu 0–17 sẽ chiếm khoảng 63 % số tham số mạng và có thể nuốt mất việc của codec.

---

## 4. Cơ sở lý thuyết (tóm tắt)

Bản đầy đủ, có trích dẫn: `theoretical_foundation.md`. Bốn ý chính:

1. **VCM** (Duan et al. 2020): thay độ méo pixel bằng độ méo đo trên đặc trưng của mạng tác vụ. Đa tác vụ là mục tiêu được nêu rõ của lĩnh vực.
2. **Giám sát bằng feature với teacher đóng băng** (Hinton 2015, FitNets 2015): gradient chỉ chảy về "học sinh" (codec), teacher giữ nguyên. Cần thiết vì mAP không khả vi (qua NMS, ghép cặp).
3. **CKA** (Kornblith 2019) và tính tổng quát của tầng nông (Yosinski 2014): cơ sở chọn tầng giám sát.
4. **Mạng Gray–Wyner** (1974) và **thông tin chung**: nếu hai tác vụ có thông tin chung dương thì một bitstream chung *có thể* đạt tổng rate không tệ hơn hai bitstream riêng. Đây là phần **đã chứng minh bằng toán**, nhưng chỉ là tồn tại (achievability): không đảm bảo mạng nơ-ron huấn luyện bằng SGD với trọng số cố định sẽ đạt được (Sener & Koltun 2018). Độ lớn thật phải đo (RQ1 đến RQ3).

**Ranh giới giữa đã chứng minh và phải đo** là điều quan trọng nhất để nói đúng trong bảo vệ.

---

## 5. Hạ tầng đã dựng

### 5.1 Vị trí mã

Mọi mã mới nằm trong thư mục `multitask_exp/` trên nhánh `multitask-exp` của remote `target` (`uetot1/bla`). **Không sửa mã cũ của dự án.**

| Tệp | Vai trò |
|---|---|
| `train_multitask.py` | Vòng huấn luyện DDP: warm start, resume, giới hạn thời gian phiên Kaggle, `status.json`, `--self_check` (kiểm tra không cần dữ liệu), `--diagnose_rates` |
| `multitask_system.py` | Ghép loss: nhánh detection (`det_clone` hoặc `det_frozen`), nhánh segmentation đóng băng, `rate_mode` |
| `frozen_feature.py` | `FrozenYoloFeature`: YOLOv5 cắt tại một tầng, không cập nhật, gradient vẫn chảy về ảnh đầu vào |
| `lambda_schedule.py` | `λ(q) = 64^(q/63)`, đúng công thức bài |
| `exact_rate.py` | Bản tính rate trên ký hiệu nguyên đã làm tròn, khớp script gốc của bài |
| `paper_reference.py` | Bản sao **nguyên văn** hàm huấn luyện P-frame của script bài, dùng làm chuẩn đối chiếu. Không "dọn dẹp" |
| `checkpoints.py` | Nạp checkpoint ở mọi định dạng đã có; nhận diện checkpoint λ 1–64 |
| `data.py` | Nạp Vimeo có cắt ngẫu nhiên và lật ngang |
| `check_task_teachers.py`, `measure_task_scales.py` | Kiểm tra teacher segmentation dùng được; đo CKA và `s_seg` |
| `prepare_eval_dataset.py` | Làm sạch nhãn Class C (xem 5.3) |
| `docs/` | Tài liệu (mục 11) |
| `output/plots/` | Đồ thị đường huấn luyện của R0/R1/R2 (lần train đầu) |
| `ketqua/` | JSON kết quả đánh giá lần đầu (paper, R0, R1, R2 trên Class D, anchor cũ), lưu dạng `.txt` |

### 5.2 Các notebook Kaggle (thư mục `train_kaggle/`)

| Notebook | Việc | Cấu hình |
|---|---|---|
| `kaggle_multitask_step4_scales_timing.ipynb` | Đo CKA, `s_seg`, thời gian mỗi epoch | 2×T4 |
| `kaggle_hevc_anchor_class_cd.ipynb` | Mã hóa HEVC làm anchor cho Class C và D | 1 GPU |
| `kaggle_multitask_train_R2_R0_R1.ipynb` | Lần train đầu (R2, R0, R1) | 2×T4, ~9,6 giờ |
| `kaggle_multitask_train_R0b_R2b_bnfix.ipynb` | Train lại sau sửa BatchNorm | 2×T4, ~6,5 giờ |
| `kaggle_multitask_train_R3_R4_frozen.ipynb` | Train với detection đóng băng, không clone | 2×T4, ~6,5 giờ |
| **`kaggle_multitask_train_R3x_R4x_exact.ipynb`** | **Train với rate "exact" + cell chẩn đoán (chờ chạy)** | 2×T4, ~6,5 giờ |
| **`kaggle_multitask_eval_box_map.ipynb`** | **Đánh giá bằng bitstream thật + BD-rate (hiện đặt cho R3x/R4x/R1)** | 1 GPU, ~1,5–2 giờ mỗi checkpoint mỗi lớp |

Các notebook còn lại trong thư mục (`classb*.ipynb`, `kaggle_compare_class_d.ipynb`, `kaggle_train_lambda_shaped_group6.ipynb`, `kaggle_visualize_detection_comparison.ipynb`) là của bài RIVF trước đó.

**Quy ước:** mỗi notebook ghim vào một commit cụ thể của nhánh `multitask-exp` để tái lập được. Phiên Kaggle tối đa 12 giờ; notebook tự dừng trước hạn và **chỉ giữ output nếu bấm Save Version**, output đó attach lại để resume.

### 5.3 Dữ liệu và anchor, những chỗ dễ vướng

- **Anchor HEVC được dựng lại ở 8-bit 4:2:0 cho cả hai lớp.** Các file anchor cũ không đồng nhất: Class D là 10-bit 4:4:4, Class C là 8-bit 4:2:0. Bài báo không ghi cấu hình này. Vì vậy số BD-rate trong dự án (tính lại trên anchor mới) **không so trực tiếp được** với Bảng III của bài.
- **Nhãn Class C có hộp vượt ra ngoài khung** (toạ độ chuẩn hóa ngoài [0,1]), khiến bộ nạp từ chối đọc. `prepare_eval_dataset.py` cắt các hộp đó về trong khung, kết quả **xác định** (cùng đầu vào cho cùng đầu ra) và **được cả notebook anchor lẫn notebook đánh giá dùng chung**. Điều này quan trọng vì `evaluation_id` (dấu vân tay của toàn bộ khung và nhãn) phải trùng thì mới tính được BD-rate giữa anchor và ứng viên.
- **Nhận diện lớp theo tên chuỗi** trong manifest (BasketballPass, BQSquare, BlowingBubbles là Class D; BasketballDrill, BQMall, PartyScene là Class C), không theo tên thư mục.
- **Bản Vimeo trên Kaggle có 7.014 clip huấn luyện và 917 clip test**, không phải 91.701 như bài viết. Lịch sử huấn luyện của bài cũng cho 1.753 bước × batch 4 ≈ 7.012 mẫu mỗi epoch, khớp với con số này. **Câu "91,701 sequences" trong bài mô tả kích thước tập dữ liệu, không phải lượng dữ liệu thực tế đã dùng.**
- SFU-HW-Objects **chỉ có nhãn hộp, không có mask**. Không thể chấm segmentation trên đó.

### 5.4 Cách đánh giá

`evaluate_vcm.py` (mã của bài, không sửa): `--mode codec` mã hóa/giải mã thật rồi chấm mAP, ghi `*_results.json`; `--mode bdrate` tính BD-rate so với các anchor. Luôn bật `--force-pretrained-frontend`.

---

## 6. Hành trình thí nghiệm và kết quả

### 6.1 Diễn biến (theo thứ tự)

1. Đặt thiết kế Kiến trúc A, xác nhận λ 1–64, đo CKA (tầng 4 và 17), đo `s_seg`.
2. Train R0 (chỉ detection), R1 (chỉ segmentation), R2 (cả hai), 8 epoch mỗi run, warm start từ checkpoint bài.
3. Val loss trên Vimeo cải thiện qua các epoch (dù dao động). **Nhưng đánh giá bitstream thật cho thấy R0 và R2 sụp nặng** (mục 6.2).
4. Nghi vấn 1, **BatchNorm**: clone chạy BatchNorm ở train mode (thống kê của batch 2 clip) trong khi script gốc gọi `freeze_bn()`. Sửa (commit `33ba7d7`), train lại thành R0b, R2b.
5. Kết quả: **R0b ≈ R0**. Sửa BatchNorm không cứu được.
6. Nghi vấn 2, **clone trôi khỏi teacher**: train lại với detection đóng băng hoàn toàn, không có clone (R3, R4, commit `8e6f481`).
7. Kết quả: **R3 ≈ R0b, R4 ≈ R2b**. Clone không phải nguyên nhân.
8. Đọc lại script train của bài, tìm thấy khác biệt thật: **cách tính rate** (mục 6.4). Viết `exact_rate.py` và kiểm khớp số học với script gốc (commit `9d82c1b`). R3x và R4x chờ chạy.

### 6.2 Bảng kết quả (Class D, mAP@0.5, BD-rate so với HEVC 8-bit 4:2:0)

| Run | Trọng số loss detection | Có clone? | BD-rate | `BD(run) − BD(paper)` |
|---|---|---|---|---|
| **paper** (checkpoint gốc) | (bài) | có | **−73,2 %** | 0 |
| R0b | 100 % | có | −3,2 % | **+70,1** |
| **R3** | 100 % | **không** | +2,8 % | **+76,0** |
| R2b | 50 % | có | −15,7 % | +57,6 |
| **R4** | 50 % | **không** | −15,9 % | +57,3 |
| R1 (chỉ segmentation) | 0 % | không có nhánh detection | −50,6 % | +22,6 |

Class C cho cùng hình dạng: paper −63,3; R0b +8,4; R3 +7,6; R4 −2,8; R1 −28,5 (hiệu với paper lần lượt +72, +71, +61, +35).

R0 và R2 cũ (trước khi sửa BatchNorm, đo trên anchor cũ 10-bit 4:4:4) cho +75 và +33; không đưa vào bảng vì khác anchor, chỉ dùng để so hướng thay đổi.

**Đọc bảng:**
- Dòng R3 dương (+2,8 %) nghĩa là R3 cần **nhiều bit hơn HEVC** để đạt cùng mAP.
- **Clone hay đóng băng không tạo khác biệt** (R0b ≈ R3, R2b ≈ R4).
- Cái dự đoán được độ hỏng là **trọng số của loss detection**: 100 % hỏng nhất, 0 % hỏng ít nhất. Nghĩa là càng huấn luyện theo `D_det` bằng công thức hiện tại, detection thật càng tệ, trong khi bài gốc đạt kết quả tốt với đúng loss đó.
- Dấu hiệu khác: ở run R0 (lần đánh giá đầu), bitrate thật tại QP 0 rơi từ 0,030 xuống còn khoảng 0,011 bpp. Model bị kéo về **bitrate thấp, chất lượng thấp**.
- **Val loss trên Vimeo không dự đoán được mAP thật**: các run hỏng vẫn có val loss thấp và cải thiện qua epoch. (Không so được val loss của các run này với val loss ghi trong lịch sử train của bài, vì hai bên dùng hai cách tính rate khác nhau; chưa ai đo val loss của checkpoint gốc bằng đúng công thức của các run này. Cell chẩn đoán ở mục 6.5 sẽ làm việc đó.)

### 6.3 Giả thuyết đã loại (không xem lại)

| Giả thuyết | Bác bỏ bởi |
|---|---|
| BatchNorm của clone ở train mode gây sụp | R0b (đã sửa) ≈ R0 (chưa sửa). Hiệu ứng có thật (loss kém nhạy 2,5–7 lần với độ sáng/tương phản/màu) nhưng không phải nguyên nhân |
| Clone tự trôi khỏi teacher | R3 (không clone) ≈ R0b, R4 ≈ R2b |
| Sai công thức λ | Đã xác nhận λ 1–64 khớp bài; trùng khớp `lambda_schedule.py` |

### 6.4 Giả thuyết đang mở: cách tính rate

Script train của bài (nhúng trong notebook `kaggle_train_random_qp_lambda_1_64.ipynb` ở git HEAD) **không phải `train_base.py`**. Nó tự viết hàm `p_frame_forward` tính rate trên **ký hiệu nguyên đã làm tròn**, đúng chi phí bộ mã hóa thật. Trong khi `DMC.forward_train` (dùng cho mọi run R0–R4) tính rate trên **giá trị liên tục chưa làm tròn**.

Đo trên dữ liệu tổng hợp: ở scale nhỏ (vùng chiếm phần lớn latent khi bitrate thấp), gradient kéo phần dư về 0 của bản liên tục lớn gấp khoảng 4 lần (scale 0,2) và 1,8 lần (scale 0,3) so với chi phí thật. Cùng chiều với quan sát bitrate bị kéo xuống.

`exact_rate.py` tái tạo đúng hàm của bài; `--self_check` khẳng định `x_hat`, `bpp` bằng nhau và gradient lệch cỡ 6e-7 so với bản sao nguyên văn.

**Trạng thái: giả thuyết, chưa kiểm bằng bitstream thật.** Hai lần chẩn đoán trước đều được nêu chắc chắn rồi đều sai, nên phải coi đây là giả thuyết có phép thử quyết định. Còn các mục chưa mô phỏng trong recipe của bài: mixed precision (AMP) và trục lật ảnh.

### 6.5 Phép thử quyết định

`BD(R3x) − BD(paper)` ở Class D. Các run trước cho +70 đến +76.
- Chỉ lệch vài điểm: nguyên nhân là cách tính rate. R4x khi đó mới trả lời RQ1.
- Vẫn cỡ +70: giả thuyết sai, nguyên nhân còn lại nằm ở AMP, learning rate, hoặc chi tiết khác.

Notebook train có sẵn **cell chẩn đoán chạy trước khi train** (vài phút): định giá *cùng một model* (checkpoint bài, chưa train) bằng cả hai ước lượng rate. Hai số bpp chênh rõ thì củng cố giả thuyết; gần bằng nhau thì nên dừng trước khi tốn 6 giờ GPU.

---

## 7. Tài liệu liên quan: người khác làm thế nào

Chi tiết: `related_work_training_recipes.md`, `combination_analysis.md`. Ba mẫu hình chung:

1. **Gần như không ai tinh chỉnh toàn bộ codec tiền huấn luyện cho đa tác vụ.** Họ đóng băng codec và huấn luyện adapter nhỏ (All-in-One, MoECodec, PAT-VCM, SEC-VCM), hoặc chỉ điều khiển encoder (Ge et al. CVPR 2024), hoặc dựng kiến trúc mới rồi train từ đầu (Gray–Wyner ba kênh).
2. **Cảnh báo trực tiếp trúng vấn đề của ta:** Ge et al., mục 4.5, tinh chỉnh toàn bộ codec video cho một tác vụ thì chất lượng khung giải mã suy giảm và **suy giảm lan truyền qua các khung** vì codec tham chiếu khung trước. Họ kết luận không thể đạt được bằng tinh chỉnh đơn giản.
3. **Giữ neo pixel hoặc huấn luyện nhiều pha** (SEC-VCM, Ge et al.); trọng số tác vụ cố định là chuẩn; lợi ích đa tác vụ được báo cáo khiêm tốn (Δm ≈ +0,25).

Ba repo đã đọc mã: **TransTIC** (ICCV 2023) và **Adapt-ICMH** (ECCV 2024) đóng băng codec, chỉ huấn luyện prompt/adapter với **đúng loại loss của bài RIVF** (MSE feature trên backbone detector đóng băng), nhưng mỗi tác vụ một bitstream riêng; **SEC-VCM** tách hai đường giải mã (đường gốc nuôi DPB, đường ngữ nghĩa thứ hai tạo đầu ra cho máy) nhưng chỉ có mã suy luận, không có mã huấn luyện, trọng số không công khai.

**Phát hiện cấu trúc có lợi:** DMC lưu `feature` (không phải ảnh) vào DPB và tạo `x̂` qua khối `recon_generation_net` riêng. Vì vậy thêm một đầu ra thứ hai đọc cùng `feature` **không chạm vào chuỗi tham chiếu**, đúng ý tưởng dual-path của SEC-VCM.

---

## 8. Hướng phát triển

### 8.1 Bước nền, cần làm bất kể chọn hướng nào

1. **Chạy R3x** (train) rồi đánh giá Class D. Đây là phép thử rẻ nhất cho giả thuyết rate.
2. **Xây đường đo mask-mAP cho segmentation.** Chưa có. Cần dữ liệu có mask thật (**KITTI-MOTS** là phương án đã dự kiến) hoặc nhãn giả từ YOLOv5s-seg trên video gốc (yếu hơn, chỉ đo độ nhất quán). Đây là đường găng của RQ2 và RQ3.
3. **Đo mask-mAP của chính checkpoint bài** làm mốc. Không cần huấn luyện.

### 8.2 Cây quyết định sau R3x

```
                       Kết quả BD(R3x) − BD(paper), Class D
                       /                                    \
              chỉ vài điểm                         vẫn cỡ +70
                  |                                      |
   Giữ Kiến trúc A, dùng rate "exact":            Chuyển hướng (một trong hai):
   - chạy R4x → trả lời RQ1                       (a) Kiến trúc B: đóng băng codec + adapter
   - đo mask-mAP → RQ2, RQ3                       (b) Train từ đầu bằng script của bài
   - đo xung đột gradient (chưa làm)                  (thêm nhánh seg vào loss), ~3 phiên
```

### 8.3 "Kiến trúc B": đóng băng codec RIVF + đầu ra/adapter mới

Ba mức, đi từ rẻ và an toàn đến giá trị cao (phân tích đầy đủ và cơ sở lý thuyết ở `combination_analysis.md`):

- **Mức 0: đóng băng hoàn toàn codec RIVF + một đầu ra segmentation riêng.** Bitstream không đổi, nên detection giữ đúng −73 % theo cấu tạo. Rate cố định nên không dính giả thuyết ước lượng rate. Huấn luyện nhẹ hơn nhiều vì không cần lan truyền ngược qua chuỗi DMC. **Trần thông tin:** đầu ra mới chỉ diễn đạt lại thông tin mà bitstream đã mang (bất đẳng thức xử lý dữ liệu), nên khoảng cách mask-mAP so với codec tối ưu cho segmentation đo đúng lượng thông tin segmentation bị encoder bỏ. Đây là phép đo chẩn đoán có giá trị dù kết quả ra sao.
- **Mức 1: thêm adapter encoder dùng chung** (kiểu SFMA của Adapt-ICMH và thiết kế của All-in-One) để `y` mang thông tin cho cả hai tác vụ. Khởi tạo bằng không để bắt đầu đúng tại chất lượng của bài. Đây mới là mức "một bitstream chung" thực sự.
- **Mức 2: đường cơ sở simulcast.** Mỗi tác vụ một adapter và một bitstream riêng theo kiểu TransTIC, cộng tổng bitrate. Cho RQ3.

**Hạn chế kỹ thuật:** mọi thay đổi kiến trúc cần lớp con `DMC` riêng trong `multitask_exp/` và một đường nạp/đánh giá riêng, vì `evaluate_vcm.py` dựng `DMC()` gốc và không nạp được tham số adapter. Làm được mà không sửa mã cũ, nhưng tốn công.

### 8.4 Nếu kết quả là âm hoặc khiêm tốn

Vẫn là kết luận bảo vệ được nếu đo cẩn thận: có khung lý thuyết (Gray–Wyner), có phép đo trần thông tin (Mức 0), có so sánh với simulcast, và dải kết quả đã được báo cáo trong tài liệu (lợi ích đa tác vụ khiêm tốn) nằm cùng vùng.

### 8.5 Cần làm cho phần viết khóa luận

- Vẽ lại sơ đồ kiến trúc (`multitask_architecture.png` hiện vẫn vẽ biến thể có clone và nhãn "FIX" BatchNorm, đã lỗi thời).
- Cập nhật `theoretical_foundation.md` theo thiết kế cuối cùng.
- Ghi rõ trong khóa luận cả các giả thuyết đã bị bác bỏ, vì chúng là kết quả thực nghiệm có giá trị.

---

## 9. Rủi ro và điểm cần lưu ý

| Điều cần lưu ý | Ghi chú |
|---|---|
| **Chưa có kết quả nào cho RQ1–RQ3** | Chưa có mốc đơn tác vụ nào giữ được chất lượng gốc |
| **Nhiều chẩn đoán đã sai** | BatchNorm rồi clone trôi đều được nêu chắc chắn rồi bị bác. Giả thuyết rate hiện tại phải được coi như giả thuyết |
| **Chưa có đường đánh giá segmentation** | SFU chỉ có nhãn hộp; đây là đường găng |
| **Không bài nào trong tài liệu chứng minh trên DCVC-RT** | Port adapter sang DMC là kỹ thuật chưa kiểm chứng |
| **Số BD-rate không so trực tiếp với Bảng III của bài** | Khác anchor (đã dựng lại 8-bit 4:2:0) và khác cách gộp (gộp theo lớp thay vì trung bình từng chuỗi) |
| **Bài viết nói 91.701 chuỗi Vimeo** | Thực tế dùng 7.014 clip/epoch; nên làm rõ trong bài |
| **Anchor gốc của bài không đồng nhất giữa hai lớp** | Class D 10-bit 4:4:4, Class C 8-bit 4:2:0; bài không ghi cấu hình |
| **`train_base.py` không phải script train của bài** | Đừng dùng nó để tái tạo. Script thật ở `git show HEAD:kaggle_train_random_qp_lambda_1_64.ipynb` |
| **Ngân sách tính toán** | Kaggle T4, phiên tối đa 12 giờ; một run train ~3,3 giờ, một checkpoint đánh giá ~1,5–2 giờ mỗi lớp. Ước lượng ban đầu của mỗi checkpoint (10–30 phút) đã sai khoảng 10 lần |
| **GPU máy cá nhân** | RTX 3050 Laptop, không có phần mở rộng CUDA cần cho bitstream thật; chỉ dùng cho kiểm tra logic (`--self_check`), không dùng để đo tốc độ hay đánh giá |

---

## 10. Hướng dẫn thực hành

### 10.1 Chạy R3x rồi đánh giá (đường đi ngắn nhất tới phép thử quyết định)

1. **Train:** mở `train_kaggle/kaggle_multitask_train_R3x_R4x_exact.ipynb` trên Kaggle. GPU **T4 ×2**, Internet bật. Attach: Vimeo-90K Septuplet; `cvpr2025_image.pth.tar` và `cvpr2025_video.pth.tar`; dataset `u-30epoch` (checkpoint bài). Notebook chạy self-check trên GPU, rồi cell chẩn đoán rate, rồi mới train. Nếu chạy tương tác từng cell, đọc kết quả cell chẩn đoán trước khi chạy cell train; nếu chạy "Save & Run All" thì đọc kết quả ở cuối. Xong thì **Save Version** và gửi `multitask_runs/summary.json` cùng `multitask_runs/diagnosis/rate_diagnosis.json`.
2. **Đánh giá:** mở `train_kaggle/kaggle_multitask_eval_box_map.ipynb`. GPU T4 (1 GPU). Attach: Class D và Class C thô; output notebook anchor (`hevc_anchor/`); `cvpr2025_image.pth.tar`; `u-30epoch`; output train lần đầu (`multi_task`, lấy R1); **output train R3x/R4x**; output phiên đánh giá trước (`eval_box/`, để tự lấy lại kết quả `paper` và R1). Nếu chỉ cần kết quả quyết định sớm, đặt `CLASSES = ['class_d']` trong cell cấu hình (~3,5 giờ).
3. **Đọc kết quả:** file `eval_box/summary_box_map.json`, cell cuối notebook cũng in bảng.

### 10.2 Kiểm tra logic cục bộ (không cần GPU hay dữ liệu)

```
python -m multitask_exp.train_multitask --self_check --device cpu
```

Kiểm tra: λ 1–64 đúng, các nhánh đóng băng không bị cập nhật, checkpoint nạp được bằng `evaluate_vcm.load_codec_checkpoint`, bản rate "exact" khớp bản sao script của bài (cả giá trị lẫn gradient). Mất vài phút trên CPU.

---

## 11. Bản đồ repo và quy ước làm việc

```
E:\LAB\SVC
├── rivf_paper/            Bài báo RIVF (LaTeX, hình, bảng, references.bib)
├── dcvc_rt/               Mã codec DCVC-RT (mô hình DMC, DMCI, lớp CUDA)
├── svc_machine/           Hàm loss và trích feature YOLO của bài
├── evaluate_vcm.py        Đánh giá codec + BD-rate (mã của bài, KHÔNG sửa)
├── evaluate_hevc.py       Mã hóa và đánh giá HEVC (dựng anchor)
├── train_base.py          KHÔNG phải script train của bài (xem mục 9)
├── train_kaggle/          Mọi notebook Kaggle (đặt mọi notebook mới ở đây)
├── multitask_exp/         Toàn bộ phần mở rộng đa tác vụ (xem mục 5.1)
│   └── docs/              Tài liệu (bên dưới)
└── ...
```

**Tài liệu trong `multitask_exp/docs/`:**

| Tệp | Nội dung |
|---|---|
| `00_TONG_QUAN_DU_AN.md` | Tệp này |
| `theoretical_foundation.md` | Cơ sở lý thuyết đầy đủ có trích dẫn; bảng ánh xạ quyết định thiết kế sang cơ sở và trạng thái chứng minh. Đã đính chính 2026-09-21 |
| `related_work_training_recipes.md` | Bảy bài về cách người khác huấn luyện codec đa tác vụ, và cảnh báo của Ge et al. |
| `combination_analysis.md` | Phân tích kết hợp SEC-VCM, TransTIC, Adapt-ICMH; Kiến trúc B ba mức; cơ sở lý thuyết theo độ chắc chắn |
| `multitask_references.bib` | Trích dẫn bổ sung (dùng chung key với `rivf_paper/references.bib` cho các bài trùng) |
| `multitask_architecture.png`, `figure/` | Sơ đồ kiến trúc (**lỗi thời**) và đoạn LaTeX để chèn |

**Quy ước bắt buộc (đã thống nhất):**
1. Mọi thử nghiệm mới nằm trong `multitask_exp/`; **không sửa mã cũ**.
2. Mọi notebook Kaggle đặt trong `train_kaggle/`.
3. **Chỉ đẩy lên remote `target` (`uetot1/bla`), tuyệt đối không đẩy lên repo DCVC.** Đẩy nhánh `multitask-exp`.
4. Khi lấy công thức từ bài, đọc **script trong notebook ở git HEAD**, không dùng `train_base.py`.
5. Nêu giả thuyết bị bác bỏ một cách thẳng thắn thay vì bảo vệ chúng.

---

## 12. Bảng thuật ngữ

| Thuật ngữ | Giải thích |
|---|---|
| **VCM** | Video Coding for Machines: nén video cho máy phân tích |
| **DCVC-RT** | Codec video nơ-ron thời gian thực (Microsoft) làm nền |
| **DMCI / DMC** | Mạng nén khung I (độc lập) / mạng nén khung P (dựa khung trước) |
| **DPB** | Bộ đệm khung đã giải mã; ở DCVC-RT lưu feature |
| **Latent** | Biểu diễn nén của một khung, được lượng tử hóa rồi mã hóa entropy thành bitstream |
| **Module bank** | Các bộ tham số theo chỉ số chất lượng `q`, cho phép một mô hình chạy nhiều mức bitrate |
| **q, q_eff** | Chỉ số chất lượng cơ sở (0–63) / chỉ số hiệu dụng từng khung (0–71) |
| **λ(q)** | Hệ số cân bằng rate và độ méo tác vụ; `64^(q/63)`, từ 1 đến 64 |
| **bpp, BD-rate** | Bit trên pixel / chênh bitrate tại cùng độ chính xác; âm là tốt |
| **mAP, mask-mAP** | Độ chính xác trung bình của detection (hộp) / của segmentation (mask) |
| **Teacher / clone / student** | Mạng đóng băng cho tín hiệu học / bản sao huấn luyện được / phần được huấn luyện |
| **Đóng băng (frozen)** | Không cập nhật trọng số; gradient vẫn có thể chảy qua để dạy phần trước nó |
| **Feature matching** | Bắt đặc trưng của ảnh giải mã khớp đặc trưng của ảnh gốc qua một mạng đóng băng |
| **CKA** | Centered Kernel Alignment, đo độ giống của hai biểu diễn |
| **Anchor** | Codec mốc để tính BD-rate (HEVC/x265) |
| **Warm start** | Bắt đầu huấn luyện từ trọng số đã train sẵn thay vì từ đầu |
| **Simulcast** | Gửi một bitstream riêng cho mỗi tác vụ (đường cơ sở đối chiếu với bitstream chung) |
| **Adapter / prompt** | Mô-đun nhỏ thêm vào mạng đóng băng, chỉ huấn luyện phần này |
| **Rate "surrogate" / "exact"** | Cách tính rate trong huấn luyện: trên giá trị chưa làm tròn (mọi run R0–R4) / trên ký hiệu đã làm tròn (script gốc, R3x/R4x) |
| **R0, R0b, R1, R2, R2b, R3, R4, R3x, R4x** | Tên các run huấn luyện. `R0*`/`R3*`: chỉ detection. `R1`: chỉ segmentation. `R2*`/`R4*`: cả hai. Hậu tố `b`: đã sửa BatchNorm. `R3/R4`: detection đóng băng. `x`: rate "exact" |
| **RQ1–RQ3** | Ba câu hỏi nghiên cứu (mục 3.1) |
| **SFU-HW-Objects** | Bộ video chuẩn có nhãn hộp; Class C (832×480) và Class D (416×240) |
| **Vimeo-90K** | Bộ video dùng để huấn luyện, mỗi mẫu 7 khung |
| **KITTI-MOTS** | Bộ video có mask thật, dự kiến cho đánh giá segmentation |

---

## 13. Nguồn chính

- Bài RIVF: `rivf_paper/` (Chương 3 phương pháp, Chương 4 thực nghiệm, `chapters/table/table1_bd.tex`).
- Script train của bài: `git show HEAD:kaggle_train_random_qp_lambda_1_64.ipynb`.
- Jia et al., *Towards Practical Real-Time Neural Video Compression*, CVPR 2025 (DCVC-RT).
- Duan et al., *Video Coding for Machines*, IEEE TIP 2020.
- Hadizadeh & Bajić, *Learned Scalable Video Coding for Humans and Machines*, EURASIP JIVP 2024 (công thức feature-matching mà bài RIVF kế thừa).
- Ge et al., *Task-Aware Encoder Control for Deep Video Compression*, CVPR 2024.
- Gray & Wyner 1974; de Andrade, Harell & Bajić, ICLR 2026 (thông tin chung).
- TransTIC (ICCV 2023), Adapt-ICMH (ECCV 2024), SEC-VCM (IEEE TIP 2026): xem `combination_analysis.md`.
