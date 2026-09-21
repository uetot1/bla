# Kết hợp SEC-VCM, TransTIC, Adapt-ICMH với DCVC-RT-VCM của bài RIVF

*Phân tích 2026-09-21. Mức đã đọc: **mã nguồn của cả ba repo** (đọc trực tiếp, các khẳng định về code bên dưới đã kiểm chứng); **toàn văn** chỉ đọc SEC-VCM qua bản HTML; TransTIC và Adapt-ICMH mới đọc README, mã và tóm tắt, chưa đọc toàn văn. Số liệu hiệu năng của các bài lấy từ tóm tắt, cần đối chiếu bản gốc trước khi dẫn.*

## 1. Mỗi repo thực sự làm gì (kiểm chứng trong mã)

| | Codec nền | Cái được huấn luyện | Loss | Nhiều tác vụ? |
|---|---|---|---|---|
| **TransTIC** (ICCV 2023) | TIC (Swin, ảnh), **đóng băng** | Chỉ tham số có tên chứa `prompt` (`configure_optimizers`) | `TaskLoss` = 0,2 × trung bình MSE trên feature FPN p2–p6 của backbone Faster R-CNN đóng băng, cộng `VPT_lmbda × bpp`. lr 1e-4, 40 epoch | Mỗi tác vụ một bộ prompt riêng, **mỗi tác vụ một bitstream riêng** |
| **Adapt-ICMH** (ECCV 2024) | TIC, **đóng băng** (`if "sfma" not in k: requires_grad=False`) | Adapter SFMA dạng residual `x + (s_mod + f_mod)·factor`, 3 cái ở encoder, 3 ở decoder | Cùng `TaskLoss` như TransTIC: `task_lmbda × perc_loss + bpp`. lr 1e-4, 40 epoch | Mỗi tác vụ một bộ adapter, mỗi tác vụ một bitstream |
| **SEC-VCM** (TIP 2026) | DCVC-HEM (có ước lượng chuyển động) | Codec cơ sở huấn luyện đầy đủ bằng MSE + LPIPS; sau đó module mới | MSE feature với ResNet-18, Swin-T, DINOv2-S đóng băng + ràng buộc entropy hai chiều | Một mô hình cho nhiều tác vụ. **Chỉ có mã suy luận, không có script huấn luyện, trọng số không công khai** |

**Ba điều rút ra, cả ba đều có trong mã:**

1. **Loss của bài RIVF là loss chuẩn của lĩnh vực.** TransTIC và Adapt-ICMH dùng đúng dạng MSE feature trên backbone đóng băng, không phải loss detection có nhãn. Lựa chọn của bài không lạ.
2. **Cả hai bài ảnh đóng băng codec.** Đây là khác biệt lớn nhất với các run R0–R4 của ta, vốn tinh chỉnh toàn bộ DMC.
3. **Không bài nào chứng minh "một bitstream cho nhiều tác vụ" bằng cách chỉ đổi adapter.** TransTIC và Adapt-ICMH huấn luyện mô hình riêng cho từng tác vụ; nhiều tác vụ ở đó nghĩa là nhiều bitstream. SEC-VCM là bài duy nhất một mô hình nhiều tác vụ, nhưng ta không tái tạo được nó.

## 2. Cái gì chuyển được sang DCVC-RT

Cấu trúc DMC (đã đọc `video_model.py`) có những đặc điểm quyết định:

- **DPB lưu `feature`, không lưu ảnh**: `compress` gọi `add_ref_frame(feature, None)`, còn `x_hat = recon_generation_net(feature, q_recon)` chỉ là đầu ra. Nghĩa là đường tham chiếu thời gian và đầu ra ảnh **đã tách sẵn** về cấu trúc. Đây chính là ý tưởng "dual-path" của SEC-VCM (`ref_frame` cho DPB, `ref_frame_semantic` cho đầu ra máy), và trong DMC ta có nó gần như miễn phí: thêm đầu ra thứ hai đọc cùng `feature` mà không chạm vào chuỗi tham chiếu.
- Encoder/decoder toàn là khối tích chập (`DepthConvBlock`), không có Swin. **Prompt của TransTIC không chuyển thẳng được** (prompt gắn vào token Swin). **Adapter SFMA chuyển được** vì nó làm việc trên feature map tích chập.
- Đường suy luận CUDA (`forward_cuda`) dùng nhân dung hợp bên trong `Encoder`/`Decoder`/`ReconGeneration`. Adapter đặt **bên ngoài**, trên `y` trước khi lượng tử hóa hoặc trên `feature` sau decoder, không đụng tới các nhân đó.
- Adapter ở encoder chỉ đổi `y`, không đổi cú pháp bitstream hay decoder. Đây cũng là tiền đề của Ge et al. (giữ decoder cố định).
- SEC-VCM dựa trên DCVC-HEM có ước lượng chuyển động; DCVC-RT đã bỏ nó. **Không port được mã, chỉ lấy ý tưởng.**

Hạn chế kỹ thuật thật: mọi thay đổi kiến trúc buộc phải có lớp con `DMC` riêng trong `multitask_exp/` và một đường nạp/đánh giá riêng, vì `evaluate_vcm.py` dựng `DMC()` gốc và nạp state dict, không nạp được tham số adapter. Làm được mà không sửa mã cũ, nhưng tốn công.

## 3. Kiến trúc kết hợp đề xuất ("Kiến trúc B")

Ba mức, đi từ rẻ và an toàn đến giá trị cao:

- **Mức 0 — codec RIVF đóng băng hoàn toàn + đầu ra segmentation riêng** (ý tưởng SEC-VCM và decoder prompt của TransTIC). Bitstream **không đổi**, nên detection giữ đúng −73 % của bài theo cấu tạo. Chỉ huấn luyện đầu ra mới bằng MSE feature với YOLOv5s-seg tầng 17. Vì bitstream cố định, rate không đổi và **giả thuyết ước lượng rate (`exact_rate.py`) không liên quan**. Huấn luyện nhẹ hơn nhiều: không cần lan truyền ngược qua chuỗi DMC.
- **Mức 1 — thêm adapter encoder dùng chung** (Adapt-ICMH SFMA, và thiết kế "encoder adaptor không phụ thuộc tác vụ + decoder riêng từng tác vụ" của All-in-One). Đây là mức mới thực sự "một bitstream chung": `y` được điều chỉnh để mang thông tin cho cả hai tác vụ, hai đầu ra đọc cùng `y`. Khởi tạo tầng chiếu lên bằng không để bắt đầu **đúng tại chất lượng của bài** (bản gốc của Adapt-ICMH khởi tạo std 0,02 chứ không phải không; khởi tạo bằng không là chỉnh sửa của ta).
- **Mức 2 — đường cơ sở simulcast** theo đúng kiểu TransTIC/Adapt-ICMH: mỗi tác vụ một bộ adapter và một bitstream riêng, cộng tổng bitrate. Cho RQ3 mà không cần huấn luyện từ đầu.

## 4. Cơ sở lý thuyết (phân loại theo độ chắc chắn)

**Đã chứng minh (toán):**

- **Trần thông tin của Mức 0.** Chuỗi Markov `X → y → f → x̂_k` (`f` là `feature`, `x̂_k` là đầu ra của đầu k). Theo bất đẳng thức xử lý dữ liệu, `I(X; x̂_k) ≤ I(X; y)`. Đầu ra mới chỉ có thể diễn đạt lại thông tin bitstream đã mang; nó không tạo ra chi tiết mà encoder đã bỏ. Hệ quả kiểm chứng được: khoảng cách mask-mAP giữa "đầu ra segmentation trên codec RIVF" và "codec tối ưu cho segmentation" đo đúng lượng thông tin segmentation bị mất ở phía encoder. Đây là phép đo chẩn đoán có giá trị dù kết quả ra sao.
- **Cấu trúc chung/riêng (Gray–Wyner).** Kênh chung là `y`; các đầu ra là bộ giải mã riêng **không có bit riêng**. Nó tương ứng góc "joint coding" của mạng Gray–Wyner. Bài de Andrade et al. (ICLR 2026) báo cáo góc này cho hiệu năng tương tự các kiến trúc phức tạp hơn nhưng mất kiểm soát mịn. Mức 2 (mỗi tác vụ bitstream riêng) là đường cơ sở độc lập của định lý.
- **Adapter encoder dùng chung là bộ mã hóa cho hai độ méo.** Cực tiểu `R + Σ λ_k·D_k` trên encoder chung là Lagrangian của bài toán RD hai độ méo, với `R(D₁,D₂) ≤ R₁(D₁) + R₂(D₂)`. Trọng số cố định chỉ chạm phần lồi của biên Pareto (Sener & Koltun); giới hạn này đã nêu ở tài liệu chính.

**Lập luận có căn cứ thực nghiệm nhưng chưa chứng minh:**

- **Vùng tin cậy (trust region) cho chuỗi tham chiếu.** Trạng thái đệ quy `f_{t+1} = D(y_t, ctx(f_t))`. Nếu mỗi bước lệch khỏi chuỗi RIVF tối đa `ε` và decoder có hằng số Lipschitz `L` theo `ctx`, độ lệch tích lũy chặn bởi `Σ (1+L)^{t-s}·ε`. Adapter dạng residual khởi tạo bằng không cộng hệ số `factor` nhỏ giữ `ε` nhỏ theo cấu tạo; tinh chỉnh toàn bộ trọng số không có chặn nào. Phù hợp với quan sát của Ge et al., mục 4.5 (tinh chỉnh toàn bộ FVC làm chất lượng khung suy giảm và lan truyền qua các khung). **Kiểm chứng được:** đo `‖f_t − f_t^{RIVF}‖` theo `t` cho từng phương án.
- **Giám sát bằng nhiều teacher đóng băng.** SEC-VCM dùng ba teacher cùng lúc và báo cáo lựa chọn backbone ảnh hưởng đáng kể; TransTIC/Adapt-ICMH cho tiền lệ loss FPN feature. Đây là tiền lệ cho tổng `α_det·D_det + α_seg·s·D_seg` của ta, không phải chứng minh nó tối ưu.

**Phải đo:** độ lớn mọi thứ trên, và liệu Mức 1 có đạt cận Gray–Wyner hay không (RQ1–RQ3).

## 5. Rủi ro và điều chưa biết

- **Không bài nào chứng minh kết quả trên codec video DCVC-RT.** Hai bài ảnh dùng codec Swin; việc port adapter sang DMC là kỹ thuật chưa được kiểm chứng.
- **Lợi ích đa tác vụ được báo cáo khiêm tốn** ở các bài gần nhất (All-in-One: Δm ≈ +0,25). Kết quả nhỏ không phải dấu hiệu làm sai.
- **Mức 0 có thể cho segmentation kém** nếu encoder RIVF đã bỏ thông tin segmentation. Đó là kết quả có nghĩa (phép đo trần ở trên), không phải thất bại.
- **Chưa có đường đánh giá segmentation nào.** SFU-HW-Objects chỉ có nhãn hộp. Cần KITTI-MOTS (mask thật) hoặc nhãn giả từ YOLOv5s-seg trên video gốc (đo độ nhất quán, yếu hơn). Đây là việc phải làm dù chọn hướng nào, và là đường găng.
- **Bản thân lỗi R0–R4 chưa được giải thích bằng ba bài này.** Mức 0 và Mức 1 tránh nó (codec đóng băng; Mức 0 miễn nhiễm với nghi vấn ước lượng rate), nhưng không chứng minh nguyên nhân.

## 6. Thứ tự làm đề xuất

1. Xây đường đánh giá mask-mAP (chưa có, cần bất kể hướng nào), đo mask-mAP của **chính checkpoint bài RIVF** làm mốc. Không cần huấn luyện.
2. Mức 0: đầu ra segmentation trên codec đóng băng. Rẻ, không rủi ro, đo trần thông tin.
3. Mức 1 nếu Mức 0 cho thấy còn khoảng trống đủ lớn; Mức 2 làm đường cơ sở simulcast.
4. Song song: R3x (đang chờ chạy) vẫn kiểm giả thuyết ước lượng rate, cần cho mọi phương án có huấn luyện encoder.
