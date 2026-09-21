# Người khác huấn luyện codec đa tác vụ / codec video cho máy như thế nào

*Ghi chú đọc tài liệu (2026-09-21). Chỉ ghi những gì đọc được trong bài; mục nào chỉ đọc được phần tóm tắt thì đánh dấu.*

## Bảng so sánh cách huấn luyện

| Bài | Codec nền | Cái gì được huấn luyện | Giám sát tác vụ | Ghi chú |
|---|---|---|---|---|
| Hadizadeh & Bajić, *Learned scalable video coding for humans and machines* (EURASIP JIVP 2024) | CANF-VC tiền huấn luyện | Base layer: **toàn bộ mạng**, 10 epoch, **lr 1e-6**; enhancement layer sau, base đóng băng | MSE feature ở front-end YOLOv5s (bản sao huấn luyện được), `λ_base ∈ {2,4,8,16}` cố định | Đây chính là công thức mà bài RIVF kế thừa. Một mô hình cho mỗi λ, không phải λ ngẫu nhiên 1–64 |
| Ge et al., *Task-Aware Encoder Control for Deep Video Compression* (CVPR 2024) | FVC, DCVC, TCM tiền huấn luyện | **Chỉ bộ điều khiển encoder** (dự đoán mode, chọn GoP); trọng số codec giữ nguyên | Loss detection thật qua YOLO đóng băng, `R + λ·L_det` | −25 % bitrate với **một** decoder. Xem mục "Cảnh báo về tinh chỉnh toàn bộ" |
| *All-in-One Transfer Image Compression…* (arXiv 2504.12997) | Codec ảnh tiền huấn luyện | **Chỉ adapter nhẹ** (2,1 % / 1,7 % số tham số codec), codec đóng băng | Loss tác vụ thật qua các mạng đóng băng, trọng số cố định lấy từ MTI-Net | **Một bitstream chung** cho nhiều tác vụ. 10 epoch, lr 1e-4. Lợi ích đa tác vụ nhỏ: Δm +0,25 / +0,28 |
| *MoECodec* (arXiv 2606.21033) | TIC tiền huấn luyện | **Chỉ router và expert**, phần còn lại đóng băng | Loss tác vụ thật qua Faster/Mask R-CNN đóng băng | Không thảo luận xung đột giữa các tác vụ |
| *PAT-VCM* (arXiv 2604.13294) | Cosmos tokenizer, **đóng băng** | Nhánh phụ trợ riêng cho mỗi tác vụ (~5,5 M tham số) | Distillation feature trên backbone đóng băng | **Không có so sánh** với codec đầy đủ riêng từng tác vụ (bài tự nêu là chỗ thiếu) |
| *SEC-VCM* (arXiv 2510.15347) | DCVC-HEM | Pha 1 (40 epoch): codec đầy đủ với **MSE pixel + LPIPS**. Pha 2 (10 epoch): **module mới** (semantic decoder, fusion) | Consistency MSE feature với **3 mạng đóng băng** (ResNet-18, Swin-T, DINOv2-S) | Nhiều tác vụ từ một codec. Bài nêu rõ: **huấn luyện nhiều pha là thiết yếu, cần MSE pixel để ổn định dự đoán liên khung** |
| de Andrade, Harell & Bajić, *Lossy Common Information in a Learnable Gray-Wyner Network* (ICLR 2026) | **Từ đầu**, kiến trúc ba kênh (chung + hai riêng) | Toàn bộ codec | Loss tác vụ thật qua mạng đóng băng | Nêu giới hạn: thông tin chung trong bài toán nén có mất mát "thường không đạt được" |

**Mức độ đã đọc.** Ge et al. đọc trực tiếp từ PDF (các trang 1–9), nên các câu trích và số liệu của bài này chắc chắn. Với bảy bài còn lại trong bảng, chi tiết huấn luyện được trích từ bản HTML/PMC đầy đủ **qua một bước tóm tắt tự động**, nên các con số cụ thể (tỉ lệ tham số, lr, số epoch, Δm) cần đối chiếu lại với bản gốc trước khi dẫn vào khóa luận. **Không đọc được toàn văn:** Chamain et al., *End-to-end optimized image compression for machines, a study* (arXiv 2011.06409) — chỉ có tóm tắt; DeepSVC (ACM MM 2023) — chỉ biết qua lời dẫn của Ge et al.

## Cảnh báo về tinh chỉnh toàn bộ codec video

Ge et al., mục 4.5: họ thử tinh chỉnh **toàn bộ FVC** với `L = R + λ1·MSE + λ2·L_det` cho tác vụ MOT. Kết quả: độ chính xác tăng nhưng **bitrate tăng đáng kể**. Lời giải thích của tác giả: chất lượng khung giải mã bị suy giảm khi tối ưu cho tác vụ, và vì codec video tham chiếu các khung trước, **suy giảm này lan truyền qua các khung**. Họ kết luận rằng hiệu năng theo tác vụ "không thể đạt được bằng cách đơn giản tinh chỉnh một DVC", dẫn chiếu thêm DeepSVC.

Chiều thay đổi ở đó (bitrate tăng) ngược với chiều của chúng ta (bitrate giảm, mAP sụp), nhưng cơ chế nghi ngờ cùng họ: tinh chỉnh toàn bộ một codec P-frame với mục tiêu tác vụ làm bộ đệm tham chiếu (DPB) dần lệch phân phối.

## Ba mẫu hình chung

1. **Gần như không ai tinh chỉnh toàn bộ codec tiền huấn luyện cho đa tác vụ.** Họ hoặc đóng băng codec và huấn luyện adapter/expert/nhánh phụ (All-in-One, MoECodec, PAT-VCM, SEC-VCM), hoặc chỉ điều khiển encoder (Ge et al.), hoặc dựng kiến trúc mới rồi huấn luyện từ đầu (Gray-Wyner).
2. **Giữ neo pixel hoặc huấn luyện nhiều pha.** Ge et al. giữ `MSE` pixel trong loss; SEC-VCM giữ MSE + LPIPS ở pha thứ hai và nói rõ cần pha MSE trước để ổn định. Vòng train của chúng ta chỉ có feature MSE, không có neo pixel nào.
3. **Trọng số tác vụ cố định là chuẩn**, không học. Lợi ích đa tác vụ được báo cáo là **khiêm tốn** (Δm ≈ +0,25) và phụ thuộc kiến trúc.

## Hệ quả cho khóa luận

- Công thức "tinh chỉnh toàn bộ DMC bằng feature MSE" (Kiến trúc A) **ít tiền lệ hơn** so với công thức đóng băng codec + adapter. Nó không sai về nguyên tắc, nhưng dễ gặp đúng hiện tượng Ge et al. mô tả.
- Hướng đã được kiểm chứng trong tài liệu và rẻ nhất: **đóng băng hoàn toàn codec RIVF** (đã giải tốt detection), thêm adapter nhỏ khởi tạo bằng không (nên bắt đầu đúng tại chất lượng gốc) và huấn luyện adapter bằng loss đa tác vụ. Mọi suy giảm khi thêm segmentation khi đó nhìn thấy được và kiểm soát được.
- Kết quả âm/khiêm tốn vẫn nằm trong dải đã được báo cáo, không phải dấu hiệu ta làm sai một cách bất thường.
