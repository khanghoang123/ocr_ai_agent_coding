# Báo cáo Kỹ thuật OCR (Phase 2B)

## 1. Tổng quan
Hệ thống OCR hiện tại đã hoàn thành Phase 2B, tập trung vào việc **Tự động hóa nghiên cứu (AutoResearch)** và **Tinh chỉnh hình học dòng chữ (Geometry-Aware Refinement)**. Mục tiêu là tối ưu hóa khả năng nhận diện tiếng Việt viết tay trên các nền giấy phức tạp (giấy kẻ ngang, giấy tập học sinh).

## 2. Các cải tiến chính

### 2.1. AutoResearch Loop
Thay vì điều chỉnh tham số thủ công, chúng tôi triển khai một vòng lặp tự động:
- **Cấu hình giả thuyết**: Tự động thay đổi `crop_padding_ratio`, `upscale_factor`, và các cờ chức năng.
- **Leaderboard**: Theo dõi và xếp hạng các cấu hình dựa trên tổ hợp metric (CER, WER, Digit Noise).
- **Kết quả**: Cấu hình `crop_pad_008` đang dẫn đầu với Score **0.0262**.

### 2.2. LineRefiner (Geometry Validation)
Đây là trái tim của Phase 2B, giúp giải quyết các vấn đề về layout:
- **Hybrid Mask/Projection**: Kết hợp mặt nạ foreground và hình chiếu ngang để xác định biên dòng chữ cực kỳ chính xác.
- **Neighbor Clamping**: Ràng buộc biên dựa trên dòng trên và dòng dưới, ngăn chặn việc cắt dính chữ của dòng khác.
- **Split Recovery**: Khi phát hiện dòng quá cao hoặc có khoảng trống lớn giữa các hàng chữ trong một box, hệ thống tự động tách thành nhiều dòng con.
- **Deskew & Rectification**: Ước lượng góc nghiêng và làm phẳng dòng chữ trước khi đưa vào bộ nhận diện.

### 2.3. Hybrid Detector Support
Hỗ trợ cơ chế dự phòng (fallback) linh hoạt:
- Ưu tiên sử dụng PaddleOCR model.
- Nếu lỗi môi trường (Runtime error), hệ thống tự động chuyển sang **OpenCV Row Projection Fallback**, đảm bảo pipeline không bao giờ bị dừng đột ngột.

## 3. Phân tích Chiến lược Huấn luyện (10k vs 50k)

Dự án đã tiến hành so sánh chéo giữa hai chiến lược dữ liệu chính để đánh giá tác động của **Pseudo-labels**:

| Experiment | Iters | Test CER | Exact Match | Ghi chú |
| :--- | :--- | :--- | :--- | :--- |
| **Baseline** | 10k | 4.38% | 32.50% | Hiệu suất ổn định nhưng hội tụ chậm. |
| **Experiment B** | 10k | **4.23%** | **32.80%** | **Pseudo-labels giúp tăng tốc học tập.** |
| **Baseline** | 50k | **2.73%** | **50.46%** | **Tốt nhất khi train lâu.** |
| **Experiment B** | 50k | 2.98% | 46.55% | Dính nhiễu từ nhãn AI khi train quá sâu. |

### 3.1. Đánh đổi giữa Số lượng và Chất lượng
Thực nghiệm cho thấy một hiện tượng thú vị: **Dữ liệu giả (Pseudo-labels) cực kỳ hữu ích để vượt qua giai đoạn đầu nhanh chóng**, nhưng khi tiến tới giới hạn hội tụ, độ chính xác của nhãn (Label Noise) trở thành yếu tố quyết định.

- **Bias vào Ground Truth**: Tập test được trích xuất từ cùng một nguồn với tập GT huấn luyện. Do đó, mô hình Baseline được tiếp xúc 100% với phong cách và quy chuẩn gán nhãn của con người, dẫn tới việc tối ưu hóa tốt hơn trên tập test này khi train đủ lâu.
- **Giới hạn của LLM Labeling**: Dù Gemini Flash đã lọc kỹ, các nhãn pseudo vẫn có thể có sai lệch nhỏ về mặt ngữ nghĩa hoặc định dạng so với Ground Truth, tạo ra một "noise floor" mà mô hình không thể vượt qua nếu không cải thiện quy trình lọc.

## 4. Các hướng cải tiến (Next Steps)
1. **Nâng cấp bộ lọc (Advanced Filtering)**:
   - Sử dụng **Gemini 1.5 Pro** hoặc **GPT-4o** để thực hiện "Double-Check" các mẫu có độ tự tin thấp.
   - Áp dụng **Entropy-based filtering** để loại bỏ các mẫu mà model dự đoán không dứt khoát.
2. **Active Learning Loop**:
   - Chỉ gán nhãn pseudo cho các vùng dữ liệu mà model hiện tại đang gặp khó khăn (OOD detection).
3. **Multi-Stage Fine-tuning**:
   - Huấn luyện giai đoạn đầu trên tập hỗn hợp (GT + Pseudo) để tận dụng số lượng.
   - Huấn luyện giai đoạn cuối (Annealing) chỉ trên tập GT thuần túy để xóa bỏ bias từ nhiễu nhãn.
4. **Perspective Correction**: Hiện tại mới chỉ xử lý dòng (line-level), chưa xử lý toàn trang (page-level). Dự kiến triển khai ở Phase 3.
5. **VietOCR Fine-tuning**: Bộ nhận diện hiện tại là pre-trained. Việc fine-tune trên dữ liệu handwriting thực tế sẽ giúp giảm CER sâu hơn nữa.
6. **Speed**: Quá trình tinh chỉnh hình học tốn khá nhiều tài nguyên CPU. Cần tối ưu hóa bằng Vectorization hoặc C++.

## 5. Kết luận
Phase 2B đã thiết lập một nền tảng vững chắc cho việc nghiên cứu tự động. Hệ thống không chỉ nhận diện tốt hơn mà còn có khả năng "tự học" để tìm ra cấu hình tối ưu cho từng loại dữ liệu khác nhau.

---
*Cập nhật lần cuối: 26/04/2026*
