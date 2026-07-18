# Thay đổi kiến trúc và hàm loss

## 1. Bỏ Rationale Statistics

- Stage 3 không còn tính hoặc trả về các đặc trưng tổng hợp `max`, `mean`, `entropy`, `sum`.
- Cổng trộn giữa rationale pooling và fallback pooling được tính trực tiếp từ ba biểu diễn học được:
  - `rationale_pool`
  - `fallback_pool`
  - `global_context`
- Stage 4 không còn nối thêm vector thống kê 8 chiều vào đầu vào của `verdict_mlp`.
- `verdict_mlp` hiện nhận trực tiếp biểu diễn tương tác `z` có kích thước `hidden`.

## 2. Đơn giản hóa objective

Hàm loss chính chỉ còn hai thành phần:

```text
L_total = 0.33 * L_RE + 0.67 * L_TP
```

Các loss phụ đã được bỏ khỏi đường huấn luyện:

- alignment loss
- consistency loss
- teacher-forcing KL loss
- các cấu hình contrastive/MoE/uncertainty weighting cũ

Teacher forcing vẫn có thể trộn nhãn rationale vào đầu vào pooling theo lịch, nhưng không tạo thêm loss phụ.

## 3. Các file chính đã sửa

- `src/models/pooling.py`
- `src/models/td_head.py`
- `src/models/pipeline.py`
- `src/losses/multitask_loss.py`
- `src/losses/factory.py`
- `src/trainer/engine.py`
- `config/config.yaml`
- `README.md`

`src/trainer/pipeline_utils.py` đã được xóa vì chỉ phục vụ tính rationale statistics.

## 4. Lưu ý checkpoint

Do kích thước tham số của Stage 3 và Stage 4 đã thay đổi, checkpoint cũ không thể nạp trực tiếp bằng `strict=True`. Nên huấn luyện lại từ đầu với kiến trúc mới.
