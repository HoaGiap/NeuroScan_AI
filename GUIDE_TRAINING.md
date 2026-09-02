# 📖 Hướng Dẫn Tự Huấn Luyện & Quản Lý Mô Hình NeuroScan AI

Tài liệu này hướng dẫn chi tiết cách bạn tự thực hiện quá trình huấn luyện (train), đánh giá (evaluate) và khởi chạy ứng dụng phân loại u não **NeuroScan AI**.

---

## 📁 1. Cấu trúc Thư mục Dữ liệu (Dataset Structure)

Đảm bảo dữ liệu ảnh MRI được sắp xếp đúng cấu trúc trong thư mục `data/`:

```text
NeuroScan_AI/
└── data/
    ├── Training/           # Dữ liệu dùng để huấn luyện (~4.853 ảnh)
    │   ├── glioma/         # U thần kinh đệm
    │   ├── meningioma/     # U màng não
    │   ├── notumor/        # Không có u (Bình thường)
    │   └── pituitary/      # U tuyến yên
    ├── Validation/         # Dữ liệu kiểm định khi train (~859 ảnh)
    │   ├── glioma/
    │   ├── meningioma/
    │   ├── notumor/
    │   └── pituitary/
    └── Testing/            # Dữ liệu kiểm thử độc lập (Test Set - 1.311 ảnh)
        ├── glioma/
        ├── meningioma/
        ├── notumor/
        └── pituitary/
```

---

## 🚀 2. Câu lệnh Huấn luyện Mô hình (Training)

Để bắt đầu huấn luyện mô hình, mở **Terminal / Command Prompt** tại thư mục dự án và chạy câu lệnh:

### 🔹 Huấn luyện mô hình EfficientNetV2-S (Khuyên dùng):
```bash
python train.py --backbone efficientnet_v2_s --epochs 30 --batch 16 --workers 4
```

### 🔹 📜 Giải thích các Tham số (Arguments):
| Tham số | Ý nghĩa | Mặc định | Gợi ý |
| :--- | :--- | :---: | :--- |
| `--backbone` | Kiến trúc mô hình (`efficientnet_v2_s`, `resnet50`, `convnext_small`) | `both` | `efficientnet_v2_s` |
| `--epochs` | Số chu kỳ huấn luyện toàn bộ tập dữ liệu | `50` | `30` - `50` |
| `--batch` | Số lượng ảnh xử lý trong một batch | `32` | `16` (Nếu GPU 4GB VRAM) |
| `--lr` | Tốc độ học (Learning Rate) | `0.0001` | `1e-4` |
| `--workers` | Số luồng CPU tải ảnh song song | `0` | `4` (Giúp train nhanh hơn) |
| `--data` | Đường dẫn đến thư mục dữ liệu | `./data` | `./data` |

### 📂 Trọng số Sau khi Train:
Sau khi huấn luyện xong, mô hình có độ chính xác cao nhất sẽ được tự động lưu vào:
`checkpoints/efficientnet_v2_s/<ngày_giờ>/efficientnet_v2_s_best.pth`

---

## 📊 3. Đánh giá & Kiểm thử Mô hình

### 🔹 1. Kiểm tra độ chính xác trên Tập Test độc lập (`data/Testing`):
```bash
python scripts/eval_testset.py
```
*Kết quả sẽ hiển thị tỉ lệ dự đoán đúng (%) chi tiết cho từng loại u.*

### 🔹 2. Kiểm tra thuật toán Tiền xử lý Auto-Crop sát viền não:
```bash
python scripts/test_crop_demo.py
```

---

## 🌐 4. Khởi chạy Ứng dụng Web & Kiểm tra API

### 🔹 1. Chạy Web Server Flask:
```bash
python app.py
```
*`app.py` sẽ tự động tìm và nạp checkpoint mới nhất vừa được tạo trong thư mục `checkpoints/`.*

### 🔹 2. Mở giao diện trên Trình duyệt:
Truy cập đường dẫn: **`http://localhost:5000`**

### 🔹 3. Kiểm tra tự động các API endpoint:
(Mở một Terminal khác trong khi `app.py` đang chạy):
```bash
python scripts/test_demo.py
```

---

## 💡 5. Mẹo Kỹ thuật & Lưu ý (Best Practices)
1. **Tiền xử lý tự động**: Toàn bộ thuật toán **Auto Brain Crop (OpenCV)** + **Letterboxing (Pad vuông 1:1)** + **CLAHE (Cân bằng tương phản)** đã được tích hợp tự động vào `src/dataset.py` và `src/inference/engine.py`. Bạn không cần chỉnh sửa lại mã nguồn khi thêm ảnh mới.
2. **GPU Memory**: Nếu gặp lỗi `Out of Memory (OOM)` trên GPU, hãy giảm `--batch 16` xuống `--batch 8`.
