# 🧠 NeuroScan AI — Brain Tumor MRI Classifier

---

> Ứng dụng hỗ trợ y tế chuyên sâu sử dụng công nghệ Deep Learning (CNN & Transformer) để nhận diện và phân loại khối u não từ ảnh MRI tĩnh với độ chính xác cao, tích hợp Explainable AI (XAI) nhằm tăng tính minh bạch trong chẩn đoán lâm sàng.

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![JavaScript](https://img.shields.io/badge/JavaScript-F7DF1E?style=for-the-badge&logo=javascript&logoColor=black)

---

## 📑 Table of Contents

- [📖 Giới Thiệu Đồ Án](#-giới-thiệu-đồ-án)
- [🌟 Tính Năng Chính](#-tính-năng-chính)
- [🏗️ Kiến Trúc Hệ Thống](#️-kiến-trúc-hệ-thống)
- [🚀 Hướng Dẫn Cài Đặt](#-hướng-dẫn-cài-đặt-chi-tiết)
- [🐳 Triển Khai với Docker](#-triển-khai-với-docker)
- [📝 Scripts và Commands](#-scripts-và-commands)
- [📚 Tài Liệu API](#-tài-liệu-api)
- [🧪 Testing và Quality](#-testing-và-quality-assurance)
- [🛠️ Công Nghệ Sử Dụng](#️-công-nghệ-sử-dụng)
- [📞 Liên Hệ](#-liên-hệ)

---

## 📖 Giới Thiệu Đồ Án

Đây là đồ án nghiên cứu ứng dụng AI vào y tế. Dự án phát triển hệ thống hỗ trợ chẩn đoán khối u não thông qua phân tích ảnh MRI sử dụng các công nghệ Computer Vision tiên tiến và cung cấp kết quả chẩn đoán tự động trên nền tảng web trực quan.

### 👥 Thông Tin Sinh Viên

- **Sinh viên thực hiện**: Hoa Giap - Quoc Dat
- **Lĩnh vực đề tài**: Medical AI / Deep Learning

### 🎯 Mục Tiêu Đồ Án

- Xây dựng hệ thống phân loại 4 nhãn ảnh MRI khối u ở não (Bình thường, U màng não, U tuyến yên, U thần kinh đệm).
- Áp dụng kiến trúc mạng Neural Network hiện đại: Đơn giản hoá kiến trúc trên môi trường bằng EfficientNet-V2-S (với khả năng hỗ trợ mở rộng thêm ResNet, Swin Transformer).
- Tích hợp XAI (Explainable AI) bằng thuật toán Grad-CAM++ để giải thích kết quả dự đoán của AI cho bác sĩ, giúp minh bạch hoá hành vi ra quyết định.
- Xây dựng Web Interface trực quan với framework Flask API mạnh mẽ để hỗ trợ tải ảnh và xuất báo cáo.

---

## 🌟 Tính Năng Chính

### 🩺 Chẩn Đoán Khối U

- **Kiến Trúc Mô Hình Hiện Đại**: Sử dụng mô hình EfficientNet-V2-S mạnh mẽ để tối ưu tốc độ và độ chuẩn xác trong nhận diện MRI.
- **Phân Loại Định Phân Tự Động**: Nhận định và cảnh báo một trong 4 loại khối u (Bình thường, U màng não, U tuyến yên, U thần kinh đệm).
- **Mức Độ Nguy Hiểm & Khuyến Nghị**: Cung cấp đánh giá mức độ nghiêm trọng và đưa ra các khuyến nghị, lời khuyên y tế ban đầu nhanh chóng.

### 🤖 Tính Năng XAI (Explainable AI Thông Minh)

Hệ thống cung cấp tính minh bạch (XAI) nhằm tăng độ tin cậy để bác sĩ có thể đánh giá và chẩn đoán:

- **Tạo Heatmap Trực Quan**: Áp dụng thuật toán Grad-CAM++ để highlight những vùng pixels / cấu trúc não bộ ảnh hưởng trức tiếp đến quyết định của model.
- **So Sánh Chéo Modals/Kiến trúc**: Tính năng hỗ trợ chạy đa mô hình giúp bác sĩ có cái nhìn tương phản logic (nếu cung cấp nhiều checkpoint weight).

### 📋 Bộ Công Cụ Lâm Sàng & Báo Cáo Y Tế (Clinical Suite)

- **📄 Xuất Báo Cáo Y Tế Chuẩn PDF**: Xuất phiếu chẩn đoán y tế hoàn chỉnh (Header bệnh viện, mã bệnh án, ảnh đối chiếu 1:1 MRI & Grad-CAM++, bảng % xác suất, khuyến nghị lâm sàng, chữ ký bác sĩ) với một chạm thông qua `html2pdf.js` hoặc chế độ in chuẩn A4.
- **🏥 Hỗ Trợ Định Dạng Y Tế Chuẩn DICOM (`.dcm`)**: Đọc và giải mã dữ liệu điểm ảnh y tế bằng `pydicom`, tự động trích xuất metadata (Patient ID, Name, Age, Sex, Modality) điền vào hồ sơ bệnh nhân.
- **🔍 Bộ Công Cụ Tương Tác Ảnh Y Khoa (Medical Viewport Tools)**: Tích hợp thanh công cụ điều chỉnh Độ tương phản (Contrast/Windowing), Độ sáng (Brightness), Đảo màu âm bản (Invert), Phóng to/Thu nhỏ (Zoom/Pan), Khôi phục gốc (Reset).
- **👤 Quản Lý Thông Tin Ca Khám**: Form quản lý hồ sơ bệnh nhân, hỗ trợ tự sinh thông tin lâm sàng mẫu để demo nhanh chóng.

---

## 🏗️ Kiến Trúc Hệ Thống

### 📋 Tổng Quan Kiến Trúc

Hệ thống sử dụng **kiến trúc Client-Server** nguyên khối tích hợp chạy Load Models Machine Learning Local, bao gồm:

- **Frontend Application**: Vanilla JS (ES6+), thao tác DOM trực tiếp, quản lí call API ẩn AJAX hiện đại để tối ưu băng thông.
- **Backend API**: Python - Flask framework xử lý các Endpoint RESTFul, validate đầu vào.
- **Inference Engine AI**: Module PyTorch quản lý vòng đời (load in-memory, forward pass prediction, tensor preprocessing) của các deep learning weights.

### 🎯 Sơ Đồ Kiến Trúc (Text Block)

```
  [Người Dùng Bác Sĩ]
       |
       |  (Tải Upload Ảnh MRI Định Dạng Chuẩn)
       v
  [Frontend (HTML/CSS DOM)]
       |
       |  (Tạo POST Request Multipart Form File)
       v
  [Backend (Flask Server Engine)]
       |
       |  (Tiền xử lý ảnh: Normalize Tensor, Augment Test)
       v
  [PyTorch AI Engine Layer]
       |
       |--(Kiến tạo Inference Predict vs EfficientNet)
       |--(Kích hoạt xử lý Hook Activation Map tạo Grad-CAM++ Heatmap Image Base64)
       |
       <--(Đóng khung Result Payload: Tên U, % Probabilities, Data URI Base64)--|
  [Backend Cấu trúc Response & Trả Client Render HTML]
```

### ⚙️ Cấu Trúc Backend & Engine Architecture

```
brain_tumor/
├── src/
│   ├── config.py         # ⚙️ Cấu hình hệ thống (IMG_SIZE, PORT...)
│   ├── dataset.py        # 🗃️ Đọc files MRI, Transforms & Augmentation parameters
│   ├── models/           # 🧠 Định nghĩa cấu trúc Neural Networks & Grad-CAM++
│   ├── inference/        # 🎮 Engine Pipeline logic đưa Tensor ảnh predict và pipeline Ensemble
│   └── utils/            # 🔧 Utilities phụ trợ (Metrics tracking logs)
├── static/               # 🎨 Tài nguyên tính cho Native Frontend Web (CSS Styles, JS Scripts)
├── templates/            # 🖼️ Templates Layout base của Jinja2 cho UI Web
├── checkpoints/          # ⚖️ Lưu trữ các phiên bản checkpoint trained weights (.pth) đã ra lò
├── tests/                # 🧪 Bài Unit Testing tự động đánh giá CI codebase
├── app.py                # 🚀 File Máy Chủ Entrypoint (Bắt đầu Flask System Server)
├── train.py              # 📈 File Script để Pipeline Data và Train cấu trúc Model Deep Learning
├── Dockerfile            # 🐳 Container Image Creation blueprint
├── docker-compose.yml    # 📦 Orchestration Setup Multi-architecture Services
└── requirements.txt      # 📦 Thông tin thư viện library dependencies của Python Runtime
```

---

## 🚀 Hướng Dẫn Cài Đặt Chi Tiết

### 📋 Yêu Cầu Hệ Thống Căn Bản

- **Python Runtime** (Khuyến khích phiên bản >= 3.8).
- **Phần Cứng Đồ Hoạ**: Cực kì khuyến nghị có NVIDIA GPU kết hợp **CUDA Toolkit** để quá trình Prediction/Inference hoặc Restrain nhanh hơn gấp nhiều lần.
- Môi trường Git Command Line để tương tác hệ thống lấy dự án.

### 🛠️ Quy Trình Cài Đặt Setup Nội Bộ

#### 1. Lấy Source Dự Án Về Máy

```bash
git clone https://github.com/HoaGiap/brain_tumor.git
cd brain_tumor
```

#### 2. Cài Đặt Thư Viện Python Phụ Thuộc (Dependencies)

_Khuyến nghị thiết lập cách ly (Virtual Environment) để tránh đụng độ rác ở không gian Python gốc: _

```bash
# Thiết lập không gian ảo (bằng công cụ venv tích hợp)
python -m venv venv

# Active Environment (Với nền tảng Windows PowerShell)
.\venv\Scripts\activate
# Đối với người dùng nền tảng Linux / MacOS
source venv/bin/activate

# Tải Thư Viện về dự án
pip install -r requirements.txt
```

#### 3. Chạy System Framework Server API Chế Độ Dev

Ứng dụng hiện tại đã được nâng cấp với tính năng **Tự động nhận diện trọng số mới nhất** (Auto-Detect Checkpoint). Mọi file trọng số huấn luyện (đuôi `.pth`) khi nằm trong thư mục `checkpoints/efficientnet_v2_s/` sẽ được app tự động quét và nạp file mới nhất.

Mở terminal và gõ dòng lệnh cực kì đơn giản sau:

```bash
python app.py
```

#### 4. Sử Dụng và Khai Khác Tính Năng Web

- API Interface Frontend: Mở bằng Browser thông dụng qua địa chỉ **http://localhost:5000** 🟢
- Bạn có thể tải một tấm ảnh `JPEG` bệnh án sọ não và test dự báo thử nghiệm API System.

---

## 🐳 Triển Khai với Docker

Kiến trúc Docker hóa mang tới việc chạy AI Platform bất kì nơi nào mà không sợ thiếu tương thích Package Python Libraries. Tốt cho Testing CICD.

### Mở Node Trực Tiếp

```bash
# Lệnh Auto setup Docker Network Background chạy detached
docker-compose up -d

# Hoặc Render Single Image cho containerization có kèm quyền Hardware GPU execution (NVIDIA Runtime)
docker build -t neuroscan-ai .
docker run --gpus all -p 5000:5000 neuroscan-ai

# Debug Logging System của Docker
docker-compose logs -f
```

---

## 📝 Scripts và Commands

### ⚙️ Script Huấn Luyện AI Models Chuyên Môn

Trong trường hợp chuyên gia/y khoa có thêm cơ sở dữ liệu bệnh án mới cần nâng độ chính xác của Base Model AI, phân bố vào thư mục `data/train` & `data/test`. Tiến hành nạp Model Train Parameter Setting bằng Cú pháp:

```bash
# Tiến hành Training cho dòng Model cơ sở EfficientNet bản V2 Small trong 50 Chu kì (Epoch)
python train.py --model efficientnet_v2_s --epochs 50 --batch_size 16 --lr 0.0001
```

### 📊 Xem Dashboard Huấn Luyện AI Realtime (Sử Dụng TensorBoard)

Tích hợp API Board nhằm xem Live Metrics cho Data Insight Metrics, Loss Curves:

```bash
python -m tensorboard.main --logdir checkpoints/
# Xem thông tin thống kê qua http://localhost:6006
```

---

## 📚 Tài Liệu API

Nếu một Developer hay dự án Di Động muốn tích hợp Module Chuẩn Đoán U Não, dự án sử dụng Base URL Endpoint duy nhất giao tiếp qua Data Form:

### `POST /predict`

- **Tác vụ Server**: Xử lí Tensor phân loại khối u từ File Binary.
- **Định dạng Request Data Body (Multipart Form-Data)**:
  - `image` (Type: Image File / required): File ảnh MRI cần đánh giá.
  - `gradcam_model` (Type: string / optional): Trỏ model nhận diện vẽ Explaintion bản đồ mức độ, ví trị (VD: `efficientnet_v2_s`). Mặc định auto chạy bản đồ.

**Ví Dụ Request Output Json Result:**

```json
{
  "prediction": {
    "class": "glioma",
    "class_vi": "U thần kinh đệm",
    "confidence": 99.2,
    "has_tumor": true
  },
  "probabilities": {
    "glioma": { "score_pct": 99.2, "label_vi": "U thần kinh đệm" },
    "notumor": { "score_pct": 0.5, "label_vi": "Bình thường" }
  },
  "gradcam": {
    "efficientnet_v2_s": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgA..."
  },
  "severity": { "level": "high", "label": "Cao" },
  "recommendation": "Cần sinh thiết và tham vấn bác sĩ..."
}
```

---

## 🧪 Testing và Quality Assurance

Đồ án có hỗ trợ Module Test Automation để Tracking Test Suite cho Backend Module Layer:

```bash
# Kích hoạt toàn bộ Test Script trong folder /tests / Unit Test Validation check lỗi Logic
python -m unittest discover tests
```

---

## 🛠️ Công Nghệ Sử Dụng

- **Thuật Toán AI (PyTorch Core)**: Xây dựng bằng `PyTorch` thực thi `Deep Learning ConvNet` & Thuật toán hiển thị `Grad-CAM++`.
- **System Service Web Layer (Python Flask)**: `Flask` kết nối HTTP Restful & Base Jinja Template.
- **Thiết Kế DOM Render UI/X**: Hoạt họa trên web thông qua `HTML5/CSS3` + `Javascript Fetch Protocol`.
- **Tăng Cường Chẩn Đoán (Albumentations)**: Hỗ trợ biến đổi dữ liệu chuyên khoa mạnh mẽ. Đầu tư Preprocessing ảnh.
- **Container Server**: Containerizing ứng dụng đồng nhất với Engine `Docker` & `Docker Compose Multi-system`.

---

## 📞 Liên Hệ

Mọi góp ý về kiến trúc, model checkpoint hoặc có nhu cầu hợp tác tích hợp Module AI trong y tế chuyên án này, vui lòng để lại vấn đề hoặc báo lỗi trong mục Tracker Github Repo.

_Disclaimer: Dự án NeuroScan AI nhằm mục đích nghiên cứu Hệ thống AI/Deep Learning Y khoa tiên tiến, nội dung đánh giá có phần trăm lỗi thực tế ngụy dương/ngụy âm. Chỉ hỗ trợ, không thể đứng ra thay thế chẩn đoán sau cùng của bác sĩ._
