import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import argparse
import logging
import glob
from PIL import Image
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

from src.inference.engine import InferenceEngine
from src.config import SERVER_CONFIG, CLASS_NAMES, CLASS_VI

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# ─── Global Engine ─────────────────────────────────────────────────────────────
engine = InferenceEngine()          # Khởi tạo InferenceEngine

def get_latest_checkpoint(model_name: str):
    """Finds the most recent .pth file in checkpoints/{model_name}/**/"""
    base_dir = os.path.join("checkpoints", model_name) # Đường dẫn đến thư mục checkpoints
    if not os.path.exists(base_dir): # Kiểm tra thư mục có tồn tại không
        return None
    
    # Tìm tất cả file .pth
    pth_files = glob.glob(f"{base_dir}/**/*.pth", recursive=True) # Tìm tất cả file .pth trong thư mục checkpoints
    if not pth_files: # Kiểm tra có file .pth nào không
        return None
        
    # Lấy file có thời gian sửa đổi gần nhất
    return max(pth_files, key=os.path.getmtime) # Trả về file .pth có thời gian sửa đổi gần nhất

def load_models_to_engine(args):
    """Load models based on CLI arguments or auto-detect."""
    configs = [
        ("resnet50",           args.resnet),
        ("efficientnet_v2_s",  args.effnet_v2),
        ("convnext_small",     args.convnext),
        ("efficientnet",       args.effnet),
        ("swin_t",             args.swin_t),
        ("swin_b",             args.swin_b),
    ]
    for name, arg_path in configs:
        # Nếu user không truyền path vào, tự đi tìm file mới nhất
        path = arg_path if arg_path else get_latest_checkpoint(name) # Lấy path từ arg_path hoặc tìm file mới nhất
        
        if path:
            if engine.load_model(name, path): # Load model
                logger.info(f"Loaded {name} from {path}")
            else:
                logger.warning(f"Failed to load {name} from {path}")

# ─── Routes ────────────────────────────────────────────────────────────────────
@app.route("/", methods=["GET"]) # Route cho trang chủ
def index():
    return render_template("index.html") # Render trang index.html

@app.route("/health", methods=["GET"]) # Route cho health check
def health():
    return jsonify({
        "status": "ok", # Trạng thái
        "models_loaded": list(engine.models.keys()), # Danh sách các model đã load
        "device": str(engine.device), # Thiết bị sử dụng
    })

import io
import base64
from src.mri_preprocessing import MRIPreprocessor

@app.route("/preview_dicom", methods=["POST"])
def preview_dicom():
    """Endpoint to quickly parse and preview a DICOM file without full inference."""
    if "image" not in request.files:
        return jsonify({"error": "No DICOM file provided"}), 400
    file = request.files["image"]
    try:
        pil_img, metadata = MRIPreprocessor.read_dicom_file(file.stream)
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        b64_img = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
        return jsonify({
            "status": "success",
            "preview_image": b64_img,
            "metadata": metadata
        })
    except Exception as e:
        logger.error(f"Error parsing DICOM: {str(e)}")
        return jsonify({"error": f"Không thể đọc file DICOM: {str(e)}"}), 400

@app.route("/predict", methods=["POST"]) # Route cho predict
def predict():
    if "image" not in request.files: # Kiểm tra có file image không
        return jsonify({"error": "No image provided"}), 400

    file = request.files["image"]
    gradcam_model_req = request.args.get("gradcam_model", "both") # Lấy gradcam model từ request
    filename = (file.filename or "").lower()

    try:
        metadata = {}
        preview_b64 = None

        if filename.endswith(".dcm"):
            try:
                pil_img, metadata = MRIPreprocessor.read_dicom_file(file.stream)
                buf = io.BytesIO()
                pil_img.save(buf, format="PNG")
                preview_b64 = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
            except Exception as dcm_err:
                logger.warning(f"Failed to read as DICOM, fallback to PIL: {dcm_err}")
                file.stream.seek(0)
                pil_img = Image.open(file.stream)
        else:
            try:
                pil_img = Image.open(file.stream)
            except Exception:
                file.stream.seek(0)
                # Fallback thử đọc như DICOM nếu PIL không mở được
                pil_img, metadata = MRIPreprocessor.read_dicom_file(file.stream)
                buf = io.BytesIO()
                pil_img.save(buf, format="PNG")
                preview_b64 = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

        # Log thông tin ảnh đầu vào
        logger.info(f"Input image – Mode: {pil_img.mode}, Size: {pil_img.size}")

        result = engine.predict(pil_img, gradcam_request=gradcam_model_req) # Dự đoán

        # Log kết quả dự đoán
        logger.info(f"Prediction: {result['class']} – {result['confidence']:.2f}%")
        
        # Add severity and recommendation (business logic)
        pred_class = result["prediction" if "prediction" in result else "class"] # Lấy class dự đoán
        confidence = result["confidence"] # Lấy confidence
        
        response = {
            "prediction": {
                "class": pred_class, # Class dự đoán
                "class_vi": result["class_vi"], # Class dự đoán (tiếng Việt)
                "confidence": confidence, # Confidence
                "has_tumor": pred_class != "notumor", # Có khối u không
            },
            "probabilities": {
                name: {"score_pct": score, "label_vi": CLASS_VI[name]} # Tỷ lệ dự đoán
                for name, score in result["probabilities"].items()
            },
            "per_model": result["per_model"], # Dự đoán theo từng model
            "gradcam": result["gradcam"], # Gradcam
            "severity": _get_severity(pred_class, confidence / 100.0), # Mức độ
            "recommendation": _get_recommendation(pred_class), # Khuyến nghị
            "metadata": metadata, # DICOM metadata
            "preview_image": preview_b64, # Base64 preview nếu là DICOM
        }
        return jsonify(response)
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route("/predict-batch", methods=["POST"])
def predict_batch():
    """Endpoint xử lý chuỗi nhiều lát cắt MRI (Multi-Slice Series Batch Scan)."""
    files = request.files.getlist("images")
    if not files:
        # Fallback thử lấy key 'image' nếu người dùng gửi đơn hoặc mảng
        files = request.files.getlist("image")
    
    if not files or len(files) == 0:
        return jsonify({"error": "Không có tệp ảnh nào được tải lên"}), 400

    gradcam_model_req = request.args.get("gradcam_model", "both")
    slices_data = []
    class_counts = {c: 0 for c in CLASS_NAMES}
    tumor_slices = []

    logger.info(f"Bắt đầu xử lý chuỗi {len(files)} lát cắt MRI...")

    for idx, file in enumerate(files):
        filename = (file.filename or f"slice_{idx+1}.jpg").lower()
        metadata = {}
        full_b64 = None

        try:
            if filename.endswith(".dcm"):
                try:
                    pil_img, metadata = MRIPreprocessor.read_dicom_file(file.stream)
                    buf = io.BytesIO()
                    pil_img.save(buf, format="JPEG", quality=85)
                    full_b64 = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
                except Exception:
                    file.stream.seek(0)
                    pil_img = Image.open(file.stream)
            else:
                pil_img = Image.open(file.stream)

            # Tạo ảnh thumbnail nhỏ cho thanh cuộn filmstrip (128x128)
            thumb = pil_img.copy()
            thumb.thumbnail((128, 128))
            t_buf = io.BytesIO()
            thumb.convert("RGB").save(t_buf, format="JPEG", quality=75)
            thumb_b64 = "data:image/jpeg;base64," + base64.b64encode(t_buf.getvalue()).decode()

            if not full_b64:
                f_buf = io.BytesIO()
                pil_img.convert("RGB").save(f_buf, format="JPEG", quality=85)
                full_b64 = "data:image/jpeg;base64," + base64.b64encode(f_buf.getvalue()).decode()

            # Chạy dự đoán cho lát cắt này
            result = engine.predict(pil_img, gradcam_request=gradcam_model_req)
            pred_class = result["class"]
            confidence = result["confidence"]
            has_tumor = pred_class != "notumor"

            class_counts[pred_class] += 1
            if has_tumor:
                tumor_slices.append((idx, pred_class, confidence))

            slices_data.append({
                "index": idx + 1,
                "filename": file.filename or f"Lát #{idx+1}",
                "class": pred_class,
                "class_vi": result["class_vi"],
                "confidence": confidence,
                "has_tumor": has_tumor,
                "probabilities": {
                    name: {"score_pct": score, "label_vi": CLASS_VI[name]}
                    for name, score in result["probabilities"].items()
                },
                "per_model": result.get("per_model", {}),
                "gradcam": result.get("gradcam", {}),
                "thumbnail": thumb_b64,
                "preview_image": full_b64,
                "metadata": metadata,
            })

        except Exception as slice_err:
            logger.warning(f"Lỗi khi xử lý lát cắt {filename}: {slice_err}")
            continue

    if not slices_data:
        return jsonify({"error": "Không thể xử lý bất kỳ ảnh nào trong chuỗi"}), 400

    # ─── Tổng hợp kết luận ca bệnh & Tìm Key-Slice ─────────────────────────────
    tumor_count = len(tumor_slices)
    total_valid_slices = len(slices_data)

    if tumor_count > 0:
        # Nhóm u xuất hiện nhiều nhất và có xác suất cao nhất
        from collections import Counter
        tumor_types = [t[1] for t in tumor_slices]
        most_common_tumor = Counter(tumor_types).most_common(1)[0][0]
        
        # Lấy Key-Slice là lát cắt có độ tin cậy cao nhất của loại u đó
        matching_slices = [s for s in slices_data if s["class"] == most_common_tumor]
        key_slice = max(matching_slices, key=lambda s: s["confidence"])
        overall_class = most_common_tumor
        overall_conf = key_slice["confidence"]
    else:
        overall_class = "notumor"
        key_slice = max(slices_data, key=lambda s: s["confidence"])
        overall_conf = key_slice["confidence"]

    response = {
        "mode": "batch",
        "total_slices": total_valid_slices,
        "patient_summary": {
            "class": overall_class,
            "class_vi": CLASS_VI[overall_class],
            "confidence": overall_conf,
            "has_tumor": overall_class != "notumor",
            "tumor_slice_count": tumor_count,
            "tumor_slice_percentage": round(100.0 * tumor_count / total_valid_slices, 1),
            "severity": _get_severity(overall_class, overall_conf / 100.0),
            "recommendation": _get_recommendation(overall_class),
            "class_counts": class_counts,
        },
        "key_slice_index": key_slice["index"],
        "slices": slices_data,
    }

    logger.info(f"Hoàn thành chuỗi: Ca bệnh = {overall_class} (Key slice #{key_slice['index']} - {overall_conf:.2f}%)")
    return jsonify(response)



def _get_severity(pred_class: str, confidence: float) -> dict:
    severity_map = {
        "notumor":    ("none",   "Bình thường"),
        "meningioma": ("medium", "Trung bình"),
        "pituitary":  ("medium", "Trung bình"),
        "glioma":     ("high",   "Cao"),
    }
    level, label = severity_map.get(pred_class, ("unknown", "Không xác định"))
    return {"level": level, "label": label, "confidence_pct": round(confidence * 100, 2)}

def _get_recommendation(pred_class: str) -> str:
    recs = {
        "notumor":    "Không phát hiện bất thường. Tiếp tục theo dõi định kỳ theo lịch hẹn.",
        "meningioma": "Phát hiện dấu hiệu u màng não. Cần chụp MRI có cản quang và tham vấn bác sĩ thần kinh.",
        "pituitary":  "Phát hiện dấu hiệu u tuyến yên. Cần xét nghiệm hormone và tham vấn chuyên khoa nội tiết.",
        "glioma":     "Phát hiện dấu hiệu u thần kinh đệm. Cần sinh thiết và tham vấn bác sĩ thần kinh-ung thư ngay.",
    }
    return recs.get(pred_class, "Vui lòng tham vấn bác sĩ chuyên khoa.")

# ─── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Brain Tumor MRI Classifier API (Refactored)")
    p.add_argument("--resnet",   default=None)
    p.add_argument("--effnet",   default=None)
    p.add_argument("--convnext", default=None)
    p.add_argument("--effnet_v2", default=None)
    p.add_argument("--swin_t",    default=None)
    p.add_argument("--swin_b",    default=None)
    p.add_argument("--port",    type=int, default=SERVER_CONFIG["port"])
    p.add_argument("--host",    default=SERVER_CONFIG["host"])
    p.add_argument("--debug",   action="store_true", default=SERVER_CONFIG["debug"])
    args = p.parse_args()

    load_models_to_engine(args)
    if not engine.models:
        logger.warning("⚠ No models loaded! API will return 500/error for /predict.")

    logger.info(f"Starting Flask on {args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug, use_reloader=False)
