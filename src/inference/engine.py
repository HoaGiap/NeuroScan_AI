import io
import base64
import torch
import numpy as np
import torch.nn.functional as F
from PIL import Image
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.models.base import BrainTumorModel
from src.models.registry import (
    build_resnet50, build_efficientnet, build_convnext_small, 
    build_efficientnet_v2_s, build_swin_t, build_swin_b, get_gradcam
)
from src.utils.gradcam import apply_gradcam_overlay
from src.config import CLASS_NAMES, CLASS_VI, IMG_SIZE, DEVICE, SERVER_CONFIG
from src.dataset import get_val_transforms
from src.mri_preprocessing import MRIPreprocessor
from src.utils.image_processing import crop_brain_contour, letterbox_pad

class InferenceEngine:
    def __init__(self, device: torch.device = DEVICE): # Khởi tạo InferenceEngine
        self.device = device # Chọn device (CPU hoặc GPU)
        self.models: Dict[str, BrainTumorModel] = {} # Lưu trữ các model
        self.gradcam_engines = {} # Lưu trữ các GradCAM engines
        self.transforms = get_val_transforms(IMG_SIZE)

    def load_model(self, name: str, path: str): # Load model từ checkpoint
        if not Path(path).exists(): # Kiểm tra xem checkpoint có tồn tại không
            print(f"[WARN] Checkpoint not found: {path}")
            return False
        
        builders = {
            "resnet50": build_resnet50,
            "efficientnet": build_efficientnet,
            "convnext_small": build_convnext_small,
            "efficientnet_v2_s": build_efficientnet_v2_s,
            "swin_t": build_swin_t,
            "swin_b": build_swin_b,
        }
        
        if name not in builders: # Kiểm tra xem model có tồn tại không
            return False

        try:
            model = builders[name](pretrained=False) # Khởi tạo model
            ckpt = torch.load(path, map_location=self.device) # Load checkpoint
            model.load_state_dict(ckpt["model_state_dict"]) # Load state dict
            model.to(self.device).eval() # Chuyển model sang device và eval mode
        except Exception as e:
            if "CUDA error: out of memory" in str(e) or "out of memory" in str(e).lower():
                print(f"[WARN] CUDA OOM while loading {name}. Falling back to CPU...")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                self.device = torch.device("cpu")
                model = builders[name](pretrained=False)
                ckpt = torch.load(path, map_location="cpu")
                model.load_state_dict(ckpt["model_state_dict"])
                model.to(self.device).eval()
            else:
                raise e
        
        self.models[name] = model # Lưu model
        self.gradcam_engines[name] = get_gradcam(model, variant=SERVER_CONFIG["gradcam_variant"]) # Khởi tạo GradCAM engine
        return True

    def preprocess(self, pil_image: Image.Image) -> Tuple[torch.Tensor, np.ndarray]: # Preprocess ảnh
        """
        Chuẩn hóa ảnh MRI từ bên ngoài nhất quán với training pipeline:
        1. Convert bất kỳ format (RGBA, L, RGB) sang grayscale
        2. Min-Max normalize → scale về [0,255] uint8
        3. Stack thành RGB (H,W,3)
        4. Crop viền não + Letterbox pad
        5. Albumentations transforms (Resize + MRI Normalize)
        """
        # Bước 1-3: PIL → grayscale → min-max → RGB uint8
        img_rgb = MRIPreprocessor.from_pil_to_rgb_array(pil_image)

        # Bước 4: Crop viền não + Letterbox pad (giống hệt dataset.__getitem__)
        img_rgb = crop_brain_contour(img_rgb)
        img_rgb = letterbox_pad(img_rgb)

        orig_np = img_rgb.copy()  # Lưu ảnh gốc cho Grad-CAM

        # Bước 5: Albumentations (Resize 224x224 + MRI Normalize)
        aug = self.transforms(image=img_rgb)
        tensor = aug["image"].unsqueeze(0).to(self.device)
        return tensor, orig_np

    def predict(self, pil_image: Image.Image, gradcam_request: str = "both") -> Dict: # Predict
        if not self.models: # Kiểm tra xem có model nào được load không
            raise RuntimeError("No models loaded.")

        tensor, orig_np = self.preprocess(pil_image) # Preprocess ảnh
        
        all_probs = {} # Lưu trữ xác suất của từng model
        for name, model in self.models.items(): # Lặp qua từng model để dự đoán
            logits = model(tensor) # Forward pass
            probs = F.softmax(logits, dim=1)[0].detach().cpu().numpy() # Tính xác suất
            all_probs[name] = probs # Lưu xác suất

        # Ensemble
        ensemble_probs = np.mean(list(all_probs.values()), axis=0) if len(all_probs) > 1 else list(all_probs.values())[0]
        
        pred_idx = int(np.argmax(ensemble_probs))
        pred_class = CLASS_NAMES[pred_idx]
        confidence = float(ensemble_probs[pred_idx])

        # Grad-CAM
        gradcam_images = {}
        target_models = list(self.models.keys()) if gradcam_request == "both" else [gradcam_request]
        
        for name in target_models:
            if name in self.gradcam_engines:
                cam, _, _ = self.gradcam_engines[name].generate(tensor.squeeze(0), class_idx=pred_idx)
                overlay = apply_gradcam_overlay(orig_np, cam, img_size=IMG_SIZE)
                
                # Base64
                img_pil = Image.fromarray(overlay)
                buf = io.BytesIO()
                img_pil.save(buf, format="PNG")
                gradcam_images[name] = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

        return {
            "class": pred_class,
            "class_vi": CLASS_VI.get(pred_class, pred_class),
            "confidence": round(confidence * 100, 2),
            "probabilities": {
                CLASS_NAMES[i]: round(float(ensemble_probs[i]) * 100, 2)
                for i in range(len(CLASS_NAMES))
            },
            "per_model": {
                name: {CLASS_NAMES[i]: round(float(probs[i]) * 100, 2) for i in range(len(CLASS_NAMES))}
                for name, probs in all_probs.items()
            },
            "gradcam": gradcam_images
        }
