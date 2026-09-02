# src/mri_preprocessing.py
"""
Tiền xử lý ảnh MRI grayscale theo chuẩn y tế.
Đảm bảo pipeline nhất quán giữa training và inference.
"""

import numpy as np
import cv2
from PIL import Image
from typing import Tuple


class MRIPreprocessor:
    """Xử lý ảnh MRI grayscale theo chuẩn y tế."""

    # Giá trị chuẩn hóa cho ảnh MRI (không phải ImageNet)
    MRI_MEAN = [0.5, 0.5, 0.5]
    MRI_STD  = [0.2, 0.2, 0.2]

    @staticmethod
    def convert_pil_to_grayscale(pil_image: Image.Image) -> np.ndarray:
        """
        Chuyển PIL image sang grayscale uint8 numpy array.
        Hỗ trợ mọi format: L, RGB, RGBA, P, ...
        
        Returns:
            np.ndarray shape (H, W), dtype uint8
        """
        if pil_image.mode == "L":
            # Đã là grayscale
            return np.array(pil_image, dtype=np.uint8)
        elif pil_image.mode == "RGBA":
            # Bỏ alpha channel rồi convert
            return np.array(pil_image.convert("L"), dtype=np.uint8)
        elif pil_image.mode in ("RGB", "P", "CMYK", "YCbCr", "LAB", "HSV"):
            # Convert sang grayscale theo công thức chuẩn (0.299R + 0.587G + 0.114B)
            return np.array(pil_image.convert("L"), dtype=np.uint8)
        else:
            # Các format khác: thử convert trực tiếp
            try:
                return np.array(pil_image.convert("L"), dtype=np.uint8)
            except Exception:
                # Fallback: convert sang RGB rồi sang grayscale
                rgb = np.array(pil_image.convert("RGB"), dtype=np.uint8)
                return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    @staticmethod
    def normalize_minmax(img: np.ndarray) -> np.ndarray:
        """
        Min-Max normalization về khoảng [0, 1].
        Dùng để chuẩn hóa khoảng giá trị pixel trước khi xử lý.
        
        Args:
            img: numpy array, bất kỳ dtype
        Returns:
            numpy array float32 trong khoảng [0, 1]
        """
        img = img.astype(np.float32)
        img_min, img_max = img.min(), img.max()
        if img_max == img_min:
            # Ảnh đen hoàn toàn hoặc trắng hoàn toàn
            return np.zeros_like(img, dtype=np.float32)
        return (img - img_min) / (img_max - img_min)

    @staticmethod
    def grayscale_to_rgb(img: np.ndarray) -> np.ndarray:
        """
        Chuyển grayscale (H, W) → RGB (H, W, 3) bằng cách stack channel 3 lần.
        Nếu ảnh đã là (H, W, 3) thì giữ nguyên.
        
        Args:
            img: numpy array shape (H, W) hoặc (H, W, 3)
        Returns:
            numpy array shape (H, W, 3), dtype giữ nguyên
        """
        if img.ndim == 2:
            # Grayscale → RGB bằng cách repeat channel
            return np.stack([img, img, img], axis=-1)
        elif img.ndim == 3 and img.shape[2] == 1:
            # (H, W, 1) → (H, W, 3)
            return np.concatenate([img, img, img], axis=-1)
        # Đã là (H, W, 3) hoặc (H, W, 4)
        return img

    @classmethod
    def from_pil_to_rgb_array(cls, pil_image: Image.Image) -> np.ndarray:
        """
        Pipeline hoàn chỉnh: PIL Image → RGB uint8 numpy array (H, W, 3).
        
        1. Convert sang grayscale (đảm bảo nhất quán với training data)
        2. Min-Max normalize về [0, 1] (chuẩn hóa khoảng giá trị)
        3. Scale lại về [0, 255] uint8
        4. Chuyển sang RGB (H, W, 3)
        
        Returns:
            np.ndarray shape (H, W, 3), dtype uint8
        """
        # Bước 1: Chuyển sang grayscale
        gray = cls.convert_pil_to_grayscale(pil_image)

        # Bước 2: Min-Max normalize về [0, 1]
        norm = cls.normalize_minmax(gray)

        # Bước 3: Scale lại về [0, 255] uint8
        uint8_img = (norm * 255).astype(np.uint8)

        # Bước 4: Stack thành RGB
        return cls.grayscale_to_rgb(uint8_img)

    @classmethod
    def read_dicom_file(cls, file_stream) -> Tuple[Image.Image, dict]:
        """
        Đọc file DICOM (.dcm), chuyển pixel array thành PIL Image 
        và trích xuất metadata y tế của bệnh nhân.
        
        Returns:
            Tuple[Image.Image, dict]: (PIL Image, patient_metadata)
        """
        import pydicom

        dcm = pydicom.dcmread(file_stream)
        pixel_array = dcm.pixel_array.astype(np.float32)

        # Áp dụng RescaleSlope và RescaleIntercept nếu có trong DICOM header
        if hasattr(dcm, "RescaleSlope") and hasattr(dcm, "RescaleIntercept"):
            pixel_array = pixel_array * float(dcm.RescaleSlope) + float(dcm.RescaleIntercept)

        # Min-Max Normalization về [0, 255] uint8
        p_min, p_max = pixel_array.min(), pixel_array.max()
        if p_max > p_min:
            uint8_img = ((pixel_array - p_min) / (p_max - p_min) * 255.0).astype(np.uint8)
        else:
            uint8_img = np.zeros_like(pixel_array, dtype=np.uint8)

        # Xử lý trường hợp ảnh đa kênh hoặc 2D grayscale
        if uint8_img.ndim == 2:
            pil_image = Image.fromarray(uint8_img, mode="L")
        else:
            pil_image = Image.fromarray(uint8_img)

        # Trích xuất metadata y tế
        metadata = {
            "patient_id": str(getattr(dcm, "PatientID", "")),
            "patient_name": str(getattr(dcm, "PatientName", "")).replace("^", " "),
            "patient_age": str(getattr(dcm, "PatientAge", "")),
            "patient_sex": str(getattr(dcm, "PatientSex", "")),
            "study_date": str(getattr(dcm, "StudyDate", "")),
            "modality": str(getattr(dcm, "Modality", "MR")),
            "institution": str(getattr(dcm, "InstitutionName", "")),
            "body_part": str(getattr(dcm, "BodyPartExamined", "BRAIN")),
        }

        return pil_image, metadata
