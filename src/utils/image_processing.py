import cv2
import numpy as np
from PIL import Image
from typing import Union


def crop_brain_contour(img: np.ndarray, min_area_ratio: float = 0.05) -> np.ndarray:
    """
    Cắt sát viền đường cong bộ não (Brain Contour Cropping) loại bỏ viền đen xung quanh.
    
    Args:
        img: RGB numpy array (H, W, 3)
        min_area_ratio: Tỷ lệ diện tích tối thiểu của contour bộ não so với toàn bộ ảnh.
        
    Returns:
        RGB numpy array đã được crop sát viền bộ não.
    """
    if img is None or img.size == 0:
        return img

    # Chuyển sang ảnh xám
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # Làm mờ khử nhiễu
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Ngưỡng phân đoạn (Otsu thresholding)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Thao tác hình thái học xóa nhiễu đốm nhỏ
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

    # Tìm đường viền (contours)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return img

    # Lấy contour có diện tích lớn nhất (bộ toàn bộ não)
    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)
    total_area = img.shape[0] * img.shape[1]

    # Kiểm tra xem contour có đủ lớn không
    if area / total_area < min_area_ratio:
        return img

    # Lấy Bounding Box xung quanh bộ não
    x, y, w, h = cv2.boundingRect(c)
    if w < 30 or h < 30:
        return img

    # Crop ảnh
    cropped = img[y:y + h, x:x + w]
    if cropped is None or cropped.size == 0:
        return img
    return cropped


def letterbox_pad(img: np.ndarray, color: tuple = (0, 0, 0)) -> np.ndarray:
    """
    Thêm viền đen cân đối xung quanh để đưa ảnh về tỷ lệ vuông (1:1) mà không bóp méo hình dạng bộ não.
    
    Args:
        img: RGB numpy array (H, W, 3)
        color: Màu viền (mặc định đen RGB: (0, 0, 0))
        
    Returns:
        RGB numpy array hình vuông.
    """
    if img is None or img.size == 0:
        return img

    h, w = img.shape[:2]
    if h <= 0 or w <= 0 or h == w:
        return img

    max_dim = max(h, w)
    top = (max_dim - h) // 2
    bottom = max_dim - h - top
    left = (max_dim - w) // 2
    right = max_dim - w - left

    padded = cv2.copyMakeBorder(
        img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )
    return padded


def apply_clahe_contrast(img: np.ndarray, clip_limit: float = 2.0, tile_grid_size: tuple = (8, 8)) -> np.ndarray:
    """Tăng cường tương phản cục bộ CLAHE trên kênh độ sáng L của ảnh LAB."""
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)


def preprocess_mri_image(img_input: Union[np.ndarray, Image.Image], enhance_contrast: bool = True) -> np.ndarray:
    """
    Pipeline tiền xử lý hoàn chỉnh cho ảnh MRI:
    1. Chuyển đổi về RGB numpy array.
    2. Auto Brain Contour Cropping (cắt sát viền hộp sọ).
    3. Letterbox Padding (pad thành hình vuông 1:1).
    4. CLAHE Contrast Enhancement (Cân bằng độ tương phản cục bộ).
    
    Returns:
        RGB numpy array (Square aspect ratio, cropped brain, enhanced contrast).
    """
    if isinstance(img_input, Image.Image):
        img_np = np.array(img_input.convert("RGB"))
    elif isinstance(img_input, np.ndarray):
        if len(img_input.shape) == 2: # Grayscale
            img_np = cv2.cvtColor(img_input, cv2.COLOR_GRAY2RGB)
        elif img_input.shape[2] == 4: # RGBA
            img_np = cv2.cvtColor(img_input, cv2.COLOR_RGBA2RGB)
        else:
            img_np = img_input.copy()
    else:
        raise ValueError("Invalid image input type. Expected PIL Image or numpy array.")

    # Step 1: Crop sát viền não
    cropped_img = crop_brain_contour(img_np)

    # Step 2: Pad thành hình vuông 1:1
    squared_img = letterbox_pad(cropped_img)

    # Step 3: Tăng cường tương phản CLAHE
    if enhance_contrast:
        squared_img = apply_clahe_contrast(squared_img)

    return squared_img
