
import os
import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
from collections import Counter


from src.config import IMG_SIZE, CLASS_NAMES, CLASS_VI, NUM_CLASSES
from src.mri_preprocessing import MRIPreprocessor
from src.utils.image_processing import crop_brain_contour, letterbox_pad

CLASS_LABELS = {name: idx for idx, name in enumerate(CLASS_NAMES)}


# ─── Quá trình tăng cường dữ liệu (Augmentation Pipelines) ────────────────────
def get_train_transforms(img_size: int = IMG_SIZE) -> A.Compose:
    return A.Compose([
        A.Resize(img_size, img_size),                   # Resize về 224×224
        A.HorizontalFlip(p=0.5),                        # Lật ảnh ngang
        A.VerticalFlip(p=0.2),                          # Lật ảnh dọc
        A.RandomRotate90(p=0.3),                        # Xoay 90 độ
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5), # Dịch chuyển, co giãn, xoay
        A.OneOf([
            A.GaussNoise(p=0.5),                        # Thêm nhiễu Gaussian
            A.GaussianBlur(blur_limit=(3, 5)),          # Làm mờ Gaussian
            A.MedianBlur(blur_limit=3),                 # Làm mờ Median
        ], p=0.3),
        A.OneOf([
            A.ElasticTransform(alpha=60, sigma=6, p=0.5),
            A.GridDistortion(num_steps=5, distort_limit=0.2, p=0.5), # Biến dạng lưới
        ], p=0.2),
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.4), # Tăng cường tương phản cục bộ
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.4), # Tăng cường độ sáng và tương phản
        A.Normalize(                                 # Chuẩn hóa cho ảnh MRI (không phải ImageNet)
            mean=[0.5, 0.5, 0.5],
            std=[0.2, 0.2, 0.2],
            max_pixel_value=255.0,
        ),
        ToTensorV2(), # Chuyển sang Tensor
    ])


def get_val_transforms(img_size: int = IMG_SIZE) -> A.Compose:
    return A.Compose([
        A.Resize(img_size, img_size),                   # Resize về 224×224
        A.Normalize(                                 # Chuẩn hóa cho ảnh MRI (không phải ImageNet)
            mean=[0.5, 0.5, 0.5],
            std=[0.2, 0.2, 0.2],
            max_pixel_value=255.0,
        ),
        ToTensorV2(), # Chuyển sang Tensor
    ])


# ─── Lớp Dataset ──────────────────────────────────────────────────────────────
class BrainTumorDataset(Dataset):
    """Dataset cho Brain Tumor MRI Classification."""

    VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"} # Các định dạng ảnh được hỗ trợ

    def __init__(self, root_dir: str, transform=None, split: str = "train"):
        """
        Args:
            root_dir: Thư mục gốc chứa các class folder (Training/ hoặc Testing/)
            transform: Albumentations transform pipeline
            split: 'train' hoặc 'val'/'test'
        """
        self.root_dir = root_dir
        self.transform = transform
        self.split = split

        self.samples: list[tuple[str, int]] = []
        self._load_samples()

        class_counts = Counter(label for _, label in self.samples)
        self.class_counts = [class_counts.get(i, 0) for i in range(NUM_CLASSES)]
        print(f"[Dataset/{split}] Loaded {len(self.samples)} samples | "
              + " | ".join(f"{CLASS_NAMES[i]}: {self.class_counts[i]}"
                           for i in range(NUM_CLASSES)))

    def _load_samples(self):
        for class_name in CLASS_NAMES:
            class_dir = os.path.join(self.root_dir, class_name)
            if not os.path.isdir(class_dir):
                print(f"  [WARNING] Directory not found: {class_dir}")
                continue
            label = CLASS_LABELS[class_name]
            for fname in os.listdir(class_dir):
                ext = os.path.splitext(fname)[1].lower()
                if ext in self.VALID_EXTS:
                    self.samples.append((os.path.join(class_dir, fname), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, label = self.samples[idx]

        # Đọc ảnh MRI dưới dạng grayscale (nhất quán với dữ liệu thực tế)
        img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img_gray is None:
            # Fallback: dùng PIL để đọc rồi chuyển sang grayscale
            img_gray = MRIPreprocessor.convert_pil_to_grayscale(Image.open(img_path))

        # Min-Max normalize → scale về [0, 255] uint8
        img_norm = MRIPreprocessor.normalize_minmax(img_gray)
        img_uint8 = (img_norm * 255).astype(np.uint8)

        # Chuyển grayscale → RGB (H, W, 3) để tương thích với augmentation pipeline
        img_rgb = MRIPreprocessor.grayscale_to_rgb(img_uint8)

        # Tiền xử lý: Crop viền não + Letterbox pad thành hình vuông
        img_rgb = crop_brain_contour(img_rgb)
        img_rgb = letterbox_pad(img_rgb)

        if self.transform:
            augmented = self.transform(image=img_rgb)
            img_rgb = augmented["image"]

        return img_rgb, label

    def get_weighted_sampler(self) -> WeightedRandomSampler:
        """Tạo sampler cân bằng class để xử lý imbalanced data."""
        total = len(self.samples)
        class_weights = [total / (NUM_CLASSES * count) if count > 0 else 0
                         for count in self.class_counts]
        sample_weights = [class_weights[label] for _, label in self.samples]
        return WeightedRandomSampler(
            weights=sample_weights,
            num_samples=total,
            replacement=True,
        )

def get_subset_weighted_sampler(subset, num_classes) -> WeightedRandomSampler:
    """Tạo sampler cân bằng class cho một Subset."""
    # Lấy lables thực tế của subset
    labels = [subset.dataset.samples[idx][1] for idx in subset.indices]
    
    # Tính toán class_counts cho subset này
    class_counts = Counter(labels)
    counts = [class_counts.get(i, 0) for i in range(num_classes)]
    
    # Tính toán weights
    total = len(labels)
    class_weights = [total / (num_classes * count) if count > 0 else 0 
                     for count in counts]
    
    # Gán weight cho từng sample trong subset
    sample_weights = [class_weights[label] for label in labels]
    
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=total,
        replacement=True,
    )


# ─── Hàm tạo DataLoader ───────────────────────────────────────────────────────
def create_dataloaders(
    data_root: str,
    batch_size: int = 32,
    num_workers: int = 4,
    img_size: int = IMG_SIZE,
    use_weighted_sampler: bool = True,
    val_split: float = 0.15,          # dùng nếu không có thư mục Testing/ riêng
) -> tuple[DataLoader, DataLoader, DataLoader | None]:
    """
    Trả về (train_loader, val_loader, test_loader).
    test_loader = None nếu không tìm thấy thư mục Testing/.
    """
    train_dir = os.path.join(data_root, "Training")
    val_dir   = os.path.join(data_root, "Validation")
    test_dir  = os.path.join(data_root, "Testing")

    # Nếu tồn tại thư mục Validation vật lý
    if os.path.isdir(val_dir):
        train_dataset = BrainTumorDataset(
            root_dir=train_dir,
            transform=get_train_transforms(img_size),
            split="train",
        )
        val_dataset = BrainTumorDataset(
            root_dir=val_dir,
            transform=get_val_transforms(img_size),
            split="val",
        )
        sampler = (train_dataset.get_weighted_sampler()
                   if use_weighted_sampler else None)

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=sampler if use_weighted_sampler else None,
            shuffle=(not use_weighted_sampler),
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )
    else:
        # Dự phòng: Tách val từ train (stratified) nếu chưa có thư mục Validation
        train_dataset = BrainTumorDataset(
            root_dir=train_dir,
            transform=get_train_transforms(img_size),
            split="train",
        )
        val_dataset = BrainTumorDataset(
            root_dir=train_dir,
            transform=get_val_transforms(img_size),
            split="val",
        )

        from sklearn.model_selection import train_test_split
        indices = list(range(len(train_dataset)))
        labels  = [train_dataset.samples[i][1] for i in indices]
        train_idx, val_idx = train_test_split(
            indices, test_size=val_split, stratify=labels, random_state=42
        )
        from torch.utils.data import Subset
        train_subset = Subset(train_dataset, train_idx)
        val_subset   = Subset(val_dataset,   val_idx)

        sampler = (get_subset_weighted_sampler(train_subset, NUM_CLASSES)
                   if use_weighted_sampler else None)

        train_loader = DataLoader(
            train_subset,
            batch_size=batch_size,
            sampler=sampler if use_weighted_sampler else None,
            shuffle=(not use_weighted_sampler),
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=True,
        )
        val_loader = DataLoader(
            val_subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    test_loader = None
    if os.path.isdir(test_dir):
        test_dataset = BrainTumorDataset(
            root_dir=test_dir,
            transform=get_val_transforms(img_size),
            split="test",
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    return train_loader, val_loader, test_loader
