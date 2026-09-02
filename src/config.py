"""
config.py — Centralized Configuration for NeuroScan AI
"""

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
from pathlib import Path
import torch

# ─── Project Structure ────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
LOG_DIR = BASE_DIR / "logs"

# ─── Dataset Config ──────────────────────────────────────────────────────────
IMG_SIZE = 224
NUM_CLASSES = 4
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
CLASS_VI = {
    "glioma":     "U thần kinh đệm",
    "meningioma": "U màng não",
    "notumor":    "Không có u",
    "pituitary":  "U tuyến yên",
}

# ─── Training Hyperparameters ────────────────────────────────────────────────
DEFAULT_TRAIN_CONFIG = {
    "batch_size": 32,   
    "lr": 1e-4,
    "epochs": 50,
    "dropout": 0.4,
    "weight_decay": 1e-4,
    "scheduler": "cosine",  # 'cosine' | 'onecycle'
    "val_split": 0.15,
    "early_stopping_patience": 10,
    "label_smoothing": 0.1,
}

# ─── Device ──────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── Flask Server ────────────────────────────────────────────────────────────
SERVER_CONFIG = {
    "host": "0.0.0.0",
    "port": 5000,
    "debug": True,
    "gradcam_variant": "gradcam++",
}

def ensure_dirs():
    """Tạo các thư mục cần thiết nếu chưa tồn tại."""
    for d in [CHECKPOINT_DIR, LOG_DIR]:
        d.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    ensure_dirs()
    print(f"Project Base: {BASE_DIR}")
    print(f"Device: {DEVICE}")
