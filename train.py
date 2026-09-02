"""
train.py — Huấn luyện ResNet50 & EfficientNet-B0 cho Brain Tumor Classification

Cách dùng:
  # Huấn luyện ResNet50
  python train.py --backbone resnet50 --data ./data --epochs 50 --batch 32

  # Huấn luyện EfficientNet-B0
  python train.py --backbone efficientnet --data ./data --epochs 50 --batch 32

  # Huấn luyện cả hai
  python train.py --backbone both --data ./data --epochs 50 --batch 32

  python train.py --backbone convnext_small --data ./data 
python train.py --backbone efficientnet_v2_s --data ./data --epochs 50 --batch 32
"""

import os
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")
import argparse
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

# Thêm src/ vào path
sys.path.insert(0, os.path.dirname(__file__))
from src.dataset import create_dataloaders, CLASS_NAMES, NUM_CLASSES
from src.models.base import BrainTumorModel
from src.models.registry import build_resnet50, build_efficientnet


# ─── Config ───────────────────────────────────────────────────────────────────
DEFAULT_CONFIG = {
    "img_size":      224,   # Kích thước ảnh đầu vào
    "batch_size":    32,    # Số lượng ảnh mỗi batch
    "epochs":        50,    # Số epoch huấn luyện
    "lr":            1e-4,  # Learning rate
    "weight_decay":  1e-4,  # Weight decay
    "dropout":       0.4,   # Dropout rate
    "patience":      10,    # Early stopping patience
    "min_delta":     0.001, # Min improvement
    "label_smooth":  0.1,   # Label smoothing epsilon
    "num_workers":   4,     # Số lượng worker
    "scheduler":     "cosine", # 'cosine' | 'onecycle'
    "mode":          "finetune", # 'finetune' | 'feature' | 'partial'
    "mixed_prec":    True,     # AMP mixed precision
    "grad_clip":     1.0,      # Gradient clipping
    "warmup_epochs": 5,      # Warmup cho feature mode trước finetune
}


# ─── Loss with Label Smoothing ────────────────────────────────────────────────
class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing: float = 0.1, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.smoothing = smoothing
        self.num_classes = num_classes

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        confidence = 1.0 - self.smoothing
        smooth_val = self.smoothing / (self.num_classes - 1)

        log_prob = torch.log_softmax(pred, dim=-1)
        nll_loss = -log_prob.gather(dim=-1, index=target.unsqueeze(1)).squeeze(1)    # Loss thông thường
        smooth_loss = -log_prob.mean(dim=-1)   # Loss làm mịn phân bổ đều cho các lớp
        loss = confidence * nll_loss + smooth_val * smooth_loss * self.num_classes
        return loss.mean()


# ─── Early Stopping ───────────────────────────────────────────────────────────
class EarlyStopping:
    def __init__(self, patience: int = 10, min_delta: float = 0.001, mode: str = "max"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best = -np.inf if mode == "max" else np.inf
        self.counter = 0
        self.stop = False

    def __call__(self, metric: float) -> bool:
        improved = (self.mode == "max" and metric > self.best + self.min_delta) or \
                   (self.mode == "min" and metric < self.best - self.min_delta)
        if improved:
            self.best = metric
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True
        return self.stop


# ─── Metrics ──────────────────────────────────────────────────────────────────
class MetricTracker:
    def __init__(self):
        self.reset()

    def reset(self):
        self.running_loss = 0.0
        self.correct = 0
        self.total = 0
        self.all_preds = []
        self.all_labels = []

    def update(self, loss: float, preds: torch.Tensor, labels: torch.Tensor):
        self.running_loss += loss
        self.correct += (preds == labels).sum().item()
        self.total += labels.size(0)
        self.all_preds.extend(preds.cpu().numpy())
        self.all_labels.extend(labels.cpu().numpy())

    @property
    def avg_loss(self) -> float:
        return self.running_loss / max(self.total, 1)

    @property
    def accuracy(self) -> float:
        return 100.0 * self.correct / max(self.total, 1)


# ─── One Epoch ────────────────────────────────────────────────────────────────
def run_epoch(
    model: BrainTumorModel,
    loader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    scaler,
    scheduler=None,
    grad_clip: float = 1.0,
    is_train: bool = True,
    epoch: int = 0,
) -> tuple[float, float, list, list]:

    model.train() if is_train else model.eval() # Chế độ train hoặc eval

    tracker = MetricTracker() # Khởi tạo MetricTracker

    pbar = tqdm(loader, desc=f"{'Train' if is_train else 'Val ':5s}",
                leave=False, dynamic_ncols=True) # Thanh tiến trình

    ctx = torch.enable_grad() if is_train else torch.no_grad() # Context manager
    with ctx:
        for imgs, labels in pbar: # Lặp qua từng batch
            imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True) # Chuyển data lên GPU

            if is_train:
                optimizer.zero_grad(set_to_none=True) # Reset gradients
                with torch.autocast(device_type="cuda" if device.type == "cuda" else "cpu", enabled=(scaler is not None)): # Tính toán bằng float16
                    logits = model(imgs) # Forward pass
                    loss = criterion(logits, labels) # Tính loss

                if scaler:
                    scaler.scale(loss).backward() # Backward pass (float16)
                    scaler.unscale_(optimizer) # Chuyển lại float32
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip) # Clip gradients
                    scaler.step(optimizer) # Cập nhật weights
                    scaler.update() # Cập nhật scaler
                else:
                    loss.backward() # Backward pass
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip) # Clip gradients
                    optimizer.step() # Cập nhật weights

                if scheduler and isinstance(scheduler, OneCycleLR):
                    scheduler.step() # Update scheduler
            else:
                with torch.autocast(device_type="cuda" if device.type == "cuda" else "cpu", enabled=(scaler is not None)): # Tính toán bằng float16
                    logits = model(imgs) # Forward pass
                    loss = criterion(logits, labels) # Tính loss

            preds = logits.argmax(dim=1) # Lấy dự đoán Class có xác suất cao nhất
            tracker.update(loss.item() * imgs.size(0), preds, labels)
            pbar.set_postfix(loss=f"{tracker.avg_loss:.4f}", acc=f"{tracker.accuracy:.1f}%") # Hiển thị loss và accuracy

    return tracker.avg_loss, tracker.accuracy, tracker.all_preds, tracker.all_labels


# ─── Save Checkpoint ──────────────────────────────────────────────────────────
def save_checkpoint(model, optimizer, epoch, val_acc, path):
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_acc": val_acc,
        "backbone": model.backbone_name,
    }, path)


# ─── Plot Results ─────────────────────────────────────────────────────────────
def plot_training_curves(history: dict, save_dir: str, backbone: str):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Training History — {backbone.upper()}", fontsize=14, fontweight="bold")

    epochs = range(1, len(history["train_loss"]) + 1)

    axes[0].plot(epochs, history["train_loss"], "b-o", label="Train Loss", markersize=3)
    axes[0].plot(epochs, history["val_loss"],   "r-o", label="Val Loss",   markersize=3)
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss"); axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, history["train_acc"], "b-o", label="Train Acc", markersize=3)
    axes[1].plot(epochs, history["val_acc"],   "r-o", label="Val Acc",   markersize=3)
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_title("Accuracy"); axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{backbone}_training_curves.png"), dpi=150, bbox_inches="tight")
    plt.close()


def plot_confusion_matrix(y_true, y_pred, save_dir: str, backbone: str):
    cm = confusion_matrix(y_true, y_pred)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm_pct, annot=True, fmt=".1f", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                ax=ax, linewidths=0.5)
    ax.set_title(f"Confusion Matrix (%) — {backbone.upper()}", fontweight="bold")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{backbone}_confusion_matrix.png"), dpi=150, bbox_inches="tight")
    plt.close()


# ─── Main Training Loop ───────────────────────────────────────────────────────
def train_model(backbone: str, args, cfg: dict) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  Training {backbone.upper()}  |  Device: {device}")
    print(f"{'='*60}")

    run_dir = Path(args.output) / backbone / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    # DataLoaders
    train_loader, val_loader, test_loader = create_dataloaders(
        data_root=args.data,
        batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"],
        img_size=cfg["img_size"],
    )

    # Model
    model = BrainTumorModel(
        backbone=backbone,
        num_classes=NUM_CLASSES,
        dropout=cfg["dropout"],
        pretrained=True,
        mode=cfg["mode"],
    ).to(device)

    total = model.get_total_params()
    trainable = model.get_trainable_params()
    print(f"  Params: {total:,} total | {trainable:,} trainable ({100*trainable/total:.1f}%)")

    # Loss & Optimizer
    criterion = LabelSmoothingCrossEntropy(smoothing=cfg["label_smooth"])
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
    )

    # Scheduler
    if cfg["scheduler"] == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=cfg["epochs"], eta_min=1e-6)
    else:
        scheduler = OneCycleLR(
            optimizer, max_lr=cfg["lr"] * 10,
            steps_per_epoch=len(train_loader),
            epochs=cfg["epochs"],
        )

    # AMP scaler
    scaler = (torch.cuda.amp.GradScaler()
              if cfg["mixed_prec"] and device.type == "cuda" else None)

    early_stop = EarlyStopping(patience=cfg["patience"], min_delta=cfg["min_delta"])
    writer = SummaryWriter(log_dir=str(run_dir / "logs"))

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_acc = 0.0
    best_ckpt = run_dir / f"{backbone}_best.pth"

    # Warmup phase: train only classifier first
    if cfg.get("warmup_epochs", 0) > 0 and cfg["mode"] == "feature":
        print(f"\n  Warmup phase ({cfg['warmup_epochs']} epochs)...")

    for epoch in range(1, cfg["epochs"] + 1):
        # Progressive unfreeze after warmup
        if epoch == cfg.get("warmup_epochs", 0) + 1 and cfg["mode"] == "feature":
            print("\n  [Unfreeze all layers for fine-tuning...]")
            model.unfreeze_all()
            optimizer.add_param_group({
                "params": [p for p in model.feature_extractor.parameters()
                           if not p.requires_grad],
                "lr": cfg["lr"] * 0.1,
            })

        t0 = time.time()
        train_loss, train_acc, _, _ = run_epoch(
            model, train_loader, criterion, optimizer, device,
            scaler, scheduler if isinstance(scheduler, OneCycleLR) else None,
            cfg["grad_clip"], is_train=True, epoch=epoch,
        )
        val_loss, val_acc, val_preds, val_labels = run_epoch(
            model, val_loader, criterion, optimizer, device,
            scaler, is_train=False, epoch=epoch,
        )

        if isinstance(scheduler, CosineAnnealingLR):
            scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        writer.add_scalars("Loss",     {"train": train_loss, "val": val_loss}, epoch)
        writer.add_scalars("Accuracy", {"train": train_acc,  "val": val_acc},  epoch)
        writer.add_scalar("LR", optimizer.param_groups[0]["lr"], epoch)

        elapsed = time.time() - t0
        print(f"  Epoch {epoch:3d}/{cfg['epochs']} | "
              f"Train {train_loss:.4f}/{train_acc:.1f}% | "
              f"Val {val_loss:.4f}/{val_acc:.1f}% | "
              f"LR {optimizer.param_groups[0]['lr']:.2e} | "
              f"{elapsed:.0f}s")

        if val_acc > best_acc:
            best_acc = val_acc
            save_checkpoint(model, optimizer, epoch, val_acc, best_ckpt)
            print(f"  [NEW BEST] {best_acc:.2f}% -> saved to {best_ckpt}")

        if early_stop(val_acc):
            print(f"\n  Early stopping triggered at epoch {epoch}!")
            break

    # Final evaluation on test set
    if test_loader:
        print("\n  [Evaluating on Test Set...]")
        model.load_state_dict(torch.load(best_ckpt)["model_state_dict"])
        _, test_acc, test_preds, test_labels = run_epoch(
            model, test_loader, criterion, optimizer, device,
            scaler, is_train=False, epoch=0,
        )
        print(f"\n  Test Accuracy: {test_acc:.2f}%")
        print("\n" + classification_report(test_labels, test_preds, target_names=CLASS_NAMES))
        plot_confusion_matrix(test_labels, test_preds, str(run_dir), backbone)

    # Plots
    plot_training_curves(history, str(run_dir), backbone)

    # Save config + history
    with open(run_dir / "config.json", "w") as f:
        json.dump({**cfg, "backbone": backbone, "best_val_acc": best_acc}, f, indent=2)
    with open(run_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    writer.close()
    print(f"\n  Done! Best Val Acc: {best_acc:.2f}% | Checkpoint: {best_ckpt}\n")
    return {"backbone": backbone, "best_acc": best_acc, "checkpoint": str(best_ckpt)}


# ─── CLI ──────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Brain Tumor MRI Classifier Trainer")
    p.add_argument("--backbone", default="both",
                   choices=["resnet50", "efficientnet", "convnext_small", "efficientnet_v2_s", "swin_t", "swin_b", "both"],
                   help="Model backbone to train")
    p.add_argument("--data",    default="./data",   help="Path to dataset root")
    p.add_argument("--output",  default="./checkpoints", help="Output directory")
    p.add_argument("--epochs",  type=int, default=50)
    p.add_argument("--batch",   type=int, default=32)
    p.add_argument("--lr",      type=float, default=1e-4)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--mode",    default="finetune",
                   choices=["finetune", "feature", "partial"])
    p.add_argument("--scheduler", default="cosine", choices=["cosine", "onecycle"])
    p.add_argument("--no-amp",  action="store_true", help="Disable mixed precision")
    p.add_argument("--workers", type=int, default=0, help="Num workers for dataloader (0 for Windows)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = {**DEFAULT_CONFIG}
    cfg["epochs"]      = args.epochs
    cfg["batch_size"]  = args.batch
    cfg["lr"]          = args.lr
    cfg["img_size"]    = args.img_size
    cfg["mode"]        = args.mode
    cfg["scheduler"]   = args.scheduler
    cfg["mixed_prec"]  = not args.no_amp
    cfg["num_workers"] = args.workers

    results = []
    backbones = (["resnet50", "efficientnet"]
                 if args.backbone == "both" else [args.backbone])

    for bb in backbones:
        r = train_model(bb, args, cfg)
        results.append(r)

    print("\n" + "="*60)
    print("  TRAINING SUMMARY")
    print("="*60)
    for r in results:
        print(f"  {r['backbone']:15s} | Best Val Acc: {r['best_acc']:.2f}%")
        print(f"  {'':15s}   Checkpoint: {r['checkpoint']}")
    print("="*60)
