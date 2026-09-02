"""
eval_testset.py — Script đánh giá độ chính xác mô hình trên tập kiểm thử độc lập (data/Testing)
"""
import os
import sys
import glob
import argparse
import torch

sys.stdout.reconfigure(encoding="utf-8")
from tqdm import tqdm
from src.dataset import BrainTumorDataset, CLASS_NAMES
from src.inference.engine import InferenceEngine

def get_latest_checkpoint(backbone: str) -> str | None:
    pths = glob.glob(f"checkpoints/{backbone}/**/*.pth", recursive=True)
    if not pths:
        return None
    return max(pths, key=os.path.getmtime)

def evaluate(backbone: str = "auto", custom_ckpt: str = None):
    test_dir = "data/Testing"
    if not os.path.exists(test_dir):
        print(f"Directory {test_dir} not found.")
        return

    # Xác định backbone & checkpoint
    if custom_ckpt:
        ckpt_path = custom_ckpt
        bb_name = backbone if backbone != "auto" else "resnet50"
    elif backbone != "auto":
        bb_name = backbone
        ckpt_path = get_latest_checkpoint(bb_name)
    else:
        # Auto detect: Ưu tiên resnet50 hoặc efficientnet_v2_s mới nhất
        all_pths = glob.glob("checkpoints/**/*.pth", recursive=True)
        if not all_pths:
            print("Không tìm thấy checkpoint nào trong thư mục checkpoints/.")
            return
        ckpt_path = max(all_pths, key=os.path.getmtime)
        # Suy ra backbone từ đường dẫn
        norm_path = ckpt_path.replace("\\", "/")
        if "resnet50" in norm_path:
            bb_name = "resnet50"
        elif "efficientnet_v2_s" in norm_path:
            bb_name = "efficientnet_v2_s"
        elif "convnext_small" in norm_path:
            bb_name = "convnext_small"
        elif "swin" in norm_path:
            bb_name = "swin_t" if "swin_t" in norm_path else "swin_b"
        else:
            bb_name = "resnet50"

    if not ckpt_path or not os.path.exists(ckpt_path):
        print(f"Không tìm thấy file checkpoint cho {bb_name}")
        return

    print(f"\n{'='*60}")
    print(f"  ĐÁNH GIÁ MÔ HÌNH: {bb_name.upper()}")
    print(f"  Checkpoint: {ckpt_path}")
    print(f"{'='*60}")

    engine = InferenceEngine()
    if not engine.load_model(bb_name, ckpt_path):
        print("Tải mô hình thất bại.")
        return

    dataset = BrainTumorDataset(root_dir=test_dir, transform=None, split="test")
    total = len(dataset)

    print(f"\nĐang đánh giá {total} ảnh trên tập Test độc lập ({test_dir})...")
    correct = 0
    class_correct = {c: 0 for c in CLASS_NAMES}
    class_total = {c: 0 for c in CLASS_NAMES}

    for idx in tqdm(range(total), desc="Evaluating"):
        img_path, label = dataset.samples[idx]
        from PIL import Image
        pil_img = Image.open(img_path)
        
        result = engine.predict(pil_img, gradcam_request="none")
        pred_class = result["class"]
        gt_class = CLASS_NAMES[label]

        class_total[gt_class] += 1
        if pred_class == gt_class:
            correct += 1
            class_correct[gt_class] += 1

    acc = 100.0 * correct / total
    print("\n" + "=" * 60)
    print(f"  KẾT QUẢ ĐỘ CHÍNH XÁC (ACCURACY) TẬP TEST: {acc:.2f}% ({correct}/{total})")
    print("=" * 60)
    for c in CLASS_NAMES:
        tot = class_total[c]
        cor = class_correct[c]
        c_acc = 100.0 * cor / tot if tot > 0 else 0
        print(f"  {c:15s}: {c_acc:6.2f}%  ({cor:3d}/{tot:3d} ảnh)")
    print("=" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Đánh giá mô hình trên tập Testing")
    parser.add_argument("--backbone", default="auto", choices=["auto", "resnet50", "efficientnet_v2_s", "convnext_small", "swin_t", "swin_b"],
                        help="Tên mô hình cần test")
    parser.add_argument("--ckpt", default=None, help="Đường dẫn trực tiếp tới file .pth (tùy chọn)")
    args = parser.parse_args()
    evaluate(args.backbone, args.ckpt)

