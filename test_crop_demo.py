"""
test_crop_demo.py — Script kiểm tra tính năng Auto Brain Contour Crop & Letterbox Padding
"""
import os
import glob
import cv2
from PIL import Image
import numpy as np
from src.utils.image_processing import preprocess_mri_image, crop_brain_contour, letterbox_pad

def test_on_samples():
    data_dir = "data/Testing"
    if not os.path.exists(data_dir):
        print(f"Directory {data_dir} not found.")
        return

    image_paths = glob.glob(os.path.join(data_dir, "**/*.jpg"), recursive=True)[:5]
    if not image_paths:
        print("No sample images found in data/Testing.")
        return

    print(f"Found {len(image_paths)} sample images to test Brain Contour Cropping:")
    for i, path in enumerate(image_paths):
        img_bgr = cv2.imread(path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        orig_shape = img_rgb.shape
        processed = preprocess_mri_image(img_rgb)
        proc_shape = processed.shape
        
        print(f"  [{i+1}] {os.path.basename(path)}:")
        print(f"      Original Shape : {orig_shape}")
        print(f"      Processed Shape: {proc_shape}")

    print("\n[SUCCESS] Brain Contour Crop test executed successfully!")

if __name__ == "__main__":
    test_on_samples()
