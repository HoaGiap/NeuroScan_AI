import unittest
import torch
import numpy as np
from PIL import Image
import os
import sys

# Thêm thư mục gốc vào path để import
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import IMG_SIZE, NUM_CLASSES
from src.dataset import BrainTumorDataset, get_train_transforms, get_val_transforms
from src.models.registry import build_resnet50, build_efficientnet, build_efficientnet_v2_s

class TestBrainTumorProject(unittest.TestCase):
    
    def setUp(self):
        # Tạo ảnh giả lập để test (3 channel RGB)
        self.fake_img = Image.fromarray(np.uint8(np.random.rand(300, 300, 3) * 255))
        self.fake_img_path = "tmp_test_img.jpg"
        self.fake_img.save(self.fake_img_path)

    def tearDown(self):
        if os.path.exists(self.fake_img_path):
            os.remove(self.fake_img_path)

    def test_config_values(self):
        """Kiểm tra các giá trị cấu hình cơ bản."""
        self.assertEqual(IMG_SIZE, 224)
        self.assertEqual(NUM_CLASSES, 4)

    def test_transforms(self):
        """Kiểm tra pipeline tiền xử lý ảnh."""
        transform = get_val_transforms(IMG_SIZE)
        img_np = np.array(self.fake_img)
        augmented = transform(image=img_np)
        tensor = augmented["image"]
        
        # Kiểm tra shape (C, H, W)
        self.assertEqual(tensor.shape, (3, IMG_SIZE, IMG_SIZE))
        # Kiểm tra kiểu dữ liệu
        self.assertEqual(tensor.dtype, torch.float32)

    @unittest.skip("Đã vô hiệu hóa mô hình ResNet50")
    def test_resnet50_architecture(self):
        """Kiểm tra cấu trúc mô hình ResNet50."""
        model = build_resnet50(pretrained=False)
        model.eval()
        
        dummy_input = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
        with torch.no_grad():
            output = model(dummy_input)
            
        # Kiểm tra output shape (batch_size, num_classes)
        self.assertEqual(output.shape, (1, NUM_CLASSES))


    @unittest.skip("Đã vô hiệu hóa mô hình EfficientNet")
    def test_efficientnet_architecture(self):
        """Kiểm tra cấu trúc mô hình EfficientNet-B0."""
        model = build_efficientnet(pretrained=False)
        model.eval()
        
        dummy_input = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
        with torch.no_grad():
            output = model(dummy_input)
            
        # Kiểm tra output shape (batch_size, num_classes)
        self.assertEqual(output.shape, (1, NUM_CLASSES))

    def test_efficientnet_v2_s_architecture(self):
        """Kiểm tra cấu trúc mô hình EfficientNet-V2-S."""
        model = build_efficientnet_v2_s(pretrained=False)
        model.eval()
        
        dummy_input = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
        with torch.no_grad():
            output = model(dummy_input)
            
        # Kiểm tra output shape (batch_size, num_classes)
        self.assertEqual(output.shape, (1, NUM_CLASSES))

    def test_mri_preprocessor_pipeline(self):
        """Kiểm tra pipeline tiền xử lý ảnh grayscale và RGB chuẩn y tế."""
        from src.mri_preprocessing import MRIPreprocessor
        rgb_arr = MRIPreprocessor.from_pil_to_rgb_array(self.fake_img)
        self.assertEqual(rgb_arr.shape, (300, 300, 3))
        self.assertEqual(rgb_arr.dtype, np.uint8)

if __name__ == '__main__':
    unittest.main()

