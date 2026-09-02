import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from src.models.base import BrainTumorModel

class GradCAM:
    def __init__(self, model: 'BrainTumorModel', target_layer: nn.Module): # Khởi tạo GradCAM
        self.model = model  
        self.target_layer = target_layer
        self._gradients: Optional[torch.Tensor] = None
        self._activations: Optional[torch.Tensor] = None
        self._register_hooks() # Hook vào layer cuối cùng của feature extractor

    def _register_hooks(self):  # Forward hook: Bắt activations (đặc trưng) khi ảnh đi qua
        def forward_hook(module, input, output): # Forward hook
            self._activations = output.detach()
        def backward_hook(module, grad_input, grad_output): # Backward hook: Bắt gradients (độ quan trọng) khi lan truyền ngược
            self._gradients = grad_output[0].detach()
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor: torch.Tensor, class_idx: Optional[int] = None, smooth: bool = True):
        self.model.eval()
        input_tensor = input_tensor.unsqueeze(0) if input_tensor.dim() == 3 else input_tensor
        output = self.model(input_tensor) # Forward pass để bắt activations
        probs = F.softmax(output, dim=1) # Tính xác suất
        confidences = probs[0].cpu().detach().numpy() # Lấy xác suất của từng lớp
        if class_idx is None:
            class_idx = output.argmax(dim=1).item() # Lấy lớp có xác suất cao nhất
        
        self.model.zero_grad() # Reset gradients
        score = output[0, class_idx] # Lấy điểm số của lớp được chọn
        score.backward(retain_graph=True)

        grads = self._gradients[0] # Lấy gradients
        acts = self._activations[0] # Lấy activations
        weights = grads.mean(dim=(1, 2)) # Tính trọng số
        cam = torch.zeros(acts.shape[1:], dtype=torch.float32, device=acts.device)
        for i, w in enumerate(weights): # Nhân trọng số với activations
            cam += w * acts[i]
        cam = F.relu(cam).cpu().detach().numpy() # Áp dụng ReLU và chuyển sang numpy
        if smooth:
            cam = cv2.GaussianBlur(cam, (7, 7), 0) # Làm mịn heatmap
        
        cam_min, cam_max = cam.min(), cam.max() # Chuẩn hóa heatmap
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
        else:
            cam = np.zeros_like(cam)
        return np.clip(cam, 0, 1), int(class_idx), confidences # Trả về heatmap, class_idx và confidences

class GradCAMPlusPlus(GradCAM):
    def generate(self, input_tensor: torch.Tensor, class_idx: Optional[int] = None, smooth: bool = True): # Grad-CAM++
        self.model.eval()
        input_tensor = input_tensor.unsqueeze(0) if input_tensor.dim() == 3 else input_tensor
        output = self.model(input_tensor)
        probs = F.softmax(output, dim=1)
        confidences = probs[0].cpu().detach().numpy()
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        self.model.zero_grad()
        score = output[0, class_idx]
        score.backward(retain_graph=True)

        grads = self._gradients[0]
        acts = self._activations[0]
        grads_power_2 = grads ** 2
        grads_power_3 = grads ** 3
        sum_acts = acts.sum(dim=(1, 2), keepdim=True)
        eps = 1e-8
        alpha_num = grads_power_2
        alpha_denom = 2 * grads_power_2 + sum_acts * grads_power_3 + eps # Alpha weights phức tạp hơn → chính xác hơn GradCAM thường
        alpha = alpha_num / alpha_denom
        relu_grad = F.relu(score.exp() * grads)
        weights = (alpha * relu_grad).sum(dim=(1, 2))

        cam = torch.zeros(acts.shape[1:], dtype=torch.float32, device=acts.device)
        for i, w in enumerate(weights):
            cam += w * acts[i]
        cam = F.relu(cam).cpu().detach().numpy()
        if smooth:
            cam = cv2.GaussianBlur(cam, (7, 7), 0)

        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
        return np.clip(cam, 0, 1), int(class_idx), confidences

def apply_gradcam_overlay(original_img, cam, colormap=cv2.COLORMAP_JET, alpha=0.45, img_size=224):
    img_resized = cv2.resize(original_img, (img_size, img_size))
    cam_resized = cv2.resize(cam, (img_size, img_size))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), colormap)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = np.float32(img_resized) * (1 - alpha) + np.float32(heatmap) * alpha
    return np.clip(overlay, 0, 255).astype(np.uint8)
