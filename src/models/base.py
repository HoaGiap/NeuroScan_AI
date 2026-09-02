import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models
from src.config import NUM_CLASSES

try:
    from efficientnet_pytorch import EfficientNet
    EFFICIENTNET_AVAILABLE = True
except ImportError:
    EFFICIENTNET_AVAILABLE = False

class BrainTumorModel(nn.Module):
    """
    Wrapper chung cho ResNet50, EfficientNet, ConvNeXt
    """
    def __init__(
        self,
        backbone: str = "resnet50",
        num_classes: int = NUM_CLASSES,
        dropout: float = 0.4,
        pretrained: bool = True,
        mode: str = "finetune",
        unfreeze_layers: int = 2,
    ):
        super().__init__()
        self.backbone_name = backbone
        self.num_classes = num_classes
        self.mode = mode

        if backbone == "resnet50":
            self._build_resnet50(pretrained, dropout, num_classes)
        elif backbone == "efficientnet":
            self._build_efficientnet(pretrained, dropout, num_classes)
        elif backbone == "convnext_small":
            self._build_convnext_small(pretrained, dropout, num_classes)
        elif backbone == "efficientnet_v2_s":
            self._build_efficientnet_v2_s(pretrained, dropout, num_classes)
        elif backbone == "swin_t":
            self._build_swin_t(pretrained, dropout, num_classes)
        elif backbone == "swin_b":
            self._build_swin_b(pretrained, dropout, num_classes)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        self._apply_freeze_mode(mode, unfreeze_layers)

    def get_total_params(self):
        return sum(p.numel() for p in self.parameters())

    def get_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def _build_resnet50(self, pretrained, dropout, num_classes):
        weights = tv_models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        base = tv_models.resnet50(weights=weights)
        self.feature_extractor = nn.Sequential(
            base.conv1, base.bn1, base.relu, base.maxpool,
            base.layer1, base.layer2, base.layer3, base.layer4,
        )
        self.avgpool = base.avgpool
        in_features = base.fc.in_features
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(512, num_classes),
        )
        self.gradcam_layer = self.feature_extractor[-1]

    def _build_efficientnet(self, pretrained, dropout, num_classes):
        if EFFICIENTNET_AVAILABLE:
            base = EfficientNet.from_pretrained("efficientnet-b0") if pretrained else EfficientNet.from_name("efficientnet-b0")
            self.feature_extractor = base
            in_features = base._fc.in_features
            base._fc = nn.Identity()
            self.avgpool = nn.AdaptiveAvgPool2d(1)
            self.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(in_features, 256),
                nn.BatchNorm1d(256),
                nn.SiLU(),
                nn.Dropout(dropout * 0.5),
                nn.Linear(256, num_classes),
            )
            self.gradcam_layer = base._blocks[-1]
        else:
            weights = tv_models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
            base = tv_models.efficientnet_b0(weights=weights)
            self.feature_extractor = base.features
            self.avgpool = base.avgpool
            in_features = base.classifier[-1].in_features
            self.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(in_features, 256),
                nn.BatchNorm1d(256),
                nn.SiLU(),
                nn.Dropout(dropout * 0.5),
                nn.Linear(256, num_classes),
            )
            self.gradcam_layer = self.feature_extractor[-1]

    def _build_convnext_small(self, pretrained, dropout, num_classes):
        weights = tv_models.ConvNeXt_Small_Weights.IMAGENET1K_V1 if pretrained else None
        base = tv_models.convnext_small(weights=weights)
        self.feature_extractor = base.features
        self.avgpool = base.avgpool
        in_features = base.classifier[-1].in_features
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(512, num_classes),
        )
        self.gradcam_layer = self.feature_extractor[-1]

    def _build_efficientnet_v2_s(self, pretrained, dropout, num_classes):
        weights = tv_models.EfficientNet_V2_S_Weights.IMAGENET1K_V1 if pretrained else None # Khởi tạo EfficientNet-V2-S #Transfer Learning
        base = tv_models.efficientnet_v2_s(weights=weights)         # Khởi tạo EfficientNet-V2-S
        self.feature_extractor = base.features          # Lấy feature extractor # Phần trích xuất đặc trưng
        self.avgpool = base.avgpool                   # Lấy avgpool
        in_features = base.classifier[-1].in_features # Lấy số features đầu vào
        self.classifier = nn.Sequential(                # Lớp phân loại
            nn.Dropout(dropout),                        #  Dropout 40% → chống overfitting
            nn.Linear(in_features, 256),                # Fully Connected: 1280 → 256
            nn.BatchNorm1d(256),                        # Batch Normalization
            nn.SiLU(),                                  # Hàm kích hoạt SiLU
            nn.Dropout(dropout * 0.5),                  # Dropout
            nn.Linear(256, num_classes),                # Lớp fully connected cuối cùng
        )
        self.gradcam_layer = self.feature_extractor[-1] # Lấy layer cuối cùng để tính Grad-CAM

    def _build_swin_t(self, pretrained, dropout, num_classes):
        weights = tv_models.Swin_T_Weights.IMAGENET1K_V1 if pretrained else None
        base = tv_models.swin_t(weights=weights)
        self.feature_extractor = base.features
        self.avgpool = base.avgpool
        in_features = base.head.in_features
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(512, num_classes),
        )
        # Swin-T gradcam: target the last normative layer
        self.gradcam_layer = self.feature_extractor[-1][-1]

    def _build_swin_b(self, pretrained, dropout, num_classes):
        weights = tv_models.Swin_B_Weights.IMAGENET1K_V1 if pretrained else None
        base = tv_models.swin_b(weights=weights)
        self.feature_extractor = base.features
        self.avgpool = base.avgpool
        in_features = base.head.in_features
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(512, num_classes),
        )
        self.gradcam_layer = self.feature_extractor[-1][-1]

    def _apply_freeze_mode(self, mode: str, unfreeze_layers: int):      
        if mode == "feature":
            for p in self.feature_extractor.parameters(): p.requires_grad = False # Freeze toàn bộ feature extractor    
        elif mode == "partial":
            for p in self.feature_extractor.parameters(): p.requires_grad = False # Freeze toàn bộ feature extractor
            children = list(self.feature_extractor.children()) # Lấy danh sách các layer
            for child in children[-unfreeze_layers:]:       
                for p in child.parameters(): p.requires_grad = True
        for p in self.classifier.parameters(): p.requires_grad = True # Unfreeze classifier

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.backbone_name == "efficientnet" and EFFICIENTNET_AVAILABLE:
            x = self.feature_extractor.extract_features(x)
            x = self.feature_extractor._avg_pooling(x)
            x = x.flatten(start_dim=1)
            x = self.feature_extractor._dropout(x)
        elif self.backbone_name in ["swin_t", "swin_b"]:
            x = self.feature_extractor(x)
            x = x.permute(0, 3, 1, 2)  # Swin output is (B, H, W, C), need (B, C, H, W)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
        else:
            x = self.feature_extractor(x)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
        return self.classifier(x)
