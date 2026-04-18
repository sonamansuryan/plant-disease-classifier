import torch
import torch.nn as nn
import timm


class DiseaseClassifier(nn.Module):
    def __init__(self, backbone: str, num_classes: int, dropout: float = 0.3):
        super().__init__()
        self.backbone = timm.create_model(
            backbone, pretrained=True, num_classes=0, global_pool="avg"
        )
        feat_dim = self.backbone.num_features
        self.head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(feat_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))

def freeze_backbone(model: DiseaseClassifier) -> None:
    for p in model.backbone.parameters():
        p.requires_grad = False


def unfreeze_backbone(model: DiseaseClassifier) -> None:
    for p in model.backbone.parameters():
        p.requires_grad = True

def build_model(backbone: str, num_classes: int, dropout: float, device: torch.device) -> DiseaseClassifier:
    model = DiseaseClassifier(backbone, num_classes, dropout).to(device)
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Backbone         : {backbone}")
    print(f"Total params     : {total:,}")
    print(f"Trainable params : {trainable:,}")
    return model

def get_optimizer(model: DiseaseClassifier, lr: float, weight_decay: float, unfrozen: bool):
    if not unfrozen:
        return torch.optim.AdamW(model.head.parameters(), lr=lr, weight_decay=weight_decay)
    return torch.optim.AdamW(
        [
            {"params": model.backbone.parameters(), "lr": lr * 0.05},
            {"params": model.head.parameters(),     "lr": lr * 0.5},
        ],
        weight_decay=weight_decay,
    )
