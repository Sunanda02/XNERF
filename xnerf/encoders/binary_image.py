from __future__ import annotations

import torch
from torch import nn

from xnerf.utils.base import BaseModule

try:
    from torchvision.models import resnet18
except Exception:  # pragma: no cover
    resnet18 = None


class BinaryImageEncoder(BaseModule):
    """ResNet18 malware byte-image encoder.

    Inputs:
        image: FloatTensor [B, 1 or 3, H, W], values in [0,1]
    Outputs:
        embedding: FloatTensor [B, 512]
    Forward:
        forward(image) -> embedding
    Usage:
        model = BinaryImageEncoder()
        z = model(torch.rand(4, 1, 256, 256))
    """

    def __init__(self, out_dim: int = 512):
        super().__init__()
        if resnet18 is None:
            raise RuntimeError("torchvision is required for BinaryImageEncoder")
        backbone = resnet18(weights=None)
        backbone.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.proj = nn.Linear(in_features, out_dim)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if image.dim() != 4:
            raise ValueError("image must be [B,C,H,W]")
        if image.shape[1] == 1:
            image = image.repeat(1, 3, 1, 1)
        return self.proj(self.backbone(image))

