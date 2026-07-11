from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from xnerf.encoders.api import APIEncoder
from xnerf.encoders.binary_image import BinaryImageEncoder
from xnerf.encoders.cfg import CFGEncoder
from xnerf.utils.base import BaseModule


class CNNMalware(BaseModule):
    """CNN byte-image baseline. Input [B,1,H,W], output logits [B,C]."""

    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 5, stride=2, padding=2), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(128, num_classes),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.net(image[:, :1])


class MalBERT(BaseModule):
    """Transformer token baseline. Input token ids [B,T], output logits [B,C]."""

    def __init__(self, vocab_size: int = 8192, num_classes: int = 2):
        super().__init__()
        self.encoder = APIEncoder(vocab_size=vocab_size)
        self.head = nn.Linear(512, num_classes)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(token_ids))


class HYDRA(BaseModule):
    """Hybrid static/dynamic baseline. Inputs image [B,1,H,W], api_ids [B,T]."""

    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.image = BinaryImageEncoder()
        self.api = APIEncoder()
        self.head = nn.Sequential(nn.Linear(1024, 512), nn.GELU(), nn.Linear(512, num_classes))

    def forward(self, image: torch.Tensor, api_ids: torch.Tensor) -> torch.Tensor:
        return self.head(torch.cat([self.image(image), self.api(api_ids)], dim=-1))


class CrossArchitectureSiamese(BaseModule):
    """Siamese cross-architecture baseline. Inputs two feature tensors [B,D]."""

    def __init__(self, input_dim: int = 512, embed_dim: int = 256):
        super().__init__()
        self.tower = nn.Sequential(nn.Linear(input_dim, 512), nn.GELU(), nn.Linear(512, embed_dim))

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> dict[str, torch.Tensor]:
        za, zb = F.normalize(self.tower(a), dim=-1), F.normalize(self.tower(b), dim=-1)
        return {"za": za, "zb": zb, "similarity": (za * zb).sum(dim=-1)}


class GNNMalware(BaseModule):
    """GNN CFG baseline. Inputs PyG graph tensors, output logits [B,C]."""

    def __init__(self, node_dim: int = 64, num_classes: int = 2):
        super().__init__()
        self.cfg = CFGEncoder(node_dim=node_dim)
        self.head = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        return self.head(self.cfg(x, edge_index, batch))

