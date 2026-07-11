from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from xnerf.utils.base import BaseModule


class GradientReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, lambd: float):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.lambd * grad_output, None


class CrossArchitectureAligner(BaseModule):
    """Adversarial cross-architecture alignment.

    Inputs:
        features: FloatTensor [B,D]
        arch_labels: LongTensor [B]
    Outputs:
        aligned: FloatTensor [B,D]
        arch_logits: FloatTensor [B,num_arch]
    Forward:
        forward(features, grl_lambda=1.0) -> dict
    Usage:
        aligner = CrossArchitectureAligner(feature_dim=2048)
        out = aligner(torch.randn(8,2048))
    """

    def __init__(self, feature_dim: int = 2048, hidden_dim: int = 512, num_arch: int = 6):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(feature_dim, feature_dim), nn.LayerNorm(feature_dim), nn.GELU())
        self.discriminator = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_arch),
        )

    def forward(self, features: torch.Tensor, grl_lambda: float = 1.0) -> dict[str, torch.Tensor]:
        aligned = self.encoder(features)
        rev = GradientReverse.apply(aligned, grl_lambda)
        return {"aligned": aligned, "arch_logits": self.discriminator(rev)}

    @staticmethod
    def losses(out: dict[str, torch.Tensor], arch_labels: torch.Tensor, paired_a: torch.Tensor | None = None, paired_b: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        losses = {"Ladv": F.cross_entropy(out["arch_logits"], arch_labels)}
        if paired_a is not None and paired_b is not None:
            losses["Lcrossarch"] = 1.0 - F.cosine_similarity(paired_a, paired_b, dim=-1).mean()
        return losses

