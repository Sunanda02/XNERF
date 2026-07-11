from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F

from xnerf.utils.base import BaseModule


class PositionalEncoding(nn.Module):
    def __init__(self, bands: int = 8):
        super().__init__()
        self.bands = bands

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        freqs = 2 ** torch.arange(self.bands, device=x.device, dtype=x.dtype) * math.pi
        xb = x.unsqueeze(-1) * freqs
        return torch.cat([x, xb.sin().flatten(-2), xb.cos().flatten(-2)], dim=-1)


class MNEF(BaseModule):
    """Malware Neural Execution Field: F(x,t,s,m,a).

    Inputs:
        x: FloatTensor [B,T,1] execution position normalized to [0,1]
        t: FloatTensor [B,T,1] normalized time
        s: FloatTensor [B,T,2048] semantic state
        m: FloatTensor [B,T,512] memory context
        a: FloatTensor [B,T,64] architecture embedding
    Outputs:
        field: FloatTensor [B,T,latent_dim]
        logits: FloatTensor [B,T,num_behaviors]
    Forward:
        forward(x,t,s,m,a) -> {"field": field, "behavior_logits": logits}
    Usage:
        model = MNEF()
        out = model(x,t,s,m,a)
    """

    def __init__(self, semantic_dim: int = 2048, memory_dim: int = 512, arch_dim: int = 64, latent_dim: int = 1024, num_behaviors: int = 5):
        super().__init__()
        self.pe = PositionalEncoding(8)
        coord_dim = 1 + 2 * 8
        in_dim = coord_dim * 2 + semantic_dim + memory_dim + arch_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim),
        )
        self.behavior_head = nn.Linear(latent_dim, num_behaviors)

    def forward(self, x: torch.Tensor, t: torch.Tensor, s: torch.Tensor, m: torch.Tensor, a: torch.Tensor) -> dict[str, torch.Tensor]:
        h = torch.cat([self.pe(x), self.pe(t), s, m, a], dim=-1)
        field = self.net(h)
        return {"field": field, "behavior_logits": self.behavior_head(field)}

    @staticmethod
    def field_losses(outputs: dict[str, torch.Tensor], behavior_targets: torch.Tensor | None = None, smooth_weight: float = 0.01) -> dict[str, torch.Tensor]:
        field = outputs["field"]
        losses = {}
        if behavior_targets is not None:
            losses["behavior_ce"] = F.cross_entropy(outputs["behavior_logits"].flatten(0, 1), behavior_targets.flatten())
        if field.shape[1] > 1:
            losses["temporal_smooth"] = (field[:, 1:] - field[:, :-1]).pow(2).mean() * smooth_weight
        return losses

