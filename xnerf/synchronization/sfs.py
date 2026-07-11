from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from xnerf.utils.base import BaseModule


class SemanticFieldSynchronizer(BaseModule):
    """Cross-modal attention and temporal synchronization.

    Inputs:
        embeddings: dict[str, FloatTensor [B,512]] for binary,cfg,api,memory,network,isr.
        time_steps: int, default 16.
    Outputs:
        unified: FloatTensor [B,time_steps,2048]
        contrastive_loss(modal_a, modal_b): scalar
    Forward:
        forward(embeddings, time_steps=16) -> unified
    Usage:
        sfs = SemanticFieldSynchronizer()
        y = sfs({"api": torch.randn(4,512), "memory": torch.randn(4,512)})
    """

    def __init__(self, in_dim: int = 512, sync_dim: int = 512, out_dim: int = 2048, heads: int = 8, max_time: int = 128):
        super().__init__()
        self.modalities = ("binary", "cfg", "api", "memory", "network", "isr")
        self.proj = nn.ModuleDict({m: nn.Linear(in_dim, sync_dim) for m in self.modalities})
        self.type_embed = nn.Parameter(torch.randn(len(self.modalities), sync_dim) * 0.02)
        self.time_embed = nn.Embedding(max_time, sync_dim)
        self.attn = nn.MultiheadAttention(sync_dim, heads, batch_first=True)
        self.ffn = nn.Sequential(nn.LayerNorm(sync_dim), nn.Linear(sync_dim, out_dim), nn.GELU(), nn.Linear(out_dim, out_dim))
        self.temporal = nn.GRU(out_dim, out_dim // 2, num_layers=1, batch_first=True, bidirectional=True)

    def forward(self, embeddings: dict[str, torch.Tensor], time_steps: int = 16) -> torch.Tensor:
        present = [m for m in self.modalities if m in embeddings and embeddings[m] is not None]
        if not present:
            raise ValueError("at least one modality embedding is required")
        tokens = []
        for i, m in enumerate(self.modalities):
            if m in embeddings and embeddings[m] is not None:
                tokens.append(self.proj[m](embeddings[m]) + self.type_embed[i])
        x = torch.stack(tokens, dim=1)
        attended, _ = self.attn(x, x, x, need_weights=False)
        summary = attended.mean(dim=1, keepdim=True)
        t = torch.arange(time_steps, device=summary.device).unsqueeze(0)
        temporal_tokens = summary + self.time_embed(t)
        h = self.ffn(temporal_tokens)
        h, _ = self.temporal(h)
        return h

    @staticmethod
    def contrastive_loss(a: torch.Tensor, b: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
        a = F.normalize(a, dim=-1)
        b = F.normalize(b, dim=-1)
        logits = a @ b.t() / temperature
        labels = torch.arange(a.shape[0], device=a.device)
        return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels)) * 0.5
