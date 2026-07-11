from __future__ import annotations

import torch
from torch import nn

from xnerf.preprocessing.ontology import TOKEN_TO_ID
from xnerf.utils.base import BaseModule


class ISREncoder(BaseModule):
    """Intermediate Semantic Representation encoder.

    Inputs:
        isr: LongTensor [B,T,4] with columns
        semantic_id, arch_id, address_delta_bucket, instruction_size.
    Outputs:
        embedding: FloatTensor [B,512]
    Forward:
        forward(isr) -> embedding
    Usage:
        enc = ISREncoder()
        z = enc(torch.zeros(4, 1024, 4, dtype=torch.long))
    """

    def __init__(self, hidden_dim: int = 256, out_dim: int = 512):
        super().__init__()
        self.semantic = nn.Embedding(len(TOKEN_TO_ID), hidden_dim, padding_idx=0)
        self.arch = nn.Embedding(7, hidden_dim, padding_idx=0)
        self.delta = nn.Embedding(256, hidden_dim, padding_idx=0)
        self.size = nn.Embedding(32, hidden_dim, padding_idx=0)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.norm = nn.LayerNorm(hidden_dim)
        self.proj = nn.Linear(hidden_dim, out_dim)

    def forward(self, isr: torch.Tensor) -> torch.Tensor:
        ids = isr.long()
        semantic = ids[..., 0].clamp(0, self.semantic.num_embeddings - 1)
        arch = ids[..., 1].clamp(0, self.arch.num_embeddings - 1)
        delta = ids[..., 2].clamp(0, self.delta.num_embeddings - 1)
        size = ids[..., 3].clamp(0, self.size.num_embeddings - 1)
        pad_mask = semantic.eq(0)

        all_pad = pad_mask.all(dim=1)

        if torch.all(all_pad):
           return torch.zeros( isr.shape[0], self.proj.out_features, device=isr.device, dtype=self.proj.weight.dtype,)
        
        if torch.any(all_pad):
             pad_mask = pad_mask.clone()
             pad_mask[all_pad, 0] = False
        h = (
            self.semantic(semantic)
            + self.arch(arch)
            + self.delta(delta)
            + self.size(size)
        )
        h = self.encoder(h, src_key_padding_mask=pad_mask)
        valid = (~pad_mask).unsqueeze(-1).to(h.dtype)
        pooled = (h * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
        return self.proj(self.norm(pooled))
