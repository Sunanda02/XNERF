from __future__ import annotations
import inspect

import torch
from torch import nn

from xnerf.utils.base import BaseModule


class NetworkEncoder(BaseModule):
    """Transformer encoder for network event tokens.

    Inputs:
        network_ids: LongTensor [B, T]
    Outputs:
        embedding: FloatTensor [B, 512]
    Forward:
        forward(network_ids) -> embedding
    Usage:
        model = NetworkEncoder(vocab_size=2048)
        z = model(torch.randint(0, 2048, (2, 128)))
    """

    def __init__(self, vocab_size: int = 4096, hidden_dim: int = 256, out_dim: int = 512, layers: int = 3, heads: int = 8, max_len: int = 1024):
        super().__init__()
        self.token = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
        self.pos = nn.Embedding(max_len, hidden_dim)
        layer = nn.TransformerEncoderLayer(hidden_dim, heads, hidden_dim * 4, dropout=0.1, batch_first=True, activation="gelu")
        encoder_kwargs = {"num_layers": layers}
        if "enable_nested_tensor" in inspect.signature(nn.TransformerEncoder).parameters:
            encoder_kwargs["enable_nested_tensor"] = False
        self.encoder = nn.TransformerEncoder(layer, **encoder_kwargs)
        self.proj = nn.Linear(hidden_dim, out_dim)

    def forward(self, network_ids: torch.Tensor) -> torch.Tensor:
        b, t = network_ids.shape
        pos = torch.arange(t, device=network_ids.device).unsqueeze(0).expand(b, t)
        mask = network_ids.eq(0)
        all_pad = mask.all(dim=1, keepdim=True)
        mask = mask & ~all_pad
        h = self.token(network_ids) + self.pos(pos)
        h = self.encoder(h, src_key_padding_mask=mask)
        denom = (~mask).sum(dim=1, keepdim=True).clamp_min(1)
        pooled = h.masked_fill(mask.unsqueeze(-1), 0).sum(dim=1) / denom
        return self.proj(pooled)

