from __future__ import annotations
import inspect
import torch
from torch import nn

from xnerf.utils.base import BaseModule


class APIEncoder(BaseModule):
    """Transformer encoder for API call sequences.

    Inputs:
        api_ids: LongTensor [B, T]
    Outputs:
        embedding: FloatTensor [B, 512]
    Forward:
        forward(api_ids) -> embedding
    Usage:
        model = APIEncoder(vocab_size=4096)
        z = model(torch.randint(0, 4096, (8, 256)))
    """

    def __init__(self, vocab_size: int = 8192, hidden_dim: int = 256, out_dim: int = 512, layers: int = 4, heads: int = 8, max_len: int = 1024):
        super().__init__()
        self.token = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
        self.pos = nn.Embedding(max_len, hidden_dim)
        layer = nn.TransformerEncoderLayer(hidden_dim, heads, hidden_dim * 4, dropout=0.1, batch_first=True, activation="gelu")
        encoder_kwargs = {"num_layers": layers}
        if "enable_nested_tensor" in inspect.signature(nn.TransformerEncoder).parameters:
            encoder_kwargs["enable_nested_tensor"] = False
        self.encoder = nn.TransformerEncoder(layer, **encoder_kwargs)
        self.proj = nn.Linear(hidden_dim, out_dim)
        
    def forward(self, api_ids: torch.Tensor) -> torch.Tensor:
        b, t = api_ids.shape
        pos = torch.arange(t, device=api_ids.device).unsqueeze(0).expand(b, t)
        mask = api_ids.eq(0)
        # Guard against rows that are entirely padding (all tokens == 0).
        # PyTorch's nested-tensor path raises "at least one constituent tensor
        # should have non-zero numel" when every position in a row is masked out.
        # Force at least the first position to be unmasked per row so the
        # encoder always receives a non-empty sequence.
        all_pad = mask.all(dim=1, keepdim=True)   # [B, 1] bool
        mask = mask & ~all_pad                     # unmask position-0 for those rows
        h = self.token(api_ids) + self.pos(pos)
        h = self.encoder(h, src_key_padding_mask=mask)
        denom = (~mask).sum(dim=1, keepdim=True).clamp_min(1)
        pooled = h.masked_fill(mask.unsqueeze(-1), 0).sum(dim=1) / denom
        return self.proj(pooled)