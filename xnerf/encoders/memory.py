from __future__ import annotations

import torch
from torch import nn

from xnerf.utils.base import BaseModule


class MemoryEncoder(BaseModule):
    """Temporal CNN for memory access traces.

    Inputs:
        memory_trace: FloatTensor [B, T, C]
    Outputs:
        embedding: FloatTensor [B, 512]
    Forward:
        forward(memory_trace) -> embedding
    Usage:
        model = MemoryEncoder(input_dim=8)
        z = model(torch.randn(4, 512, 8))
    """

    def __init__(self, input_dim: int = 8, hidden_dim: int = 256, out_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(input_dim, 128, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(128, hidden_dim, kernel_size=5, padding=4, dilation=2),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=4, dilation=4),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.proj = nn.Linear(hidden_dim, out_dim)

    def forward(self, memory_trace: torch.Tensor) -> torch.Tensor:
        h = memory_trace.transpose(1, 2)
        return self.proj(self.net(h).squeeze(-1))

