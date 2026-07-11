from __future__ import annotations

import torch
from torch import nn

from xnerf.utils.base import BaseModule

try:
    from torch_geometric.nn import GATConv, global_mean_pool
except Exception:  # pragma: no cover
    GATConv = None
    global_mean_pool = None


class CFGEncoder(BaseModule):
    """Function call graph encoder using Graph Attention Networks.

    Inputs:
        x: FloatTensor [N, node_dim]
        edge_index: LongTensor [2, E]
        batch: LongTensor [N] graph id per node
    Outputs:
        embedding: FloatTensor [B, 512]
    Forward:
        forward(x, edge_index, batch) -> embedding
    Usage:
        model = CFGEncoder(node_dim=64)
        z = model(x, edge_index, batch)
    """

    def __init__(self, node_dim: int = 64, hidden_dim: int = 256, out_dim: int = 512, heads: int = 4):
        super().__init__()
        if GATConv is None:
            raise RuntimeError("torch_geometric is required for CFGEncoder")
        self.gat1 = GATConv(node_dim, hidden_dim // heads, heads=heads, dropout=0.1)
        self.gat2 = GATConv(hidden_dim, hidden_dim // heads, heads=heads, dropout=0.1)
        self.norm = nn.LayerNorm(hidden_dim)
        self.proj = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.gat1(x, edge_index))
        h = self.norm(torch.relu(self.gat2(h, edge_index)))
        pooled = global_mean_pool(h, batch)
        return self.proj(pooled)

