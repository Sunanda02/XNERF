from __future__ import annotations

import networkx as nx
import torch
from torch import nn

from xnerf.utils.base import BaseModule


STAGES = [
    "Environment Check",
    "Privilege Escalation",
    "Persistence",
    "Credential Access",
    "Exfiltration",
]


class TrajectoryDecoder(BaseModule):
    """Decode latent execution fields into explainable behavior trajectories.

    Inputs:
        field: FloatTensor [B,T,D]
    Outputs:
        stage_logits: FloatTensor [B,T,5]
        transition_logits: FloatTensor [B,T-1,5,5]
        graphs: list[networkx.DiGraph]
    Forward:
        forward(field) -> dict
    Usage:
        dec = TrajectoryDecoder()
        out = dec(torch.randn(2,16,1024))
        graphs = dec.reconstruct_graphs(out["stage_logits"])
    """

    def __init__(self, latent_dim: int = 1024, hidden_dim: int = 512, num_stages: int = 5):
        super().__init__()
        self.stage_head = nn.Sequential(nn.Linear(latent_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, num_stages))
        self.trans_head = nn.Sequential(nn.Linear(latent_dim * 2, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, num_stages * num_stages))
        self.num_stages = num_stages

    def forward(self, field: torch.Tensor) -> dict[str, torch.Tensor]:
        stage_logits = self.stage_head(field)
        if field.shape[1] > 1:
            pairs = torch.cat([field[:, :-1], field[:, 1:]], dim=-1)
            transition_logits = self.trans_head(pairs).view(field.shape[0], field.shape[1] - 1, self.num_stages, self.num_stages)
        else:
            transition_logits = field.new_zeros(field.shape[0], 0, self.num_stages, self.num_stages)
        return {"stage_logits": stage_logits, "transition_logits": transition_logits}

    def reconstruct_graphs(self, stage_logits: torch.Tensor) -> list[nx.DiGraph]:
        stages = stage_logits.argmax(dim=-1).detach().cpu().tolist()
        graphs = []
        for seq in stages:
            g = nx.DiGraph()
            for i, sid in enumerate(seq):
                label = STAGES[sid]
                g.add_node(i, stage=label)
                if i:
                    g.add_edge(i - 1, i)
            graphs.append(g)
        return graphs

