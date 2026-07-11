from __future__ import annotations

import torch
from torch import nn

from xnerf.alignment.adversarial import CrossArchitectureAligner
from xnerf.encoders.api import APIEncoder
from xnerf.encoders.binary_image import BinaryImageEncoder
from xnerf.encoders.memory import MemoryEncoder
from xnerf.encoders.network import NetworkEncoder
from xnerf.encoders.isr import ISREncoder
from xnerf.fields.mnef import MNEF
from xnerf.renderer.trajectory_decoder import TrajectoryDecoder
from xnerf.synchronization.sfs import SemanticFieldSynchronizer
from xnerf.utils.base import BaseModule
from xnerf.encoders.cfg import CFGEncoder

class XNERFPlusPlus(BaseModule):
    """End-to-end X-NERF++ model.

    Inputs:
        batch dict with binary_image [B,1,H,W], api_ids [B,T], memory_trace [B,T,C],
        network_ids [B,T], arch_id [B].
    Outputs:
        malware_logits [B,num_classes], family_logits [B,num_families],
        zero_shot_embedding [B,2048], trajectory logits, MNEF field.
    Tensor dimensions:
        synchronized state [B,field_time,2048], field [B,field_time,1024].
    Usage:
        model = XNERFPlusPlus(num_classes=2, num_families=32)
        out = model(batch)
    """

    def __init__(self, num_classes: int = 2, num_families: int = 32, field_time: int = 16, use_binary: bool = True):
        super().__init__()
        self.use_binary = use_binary
        self.field_time = field_time
        self.binary = BinaryImageEncoder() if use_binary else None
        self.api = APIEncoder()
        self.graph = CFGEncoder(node_dim=4)
        self.memory = MemoryEncoder()
        self.network = NetworkEncoder()
        self.isr = ISREncoder()
        self.sfs = SemanticFieldSynchronizer()
        self.arch_embed = nn.Embedding(7, 64)
        self.memory_context = nn.Linear(512, 512)
        self.mnef = MNEF()
        self.aligner = CrossArchitectureAligner(feature_dim=2048, num_arch=7)
        self.renderer = TrajectoryDecoder()
        self.malware_head = nn.Linear(2048, num_classes)
        self.family_head = nn.Linear(2048, num_families)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        embeddings = {
            "api": self.api(batch["api_ids"]),
            "memory": self.memory(batch["memory_trace"]),
            "network": self.network(batch["network_ids"]),
        }
        if self.use_binary and "binary_image" in batch:
            embeddings["binary"] = self.binary(batch["binary_image"])

        if ("graph_x" in batch and "graph_edge_index" in batch and batch["graph_x"].numel() > 0):
               cfg = self.graph( batch["graph_x"], batch["graph_edge_index"], batch["graph_batch"],)  
               batch_size = batch["api_ids"].shape[0]
               cfg_full = torch.zeros( batch_size, cfg.shape[1], device=cfg.device, dtype=cfg.dtype,)
               cfg_full[batch["graph_sample_ids"]] = cfg
               embeddings["cfg"] = cfg_full

        else:
             embeddings["cfg"] = torch.zeros( batch["api_ids"].shape[0], 512, device=batch["api_ids"].device, )      
        
        
        if "isr" in batch and batch["isr"].numel() > 0:
            embeddings["isr"] = self.isr(batch["isr"])

        semantic = self.sfs(embeddings, time_steps=self.field_time)
        pooled = semantic.mean(dim=1)
        aligned = self.aligner(pooled)
        b, t, _ = semantic.shape
        coords = torch.linspace(0, 1, t, device=semantic.device).view(1, t, 1).expand(b, t, 1)
        arch = self.arch_embed(batch["arch_id"].clamp(min=0, max=self.arch_embed.num_embeddings - 1)).unsqueeze(1).expand(b, t, 64)
        mem = self.memory_context(embeddings["memory"]).unsqueeze(1).expand(b, t, 512)
        field_out = self.mnef(coords, coords, semantic, mem, arch)
        traj = self.renderer(field_out["field"])
        return {
            "malware_logits": self.malware_head(aligned["aligned"]),
            "family_logits": self.family_head(aligned["aligned"]),
            "zero_shot_embedding": aligned["aligned"],
            "arch_logits": aligned["arch_logits"],
            **field_out,
            **traj,
        }
