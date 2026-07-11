"""Shared contracts for X-NERF++.

Every model inherits BaseModule, every preprocessing component inherits
Processor, every trainer inherits Trainer, and every dataset inherits
DatasetLoader. Subclasses document inputs, outputs, tensor dimensions, and a
minimal usage example in their docstrings.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

import torch
from torch import nn
from torch.utils.data import Dataset


class BaseModule(nn.Module):
    """Base class for all neural modules.

    Inputs: implementation-specific tensors or dictionaries.
    Outputs: implementation-specific tensors or dictionaries.
    Tensor dimensions: subclasses must declare dimensions in docstrings.
    Usage:
        model = SomeModule(...)
        y = model(x)
    """

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def save(self, path: str | Path) -> None:
        torch.save({"state_dict": self.state_dict(), "class": self.__class__.__name__}, path)


class Processor(ABC):
    """Base class for deterministic preprocessing components."""

    @abstractmethod
    def process(self, item: Any) -> Any:
        raise NotImplementedError


class Trainer(ABC):
    """Base class for training loops."""

    @abstractmethod
    def fit(self) -> Dict[str, Any]:
        raise NotImplementedError


class DatasetLoader(Dataset, ABC):
    """Base dataset contract compatible with torch DataLoader."""

    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, index: int) -> Mapping[str, Any]:
        raise NotImplementedError


@dataclass
class XNerfBatch:
    """Canonical multimodal batch.

    binary_image: [B, 1 or 3, H, W]
    cfg_x: [N, node_dim], cfg_edge_index: [2, E], cfg_batch: [N]
    api_ids: [B, T_api], memory_trace: [B, T_mem, C_mem]
    network_ids: [B, T_net], isr: [B, T_isr]
    arch_id: [B], label: [B], family: optional list[str]
    """

    binary_image: Optional[torch.Tensor] = None
    cfg_x: Optional[torch.Tensor] = None
    cfg_edge_index: Optional[torch.Tensor] = None
    cfg_batch: Optional[torch.Tensor] = None
    api_ids: Optional[torch.Tensor] = None
    memory_trace: Optional[torch.Tensor] = None
    network_ids: Optional[torch.Tensor] = None
    isr: Optional[torch.Tensor] = None
    arch_id: Optional[torch.Tensor] = None
    label: Optional[torch.Tensor] = None
    family: Optional[list[str]] = None

    def to(self, device: torch.device | str) -> "XNerfBatch":
        payload = {}
        for k, v in self.__dict__.items():
            payload[k] = v.to(device) if isinstance(v, torch.Tensor) else v
        return XNerfBatch(**payload)


def move_to_device(batch: Mapping[str, Any], device: torch.device | str) -> Dict[str, Any]:
    return {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}


def collate_dicts(items):
    items = list(items)

    out = {}

    # ----- Graph batching -----
    has_graph = any(
        isinstance(x.get("graph_x"), torch.Tensor)
        and x["graph_x"].numel() > 0
        for x in items
    )

    if has_graph:
        graph_x = []
        graph_edge_index = []
        graph_batch = []

        node_offset = 0
        graph_id = 0
        graph_sample_ids = []

        for i, sample in enumerate(items):

            gx = sample.get("graph_x")
            ge = sample.get("graph_edge_index")

            if (not isinstance(gx, torch.Tensor)  or gx.numel() == 0):
                continue

            graph_x.append(gx)

            graph_edge_index.append(ge + node_offset )

            graph_batch.append( torch.full( (gx.shape[0],), graph_id, dtype=torch.long, ) )
            graph_sample_ids.append(i)
            graph_id += 1

            node_offset += gx.shape[0]
        
        out["graph_x"] = torch.cat(graph_x, dim=0)
        out["graph_edge_index"] = torch.cat(
            graph_edge_index,
            dim=1,
        )
        out["graph_batch"] = torch.cat( graph_batch, dim=0,)
        out["graph_sample_ids"] = torch.tensor( graph_sample_ids, dtype=torch.long,)
    else:
            out["graph_x"] = torch.zeros((0, 4),dtype=torch.float32,)

            out["graph_edge_index"] = torch.zeros((2, 0),dtype=torch.long,)

            out["graph_batch"] = torch.zeros((0,),dtype=torch.long,)
            out["graph_sample_ids"] = torch.zeros((0,), dtype=torch.long)

    # ----- Everything else -----
    for key in items[0].keys():

        if key in {
            "graph_x",
            "graph_edge_index",
            "graph_batch",
        }:
            continue

        vals = [x[key] for x in items]

        if all(isinstance(v, torch.Tensor) for v in vals):
            try:
                out[key] = torch.stack(vals)
            except RuntimeError:
                out[key] = vals
        else:
            out[key] = vals

    return out
