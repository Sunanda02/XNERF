from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from xnerf.utils.base import BaseModule


class ZeroShotPrototypeClassifier(BaseModule):
    """Cosine-similarity zero-shot classifier over semantic prototypes.

    Inputs:
        embeddings: FloatTensor [B, D], usually XNERFPlusPlus["zero_shot_embedding"].
        prototypes: FloatTensor [K, D] stored in a prototype bank.
    Outputs:
        logits: FloatTensor [B, K]
        probabilities: FloatTensor [B, K]
        prediction: LongTensor [B]
    Forward:
        forward(embeddings) -> dict
    Usage:
        bank = load_prototype_bank("runs/zero_shot/prototypes.pt")
        clf = ZeroShotPrototypeClassifier(bank["prototypes"], bank["labels"])
        out = clf(torch.randn(4, 2048))
    """

    def __init__(self, prototypes: torch.Tensor, labels: list[str], temperature: float = 0.07):
        super().__init__()
        if prototypes.dim() != 2:
            raise ValueError("prototypes must be [K,D]")
        self.register_buffer("prototypes", F.normalize(prototypes.float(), dim=-1))
        self.labels = labels
        self.temperature = temperature

    def forward(self, embeddings: torch.Tensor) -> dict[str, torch.Tensor]:
        emb = F.normalize(embeddings.float(), dim=-1)
        logits = emb @ self.prototypes.t() / self.temperature
        return {
            "logits": logits,
            "probabilities": torch.softmax(logits, dim=-1),
            "prediction": logits.argmax(dim=-1),
        }


def save_prototype_bank(path: str | Path, prototypes: torch.Tensor, labels: list[str], metadata: dict[str, Any] | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": "xnerf-zero-shot-prototype-bank-v1",
            "prototypes": F.normalize(prototypes.detach().cpu().float(), dim=-1),
            "labels": labels,
            "metadata": metadata or {},
        },
        path,
    )
    return path


def load_prototype_bank(path: str | Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu")

