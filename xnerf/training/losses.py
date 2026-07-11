from __future__ import annotations

import torch
import torch.nn.functional as F


def classification_losses(outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor], family_weight: float = 0.1) -> dict[str, torch.Tensor]:
    losses = {"malware_ce": F.cross_entropy(outputs["malware_logits"], batch["label"])}
    if "family_label" in batch:
        family_label = batch["family_label"]
        if torch.any(family_label.ge(0)):
            losses["family_ce"] = F.cross_entropy(outputs["family_logits"], family_label, ignore_index=-1) * family_weight
        else:
            losses["family_ce"] = outputs["family_logits"].sum() * 0.0

    if "arch_id" in batch and "arch_logits" in outputs:
         arch_id = batch["arch_id"]
         unknown_id = 0  # ARCH_TO_ID["unknown"]
         valid_mask = arch_id != unknown_id

    if torch.any(valid_mask):
        losses["arch_adv"] = (  F.cross_entropy( outputs["arch_logits"][valid_mask],  arch_id[valid_mask] ) * 0.1)

    if outputs["field"].shape[1] > 1:
        losses["field_smooth"] = (outputs["field"][:, 1:] - outputs["field"][:, :-1]).pow(2).mean() * 0.01
        
    return losses


def total_loss(losses: dict[str, torch.Tensor]) -> torch.Tensor:
    return sum(losses.values())
   