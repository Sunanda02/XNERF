from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class SandboxConfig:
    checkpoint: Path = Path("checkpoints/best_model.pt")
    decision_threshold: float = 0.5
    num_classes: int = 2
    num_families: int = 32
    arch: str = "unknown"
    device: str | None = None


def _load_yaml(path: Path) -> dict[str,
                                    Any]:
    try:
        import yaml
    except ImportError:
        return {}
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return data if isinstance(data, dict) else {}


def _resolve_checkpoint_path(checkpoint: str | Path) -> Path:
    candidate = Path(checkpoint)
    if candidate.exists():
        return candidate

    fallback_candidates = [
        Path("models/best.pt"),
        Path("checkpoints/best.pt"),
        Path("checkpoints/best_model.pt"),
        Path("checkpoints_balanced_small/best.pt"),
        Path("checkpoints_balanced_small/last.pt"),
        Path("checkpoints_balanced_label_modality/last.pt"),
        Path("checkpoints_publication_v2_50k/best.pt"),
    ]
    for fallback in fallback_candidates:
        if fallback.exists():
            return fallback

    return candidate


def load_sandbox_config(config_path: str | Path | None = None) -> SandboxConfig:
    path = Path(config_path or os.getenv("XNERF_SANDBOX_CONFIG", "config_publication_v2_50k.yaml"))
    cfg = _load_yaml(path)
    model_cfg = cfg.get("model", {}) if isinstance(cfg.get("model", {}), dict) else {}
    sandbox_cfg = cfg.get("sandbox", {}) if isinstance(cfg.get("sandbox", {}), dict) else {}
    export_cfg = cfg.get("export", {}) if isinstance(cfg.get("export", {}), dict) else {}
    local_cfg = cfg.get("local_inference", {}) if isinstance(cfg.get("local_inference", {}), dict) else {}

    checkpoint = (
        os.getenv("XNERF_SANDBOX_CHECKPOINT")
        or sandbox_cfg.get("checkpoint")
        or export_cfg.get("checkpoint")
        or local_cfg.get("checkpoint")
        or "checkpoints_publication_v2_50k/best.pt"
    )

    return SandboxConfig(
        checkpoint=_resolve_checkpoint_path(checkpoint),
        decision_threshold=float(sandbox_cfg.get("decision_threshold", 0.5)),
        num_classes=int(model_cfg.get("num_classes", 2)),
        num_families=int(model_cfg.get("num_families", 32)),
        arch=str(sandbox_cfg.get("arch", "unknown")).strip().lower(),
        device=sandbox_cfg.get("device"),
    )
