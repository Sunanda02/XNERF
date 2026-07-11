from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from xnerf.preprocessing.static_features import (
    binary_image_from_bytes,
    extract_static_modalities,
    zero_memory_trace,
)


SUPPORTED_EXTENSIONS = {
    "",
    ".bin",
    ".dat",
    ".dll",
    ".elf",
    ".exe",
    ".scr",
    ".so",
    ".sys",
    ".apk",
}


class FeatureExtractionError(RuntimeError):
    pass


def validate_input_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"file not found: {path}")
    if not path.is_file():
        raise FeatureExtractionError(f"unsupported input, expected a file: {path}")
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise FeatureExtractionError(f"unsupported file extension '{path.suffix or '<none>'}' for {path}")
    try:
        with open(path, "rb") as f:
            f.read(1)
    except OSError as exc:
        raise FeatureExtractionError(f"could not read file: {path}") from exc


def memory_trace_from_bytes(data: bytes, rows: int = 512, cols: int = 8) -> torch.Tensor:
    """Deprecated compatibility wrapper.

    Training only supplies memory traces from cached feature-vector tensors. A
    standalone binary has no training-equivalent memory cache, so this returns
    the same zero tensor shape the DatasetLoader would emit.
    """

    if rows == 512 and cols == 8:
        return zero_memory_trace()
    return torch.zeros(rows, cols, dtype=torch.float32)


def extract_modalities(path: str | Path, arch: str = "unknown") -> dict[str, Any]:
    sample_path = Path(path)
    arch = str(arch).strip().lower()
    validate_input_file(sample_path)
    try:
        return extract_static_modalities(sample_path, arch_hint=arch)
    except OSError as exc:
        raise FeatureExtractionError(f"feature extraction failed while reading {sample_path}") from exc
    except Exception as exc:
        raise FeatureExtractionError(f"feature extraction failed for {sample_path}: {type(exc).__name__}: {exc}") from exc


def make_model_batch(features: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    batch = {}
    for key, value in features.items():
        if isinstance(value, torch.Tensor):
            if key in {"graph_x", "graph_edge_index"}:
                batch[key] = value.to(device)
            else:
                batch[key] = value.unsqueeze(0).to(device)
    graph_x = batch.get("graph_x")
    if graph_x is not None and graph_x.numel() > 0:
        batch["graph_batch"] = torch.zeros(graph_x.shape[0], dtype=torch.long, device=device)
        batch["graph_sample_ids"] = torch.zeros(1, dtype=torch.long, device=device)
    return batch
