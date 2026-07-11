from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import torch

from config import SandboxConfig
from feature_extractor import extract_modalities, make_model_batch
from xnerf.datasets.validation import family_names_from_metadata
from xnerf.model import XNERFPlusPlus
from xnerf.preprocessing.ontology import ARCH_TO_ID


ID_TO_ARCH = {idx: name for name, idx in ARCH_TO_ID.items()}


class InferenceError(RuntimeError):
    pass


def family_name(index: int, checkpoint_meta: dict[str, Any] | None = None) -> str:
    families = family_names_from_metadata(checkpoint_meta)
    if isinstance(families, list) and 0 <= index < len(families):
        return str(families[index])

    sidecar_candidates = [
        Path("data/processed/family_vocab.json"),
        Path("data/family_vocab.json"),
        Path("models/family_vocab.json"),
        Path("models/xnerf_local_inference.families.json"),
        Path("models/xnerf_local_inference.family_names.json"),
        Path("models/xnerf_local_inference.family_vocab.json"),
        Path("models/best.families.json"),
        Path("models/best.family_names.json"),
        Path("models/best.family_vocab.json"),
    ]
    for sidecar in sidecar_candidates:
        if not sidecar.exists():
            continue
        try:
            loaded = json.loads(sidecar.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(loaded, list) and 0 <= index < len(loaded):
            return str(loaded[index])
        if isinstance(loaded, dict):
            family_names = loaded.get("family_names") or loaded.get("id_to_family") or loaded.get("families")
            if isinstance(family_names, list) and 0 <= index < len(family_names):
                return str(family_names[index])

    return f"family_{index}"


def architecture_name(index: int) -> str:
    return ID_TO_ARCH.get(index, "unknown")


def load_model(config: SandboxConfig, device: torch.device) -> tuple[XNERFPlusPlus, dict[str, Any]]:
    checkpoint = Path(config.checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(f"missing checkpoint: {checkpoint}")
    try:
        payload = torch.load(checkpoint, map_location=device)
    except Exception as exc:
        raise InferenceError(f"could not load checkpoint {checkpoint}: {type(exc).__name__}: {exc}") from exc

    payload_dict = payload if isinstance(payload, dict) else {}
    model_cfg = payload_dict.get("model_config", {})
    model = XNERFPlusPlus(
        num_classes=int(model_cfg.get("num_classes", config.num_classes)),
        num_families=int(model_cfg.get("num_families", config.num_families)),
    ).to(device)
    state = payload_dict.get("model", payload_dict.get("state_dict", payload))
    if not isinstance(state, dict):
        raise InferenceError(f"checkpoint does not contain a model state dict: {checkpoint}")
    state = {str(k).removeprefix("module."): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    model.eval()
    return model, payload_dict


def _check_outputs(outputs: dict[str, torch.Tensor]) -> None:
    for name, value in outputs.items():
        if torch.is_tensor(value) and not torch.isfinite(value).all():
            raise InferenceError(f"model produced non-finite output tensor: {name}")


@torch.no_grad()
def run_inference(file_path: str | Path, config: SandboxConfig) -> dict[str, Any]:
    device = torch.device(config.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    started = time.perf_counter()
    features = extract_modalities(file_path, arch=config.arch)
    model, checkpoint_meta = load_model(config, device)
    batch = make_model_batch(features, device)
    print("\n========== SANDBOX SAMPLE ==========")

    print("Binary")
    print(" shape:", batch["binary_image"].shape)
    print(" mean :", batch["binary_image"].float().mean().item())
    print(" max  :", batch["binary_image"].float().max().item())

    print("\nMemory")
    print(" shape:", batch["memory_trace"].shape)
    print(" mean :", batch["memory_trace"].float().mean().item())
    print(" max  :", batch["memory_trace"].float().max().item())

    print("\nAPI")
    print(" shape:", batch["api_ids"].shape)
    print(" nonzero:", (batch["api_ids"] != 0).sum().item())
    print(" sum:", batch["api_ids"].sum().item())

    print("\nNetwork")
    print(" shape:", batch["network_ids"].shape)
    print(" nonzero:", (batch["network_ids"] != 0).sum().item())
    print(" sum:", batch["network_ids"].sum().item())

    print("\nISR")
    print(" shape:", batch["isr"].shape)
    print(" sum:", batch["isr"].sum().item())

    print("\nCFG")
    print(" nodes:", batch["graph_x"].shape)
    print(" edges:", batch["graph_edge_index"].shape)

    print("\nArch:", batch["arch_id"].item())
    print("===================================\n")

    try:
        outputs = model(batch)
    except Exception as exc:
        raise InferenceError(f"model inference failed: {type(exc).__name__}: {exc}") from exc
    _check_outputs(outputs)

    malware_prob = torch.softmax(outputs["malware_logits"], dim=-1)[0, -1].item()
    class_probs = torch.softmax(outputs["malware_logits"], dim=-1)[0]
    family_idx = int(torch.softmax(outputs["family_logits"], dim=-1)[0].argmax().item())
    elapsed = time.perf_counter() - started
    metadata = features["metadata"]

    return {
        "file_name": metadata["file_name"],
        "file_path": metadata["path"],
        "sha256": metadata["sha256"],
        "executable_format": metadata.get("format", "unknown"),
        "malware_probability": malware_prob,
        "decision": "Malware" if malware_prob >= config.decision_threshold else "Benign",
        "predicted_family": family_name(family_idx, checkpoint_meta),
        "input_architecture": metadata.get("arch", "unknown"),
        "detected_architecture_raw": metadata.get("arch_raw", "unknown"),
        "api_token_count": metadata.get("api_token_count", 0),
        "network_token_count": metadata.get("network_token_count", 0),
        "cfg_node_count": metadata.get("cfg_node_count", 0),
        "cfg_edge_count": metadata.get("cfg_edge_count", 0),
        "feature_cache_hit": bool(metadata.get("cache_hit", False)),
        "feature_warnings": list(metadata.get("warnings", [])),
        "confidence_score": float(class_probs.max().item()),
        "inference_time_seconds": elapsed,
        "checkpoint": str(config.checkpoint),
        "device": str(device),
    }


def format_terminal_report(result: dict[str, Any]) -> str:
    return "\n".join(
        [
            "XNERF Terminal Inference Report",
            "=" * 31,
            f"File Name: {result['file_name']}",
            f"Malware Probability: {result['malware_probability']:.6f}",
            f"Malware/Benign Decision: {result['decision']}",
            f"Predicted Family: {result['predicted_family']}",
            f"Executable Format: {result['executable_format']}",
            f"Input Architecture: {result['input_architecture']}",
            f"Extracted API Tokens: {result['api_token_count']}",
            f"Extracted Network Tokens: {result['network_token_count']}",
            f"CFG Nodes/Edges: {result['cfg_node_count']}/{result['cfg_edge_count']}",
            f"Feature Cache Hit: {result['feature_cache_hit']}",
            f"Confidence Score: {result['confidence_score']:.6f}",
            f"Inference Time: {result['inference_time_seconds']:.3f}s",
            *([f"Feature Warnings: {len(result['feature_warnings'])}"] if result.get("feature_warnings") else []),
            *[f"  - {warning}" for warning in result.get("feature_warnings", [])],
        ]
    )
