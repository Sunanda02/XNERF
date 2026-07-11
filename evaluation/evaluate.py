from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from evaluation.metrics import classification_metrics
from evaluation.reports import write_standard_results
from xnerf.datasets.loaders import MalwareManifestDataset
from xnerf.evaluation.test_after_training import load_model
from xnerf.utils.base import move_to_device
from xnerf.utils.config import load_config


def evaluate_npz(predictions: Path, out_dir: Path) -> dict[str, Any]:
    data = np.load(predictions, allow_pickle=True)
    y_true = data["y_true"]
    y_prob = data["y_prob"]
    metrics = classification_metrics(
        y_true=y_true,
        y_prob=y_prob,
        family_true=data["family_true"] if "family_true" in data else None,
        family_pred=data["family_pred"] if "family_pred" in data else None,
        arch_true=data["arch_true"] if "arch_true" in data else None,
        arch_pred=data["arch_pred"] if "arch_pred" in data else None,
    )
    write_standard_results(metrics, y_true, y_prob, out_dir)
    return metrics


@torch.no_grad()
def evaluate_manifest(config_path: Path, manifest: Path, checkpoint: Path, out_dir: Path) -> dict[str, Any]:
    cfg = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(checkpoint, cfg, device)
    ds = MalwareManifestDataset(manifest, require_cache=True)
    loader = DataLoader(
        ds,
        batch_size=int(cfg["training"].get("batch_size", 4)),
        shuffle=False,
        num_workers=int(cfg["training"].get("num_workers", 2)),
    )

    probs, labels, arch_true = [], [], []
    family_pred = []
    for batch in tqdm(loader, desc="evaluate"):
        batch = move_to_device(batch, device)
        outputs = model(batch)
        probs.append(torch.softmax(outputs["malware_logits"], dim=-1).cpu().numpy())
        labels.append(batch["label"].cpu().numpy())
        arch_true.append(batch["arch_id"].cpu().numpy())
        family_pred.append(outputs["family_logits"].argmax(dim=-1).cpu().numpy())

    y_prob = np.concatenate(probs)
    y_true = np.concatenate(labels)
    arch_true_arr = np.concatenate(arch_true)
    family_pred_arr = np.concatenate(family_pred)
    metrics = classification_metrics(
        y_true=y_true,
        y_prob=y_prob,
        arch_true=arch_true_arr,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    write_standard_results(metrics, y_true, y_prob, out_dir)
    np.savez_compressed(
        out_dir / "predictions.npz",
        y_true=y_true,
        y_prob=y_prob,
        arch_true=arch_true_arr,
        family_pred=family_pred_arr,
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate XNERF predictions or a checkpoint+manifest")
    parser.add_argument("--predictions", type=Path, help="NPZ containing y_true/y_prob and optional family/arch arrays")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--out", type=Path, default=Path("results"))
    args = parser.parse_args()

    if args.predictions:
        metrics = evaluate_npz(args.predictions, args.out)
    else:
        if not args.manifest or not args.checkpoint:
            raise SystemExit("--manifest and --checkpoint are required when --predictions is not supplied")
        metrics = evaluate_manifest(args.config, args.manifest, args.checkpoint, args.out)
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
