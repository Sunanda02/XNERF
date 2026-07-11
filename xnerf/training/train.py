"""train.py — training and validation entry-points.

Called by kaggle_run.py and local_run.py, but also usable standalone:

    # Train
    python -m xnerf.training.train --config config.yaml
    python -m xnerf.training.train --config config.yaml --resume checkpoints/last.pt

    # Validate an existing checkpoint
    python -m xnerf.training.train --config config.yaml --validate-only
    python -m xnerf.training.train --config config.yaml --validate-only --checkpoint checkpoints/best.pt
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from xnerf.datasets.loaders import MalwareManifestDataset
from xnerf.evaluation.evaluate import evaluate_predictions
from xnerf.evaluation.test_after_training import load_model
from xnerf.model import XNERFPlusPlus
from xnerf.training.trainer import XNerfTrainer
from xnerf.utils.base import collate_dicts, move_to_device
from xnerf.utils.config import load_config
from xnerf.utils.seed import seed_everything

# Keys forwarded to XNerfTrainer; anything else in cfg["training"] is ignored.
_TRAINER_KEYS = {
    "batch_size", "lr", "epochs", "grad_accum",
    "num_workers", "checkpoint_dir", "patience", "resume_from", "grad_clip",
    "use_amp", "debug_max_batches",
}


def run_training(config_path: str = "config.yaml", resume_from: str | None = None) -> dict:
    """Train the model and return a metrics dict.

    Writes:
        <checkpoint_dir>/best.pt      — best checkpoint (via XNerfTrainer)
        runs/train_metrics.json       — metrics (also returned)
    """
    cfg = load_config(config_path)
    seed_everything(cfg.get("seed", 1337))

    train_ds = MalwareManifestDataset(cfg["data"]["train_manifest"], require_cache=True)
    val_path = cfg["data"].get("val_manifest")
    val_ds = MalwareManifestDataset(val_path, require_cache=True) if val_path else None

    model = XNERFPlusPlus(
        num_classes=cfg["model"]["num_classes"],
        num_families=cfg["model"]["num_families"],
    )

    trainer_kwargs = {k: v for k, v in cfg["training"].items() if k in _TRAINER_KEYS}
    if resume_from:
        trainer_kwargs["resume_from"] = resume_from
    trainer = XNerfTrainer(model, train_ds, val_ds, family_names=getattr(train_ds, "family_names", None), **trainer_kwargs)
    metrics = trainer.fit()

    out_dir = Path("runs")
    out_dir.mkdir(exist_ok=True)
    (out_dir / "train_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    return metrics


@torch.no_grad()
def run_validation(config_path: str = "config.yaml", checkpoint_path: str | None = None) -> dict:
    """Run the validation loop on the val split and return metrics.

    Loads the checkpoint, runs inference on val_manifest, and computes
    the same metrics as the test stage (accuracy, F1, ROC-AUC, etc.).

    Writes:
        runs/val_metrics.json
    """
    cfg = load_config(config_path)
    val_manifest = cfg["data"].get("val_manifest")
    if not val_manifest:
        raise ValueError("config data.val_manifest is required for validation")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = Path(
        checkpoint_path
        or cfg.get("export", {}).get("checkpoint")
        or (Path(cfg["training"]["checkpoint_dir"]) / "best.pt")
    )
    model = load_model(ckpt_path, cfg, device)

    ds = MalwareManifestDataset(val_manifest, require_cache=True)
    loader = DataLoader(
        ds,
        batch_size=cfg["training"].get("batch_size", 4),
        shuffle=False,
        num_workers=cfg["training"].get("num_workers", 2),
         collate_fn=collate_dicts,
    )

    probs, labels = [], []
    for batch in tqdm(loader, desc="validate"):
        batch = move_to_device(batch, device)
       # Print the first REAL executable sample only
    paths = batch["path"]

    for i, p in enumerate(paths):
     p = str(p).lower()

     if p.endswith((".exe", ".dll", ".elf", ".so", ".apk")):
        print("\n========== TRAIN EXECUTABLE SAMPLE ==========")

        print("Path:", batch["path"][i])
        print("Dataset:", batch["dataset"][i])
        print("Sample ID:", batch["sample_id"][i])

        print("\nBinary")
        print(" mean:", batch["binary_image"][i].float().mean().item())
        print(" max :", batch["binary_image"][i].float().max().item())

        print("\nMemory")
        print(" mean:", batch["memory_trace"][i].float().mean().item())
        print(" max :", batch["memory_trace"][i].float().max().item())

        print("\nAPI")
        print(" nonzero:", (batch["api_ids"][i] != 0).sum().item())
        print(" sum:", batch["api_ids"][i].sum().item())

        print("\nNetwork")
        print(" nonzero:", (batch["network_ids"][i] != 0).sum().item())
        print(" sum:", batch["network_ids"][i].sum().item())

        print("\nISR")
        print(" sum:", batch["isr"][i].sum().item())

        print("\nCFG")
        print(" nodes:", batch["graph_x"].shape)
        print(" edges:", batch["graph_edge_index"].shape)

        print("\nArch:", batch["arch_id"][i].item())
        print("=============================================\n")

        raise SystemExit
        outputs = model(batch)
        probs.append(torch.softmax(outputs["malware_logits"], dim=-1).cpu().numpy())
        labels.append(batch["label"].cpu().numpy())

    y_prob = np.concatenate(probs)
    y_true = np.concatenate(labels)
    metrics = evaluate_predictions(y_true, y_prob)

    out_dir = Path("runs")
    out_dir.mkdir(exist_ok=True)
    (out_dir / "val_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="X-NERF++ training / validation")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Skip training; run the validation loop on an existing checkpoint",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Checkpoint path for --validate-only (defaults to config checkpoint_dir/best.pt)",
    )
    parser.add_argument(
        "--resume",
        default=None,
        help="Resume training from a full checkpoint, for example checkpoints/last.pt",
    )
    args = parser.parse_args()

    if args.validate_only:
        metrics = run_validation(args.config, checkpoint_path=args.checkpoint)
    else:
        metrics = run_training(args.config, resume_from=args.resume)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
