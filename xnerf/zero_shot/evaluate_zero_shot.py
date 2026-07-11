from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from xnerf.datasets.loaders import MalwareManifestDataset
from xnerf.evaluation.test_after_training import load_model
from xnerf.utils.base import collate_dicts, move_to_device
from xnerf.utils.config import load_config
from xnerf.zero_shot.prototypes import ZeroShotPrototypeClassifier, load_prototype_bank


@torch.no_grad()
def evaluate_zero_shot(config_path: str, checkpoint_path: str, manifest_path: str, prototype_path: str, out_dir: str | Path) -> dict:
    cfg = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(Path(checkpoint_path), cfg, device)
    bank = load_prototype_bank(prototype_path)
    classifier = ZeroShotPrototypeClassifier(bank["prototypes"], bank["labels"]).to(device)
    label_to_id = {label: idx for idx, label in enumerate(bank["labels"])}

    ds = MalwareManifestDataset(manifest_path, require_cache=True)
    loader = DataLoader(ds, batch_size=cfg["training"].get("batch_size", 4), shuffle=False, num_workers=cfg["training"].get("num_workers", 2), collate_fn=collate_dicts)

    y_true, y_pred, probs = [], [], []
    for batch in tqdm(loader, desc="zero-shot eval"):
        families = [str(x) for x in batch.get("family", [])]
        keep = [i for i, fam in enumerate(families) if fam in label_to_id]
        if not keep:
            continue
        batch = move_to_device(batch, device)
        outputs = model(batch)
        zs = classifier(outputs["zero_shot_embedding"])
        pred = zs["prediction"].detach().cpu().numpy()
        prob = zs["probabilities"].detach().cpu().numpy()
        for i in keep:
            y_true.append(label_to_id[families[i]])
            y_pred.append(int(pred[i]))
            probs.append(prob[i])

    if not y_true:
        metrics = {"zero_shot_accuracy": None, "zero_shot_f1": None, "message": "No test families matched prototype labels."}
    else:
        metrics = {
            "zero_shot_accuracy": accuracy_score(y_true, y_pred),
            "zero_shot_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
            "prototype_count": len(bank["labels"]),
            "evaluated_samples": len(y_true),
        }

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "zero_shot_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    if probs:
        np.savez_compressed(out / "zero_shot_predictions.npz", y_true=np.array(y_true), y_pred=np.array(y_pred), y_prob=np.array(probs), labels=np.array(bank["labels"]))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate zero-shot prototype classification")
    parser.add_argument("--config", default="xnerf/configs/kaggle.yaml")
    parser.add_argument("--checkpoint", default="/kaggle/working/checkpoints/best.pt")
    parser.add_argument("--manifest", default="/kaggle/working/data/processed/test_manifest.jsonl")
    parser.add_argument("--prototypes", default="/kaggle/working/runs/zero_shot/prototypes.pt")
    parser.add_argument("--out", default="/kaggle/working/runs/zero_shot")
    args = parser.parse_args()
    metrics = evaluate_zero_shot(args.config, args.checkpoint, args.manifest, args.prototypes, args.out)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
