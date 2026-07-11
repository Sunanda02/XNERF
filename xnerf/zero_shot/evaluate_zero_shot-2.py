"""evaluate_zero_shot.py — zero-shot prototype classification evaluation.

Changes vs original:
  - Skips benign samples (label == 0) from evaluation; zero-shot family
    classification only makes sense for malware.
  - Skips samples whose family is not represented in the prototype bank
    (already done) but now also logs how many were skipped and why.
  - L2-normalises query embeddings before cosine retrieval (consistent with
    normalised prototypes built by build_prototypes.py).
  - Adds per-family breakdown metrics to the output JSON for debugging.
  - Reports top-5 accuracy in addition to top-1.
  - Sample debug prints show confidence score alongside true/predicted family.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from xnerf.datasets.loaders import MalwareManifestDataset
from xnerf.evaluation.test_after_training import load_model
from xnerf.utils.base import collate_dicts, move_to_device
from xnerf.utils.config import load_config
from xnerf.zero_shot.prototypes import ZeroShotPrototypeClassifier, load_prototype_bank


@torch.no_grad()
def evaluate_zero_shot(
    config_path: str,
    checkpoint_path: str,
    manifest_path: str,
    prototype_path: str,
    out_dir: str | Path,
) -> dict:
    cfg    = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = load_model(Path(checkpoint_path), cfg, device)

    bank       = load_prototype_bank(prototype_path)
    classifier = ZeroShotPrototypeClassifier(bank["prototypes"], bank["labels"]).to(device)
    label_to_id = {label: idx for idx, label in enumerate(bank["labels"])}

    ds = MalwareManifestDataset(manifest_path)
    loader = DataLoader(
        ds,
        batch_size=cfg["training"].get("batch_size", 4),
        shuffle=False,
        num_workers=cfg["training"].get("num_workers", 2), 
        collate_fn=collate_dicts
    )

    y_true, y_pred, probs = [], [], []
    skipped_benign  = 0
    skipped_no_proto = 0

    for batch in tqdm(loader, desc="zero-shot eval"):
        families = [str(x) for x in batch.get("family", [])]
        labels   = batch["label"].tolist()

        keep = []
        for i, (fam, lbl) in enumerate(zip(families, labels)):
            if lbl == 0:
                skipped_benign += 1
            elif fam not in label_to_id:
                skipped_no_proto += 1
            else:
                keep.append(i)

        if not keep:
            continue

        keep_t  = torch.tensor(keep, dtype=torch.long)
        batch_g = move_to_device(batch, device)
        batch_sub = {
            k: v.index_select(0, keep_t.to(device))
            if torch.is_tensor(v) and v.shape[0] == len(families)
            else v
            for k, v in batch_g.items()
        }

        outputs = model(batch_sub)

        # L2-normalise query embedding before cosine retrieval.
        emb_norm = F.normalize(outputs["zero_shot_embedding"].float(), dim=-1)
        zs   = classifier(emb_norm)
        pred = zs["prediction"].detach().cpu().numpy()
        prob = zs["probabilities"].detach().cpu().numpy()

        for j, i in enumerate(keep):
            y_true.append(label_to_id[families[i]])
            y_pred.append(int(pred[j]))
            probs.append(prob[j])

    # --- Debug sample predictions ---
    print(f"\n[zero-shot] skipped {skipped_benign} benign, {skipped_no_proto} no-prototype samples")
    print(f"[zero-shot] evaluated {len(y_true)} malware samples over {len(bank['labels'])} families\n")

    print("Sample predictions (true | predicted | confidence):")
    for i in range(min(20, len(y_true))):
        true_label = bank["labels"][y_true[i]]
        pred_label = bank["labels"][y_pred[i]]
        conf       = float(probs[i][y_pred[i]])
        match      = "✓" if y_true[i] == y_pred[i] else "✗"
        print(f"  {match}  True: {true_label:20} | Pred: {pred_label:20} | conf: {conf:.3f}")

    # --- Metrics ---
    if not y_true:
        metrics = {
            "zero_shot_accuracy": None,
            "zero_shot_f1": None,
            "message": "No test families matched prototype labels.",
            "skipped_benign": skipped_benign,
            "skipped_no_prototype": skipped_no_proto,
        }
    else:
        y_true_arr = np.array(y_true)
        y_pred_arr = np.array(y_pred)
        y_prob_arr = np.array(probs)

        # Top-5 accuracy.
        top5_correct = 0
        for i, prob_row in enumerate(y_prob_arr):
            top5 = np.argsort(prob_row)[::-1][:5]
            if y_true_arr[i] in top5:
                top5_correct += 1
        top5_acc = top5_correct / len(y_true_arr)

        # Per-family breakdown.
        family_names   = bank["labels"]
        per_family: dict[str, dict] = {}
        for fam_id, fam_name in enumerate(family_names):
            mask = y_true_arr == fam_id
            if mask.sum() == 0:
                continue
            fam_correct = (y_pred_arr[mask] == fam_id).sum()
            per_family[fam_name] = {
                "support": int(mask.sum()),
                "correct": int(fam_correct),
                "accuracy": float(fam_correct / mask.sum()),
            }

        metrics = {
            "zero_shot_accuracy":    float(accuracy_score(y_true_arr, y_pred_arr)),
            "zero_shot_top5_accuracy": float(top5_acc),
            "zero_shot_f1":          float(f1_score(y_true_arr, y_pred_arr, average="weighted", zero_division=0)),
            "zero_shot_f1_macro":    float(f1_score(y_true_arr, y_pred_arr, average="macro",    zero_division=0)),
            "prototype_count":       len(bank["labels"]),
            "evaluated_samples":     len(y_true),
            "skipped_benign":        skipped_benign,
            "skipped_no_prototype":  skipped_no_proto,
            "per_family":            per_family,
        }

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "zero_shot_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    if probs:
        np.savez_compressed(
            out / "zero_shot_predictions.npz",
            y_true=np.array(y_true),
            y_pred=np.array(y_pred),
            y_prob=np.array(probs),
            labels=np.array(bank["labels"]),
        )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate zero-shot prototype classification")
    parser.add_argument("--config",     default="xnerf/configs/kaggle.yaml")
    parser.add_argument("--checkpoint", default="/kaggle/working/checkpoints/best.pt")
    parser.add_argument("--manifest",   default="/kaggle/working/data/processed/test_manifest.jsonl")
    parser.add_argument("--prototypes", default="/kaggle/working/runs/zero_shot/prototypes.pt")
    parser.add_argument("--out",        default="/kaggle/working/runs/zero_shot")
    args = parser.parse_args()
    metrics = evaluate_zero_shot(
        args.config, args.checkpoint, args.manifest, args.prototypes, args.out
    )
    print(json.dumps(
        {k: v for k, v in metrics.items() if k != "per_family"},
        indent=2,
    ))


if __name__ == "__main__":
    main()