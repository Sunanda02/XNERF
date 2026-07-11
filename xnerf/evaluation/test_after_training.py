from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from xnerf.datasets.loaders import MalwareManifestDataset
from xnerf.evaluation.evaluate import evaluate_predictions, save_confusion_matrix, save_tsne, save_umap
from xnerf.model import XNERFPlusPlus
from xnerf.utils.base import collate_dicts, move_to_device
from xnerf.utils.config import load_config


def load_model(checkpoint_path: Path, cfg: dict, device: torch.device) -> XNERFPlusPlus:
    model = XNERFPlusPlus(
        num_classes=int(cfg["model"].get("num_classes", 2)),
        num_families=int(cfg["model"].get("num_families", 32)),
    ).to(device)
    payload = torch.load(checkpoint_path, map_location=device)
    state = payload.get("model", payload.get("state_dict", payload))
    state = {k.removeprefix("module."): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    return model.eval()


@torch.no_grad()
def run_test(config_path: str, checkpoint_path: str | None = None, out_dir: str | Path = "runs/test") -> dict:
    cfg = load_config(config_path)
    test_manifest = cfg["data"].get("test_manifest")
    if not test_manifest:
        raise ValueError("config data.test_manifest is required for test evaluation")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = Path(checkpoint_path or cfg["training"]["checkpoint_dir"]) / "best.pt"
    if checkpoint_path:
        checkpoint = Path(checkpoint_path)

    model = load_model(checkpoint, cfg, device)
    ds = MalwareManifestDataset(test_manifest, require_cache=True)
    loader = DataLoader(ds, batch_size=cfg["training"].get("batch_size", 4), shuffle=False, num_workers=cfg["training"].get("num_workers", 2), collate_fn=collate_dicts)

    probs, labels, embeddings, arch_labels = [], [], [], []
    for batch in tqdm(loader, desc="test"):
        batch = move_to_device(batch, device)
        outputs = model(batch)
        probs.append(torch.softmax(outputs["malware_logits"], dim=-1).cpu().numpy())
        labels.append(batch["label"].cpu().numpy())
        embeddings.append(outputs["zero_shot_embedding"].cpu().numpy())
        arch_labels.append(batch["arch_id"].cpu().numpy())

    y_prob = np.concatenate(probs)
    y_true = np.concatenate(labels)
    emb = np.concatenate(embeddings)
    arch_true = np.concatenate(arch_labels)

    metrics = evaluate_predictions(y_true, y_prob, arch_true=arch_true)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    (out_path / "test_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    np.savez_compressed(out_path / "test_predictions.npz", y_true=y_true, y_prob=y_prob, embeddings=emb, arch_true=arch_true)
    save_confusion_matrix(y_true, y_prob.argmax(axis=1), out_path / "confusion_matrix.png")
    if len(y_true) >= 3:
        save_tsne(emb, y_true, out_path / "tsne.png")
        save_umap(emb, y_true, out_path / "umap.png")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the best X-NERF++ checkpoint on the test split")
    parser.add_argument("--config", default="xnerf/configs/kaggle.yaml")
    parser.add_argument("--checkpoint")
    parser.add_argument("--out", default="/kaggle/working/runs/test")
    args = parser.parse_args()
    metrics = run_test(args.config, args.checkpoint, args.out)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
