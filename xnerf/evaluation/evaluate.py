from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

from xnerf.preprocessing.ontology import ARCH_TO_ID


def evaluate_predictions(y_true: np.ndarray, y_prob: np.ndarray, arch_true: np.ndarray | None = None, arch_pred: np.ndarray | None = None, zero_shot_mask: np.ndarray | None = None) -> dict:
    y_pred = y_prob.argmax(axis=1)
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }
    if y_prob.shape[1] == 2:
        metrics["roc_auc"] = roc_auc_score(y_true, y_prob[:, 1])
    if zero_shot_mask is not None and zero_shot_mask.any():
        metrics["zero_shot_accuracy"] = accuracy_score(y_true[zero_shot_mask], y_pred[zero_shot_mask])
    if arch_true is not None:
        arch_true_arr = np.asarray(arch_true)
        mask = arch_true_arr != ARCH_TO_ID["unknown"]
        if mask.any():
            arch_malware_accuracy = accuracy_score(y_true[mask], y_pred[mask])
            metrics["architecture_malware_accuracy"] = arch_malware_accuracy
            metrics["cross_architecture_accuracy"] = arch_malware_accuracy
            id_to_arch = {idx: name for name, idx in ARCH_TO_ID.items()}
            per_architecture_accuracy = {}
            for arch_id in sorted(set(arch_true_arr[mask].tolist())):
                arch_mask = arch_true_arr == arch_id
                per_architecture_accuracy[id_to_arch.get(int(arch_id), f"arch_{int(arch_id)}")] = accuracy_score(y_true[arch_mask], y_pred[arch_mask])
            metrics["per_architecture_accuracy"] = per_architecture_accuracy
        else:
            metrics["architecture_malware_accuracy"] = None
            metrics["cross_architecture_accuracy"] = None
            metrics["per_architecture_accuracy"] = {}
    return metrics


def save_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, out: Path) -> None:
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.colorbar()
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(out)
    plt.close()


def save_tsne(embeddings: np.ndarray, labels: np.ndarray, out: Path) -> None:
    coords = TSNE(n_components=2, perplexity=min(30, max(2, len(labels) // 3)), init="pca", learning_rate="auto").fit_transform(embeddings)
    plt.figure(figsize=(7, 6))
    plt.scatter(coords[:, 0], coords[:, 1], c=labels, s=8, cmap="tab10")
    plt.tight_layout()
    plt.savefig(out)
    plt.close()


def save_umap(embeddings: np.ndarray, labels: np.ndarray, out: Path) -> None:
    try:
        import umap
        coords = umap.UMAP(n_components=2, random_state=42).fit_transform(embeddings)
    except Exception:
        coords = TSNE(n_components=2, init="pca", learning_rate="auto").fit_transform(embeddings)
    plt.figure(figsize=(7, 6))
    plt.scatter(coords[:, 0], coords[:, 1], c=labels, s=8, cmap="tab10")
    plt.tight_layout()
    plt.savefig(out)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True, help=".npz with y_true, y_prob, embeddings")
    parser.add_argument("--out", default="runs/eval", type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    data = np.load(args.predictions)
    metrics = evaluate_predictions(data["y_true"], data["y_prob"])
    (args.out / "metrics.json").write_text(__import__("json").dumps(metrics, indent=2), encoding="utf-8")
    save_confusion_matrix(data["y_true"], data["y_prob"].argmax(axis=1), args.out / "confusion_matrix.png")
    if "embeddings" in data:
        save_tsne(data["embeddings"], data["y_true"], args.out / "tsne.png")
        save_umap(data["embeddings"], data["y_true"], args.out / "umap.png")


if __name__ == "__main__":
    main()
