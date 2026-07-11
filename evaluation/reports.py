from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix


def write_metrics_json(metrics: dict[str, Any], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")


def write_metrics_csv(metrics: dict[str, Any], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for key, value in metrics.items():
            writer.writerow([key, "" if value is None else value])


def save_confusion_matrix_png(y_true: np.ndarray, y_pred: np.ndarray, out: Path, labels: list[str] | None = None) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(7, 6))
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.colorbar()
    plt.xlabel("Predicted")
    plt.ylabel("True")
    if labels and len(labels) == cm.shape[0]:
        ticks = np.arange(len(labels))
        plt.xticks(ticks, labels, rotation=45, ha="right")
        plt.yticks(ticks, labels)
    plt.tight_layout()
    plt.savefig(out, dpi=180)
    plt.close()


def write_standard_results(metrics: dict[str, Any], y_true: np.ndarray, y_prob: np.ndarray, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    write_metrics_json(metrics, out_dir / "metrics.json")
    write_metrics_csv(metrics, out_dir / "metrics.csv")
    save_confusion_matrix_png(y_true, y_prob.argmax(axis=1), out_dir / "confusion_matrix.png", labels=["benign", "malware"])

