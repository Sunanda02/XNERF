from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from xnerf.preprocessing.ontology import ARCH_TO_ID


def _safe_metric(fn, *args, default: float | None = None, **kwargs):
    try:
        return float(fn(*args, **kwargs))
    except Exception:
        return default


def classification_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    family_true: np.ndarray | None = None,
    family_pred: np.ndarray | None = None,
    arch_true: np.ndarray | None = None,
    arch_pred: np.ndarray | None = None,
) -> dict[str, Any]:
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = y_prob.argmax(axis=1)
    metrics: dict[str, Any] = {
        "accuracy": _safe_metric(accuracy_score, y_true, y_pred),
        "precision": _safe_metric(precision_score, y_true, y_pred, average="weighted", zero_division=0),
        "recall": _safe_metric(recall_score, y_true, y_pred, average="weighted", zero_division=0),
        "f1": _safe_metric(f1_score, y_true, y_pred, average="weighted", zero_division=0),
        "roc_auc": None,
        "family_accuracy": None,
        "architecture_malware_accuracy": None,
        "cross_architecture_accuracy": None,
        "per_architecture_accuracy": {},
    }
    if y_prob.ndim == 2 and y_prob.shape[1] == 2:
        metrics["roc_auc"] = _safe_metric(roc_auc_score, y_true, y_prob[:, 1])
    if family_true is not None and family_pred is not None and len(family_true):
        metrics["family_accuracy"] = _safe_metric(accuracy_score, np.asarray(family_true), np.asarray(family_pred))
    if arch_true is not None and len(arch_true):
        arch_true_arr = np.asarray(arch_true)
        mask = arch_true_arr != ARCH_TO_ID["unknown"]
        if mask.any():
            arch_malware_accuracy = _safe_metric(accuracy_score, y_true[mask], y_pred[mask])
            metrics["architecture_malware_accuracy"] = arch_malware_accuracy
            metrics["cross_architecture_accuracy"] = arch_malware_accuracy
            id_to_arch = {idx: name for name, idx in ARCH_TO_ID.items()}
            metrics["per_architecture_accuracy"] = {
                id_to_arch.get(int(arch_id), f"arch_{int(arch_id)}"): _safe_metric(accuracy_score, y_true[arch_true_arr == arch_id], y_pred[arch_true_arr == arch_id])
                for arch_id in sorted(set(arch_true_arr[mask].tolist()))
            }
    return metrics
