"""local_run.py — local pipeline with both monolithic and split modes.

Split subcommands (run each independently):
    build-manifest   Scan data/raw and build train/val/test manifests.
    train            Train the model; writes checkpoints/best.pt + train_done.json.
    validate         Run validation loop on val split from an existing checkpoint.
    test             Evaluate best.pt on the test split.
    zero-shot        Build prototype bank then evaluate zero-shot accuracy.
    export           Package best.pt into a local-inference checkpoint.

Monolithic subcommand (runs everything at once):
    pipeline         Run all stages end-to-end in a single call.

Examples — split mode:
    python -m xnerf.pipeline.local_run build-manifest --config config.yaml
    python -m xnerf.pipeline.local_run train           --config config.yaml
    python -m xnerf.pipeline.local_run validate        --config config.yaml
    python -m xnerf.pipeline.local_run test            --config config.yaml
    python -m xnerf.pipeline.local_run zero-shot       --config config.yaml
    python -m xnerf.pipeline.local_run export          --config config.yaml

    # Point at a specific checkpoint without retraining:
    python -m xnerf.pipeline.local_run test --checkpoint checkpoints/epoch3.pt

Examples — monolithic mode:
    python -m xnerf.pipeline.local_run pipeline --config config.yaml
    python -m xnerf.pipeline.local_run pipeline --rebuild-manifests
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from xnerf.datasets.build_dataset import build_manifest
from xnerf.deployment.export_checkpoint import export_checkpoint
from xnerf.evaluation.test_after_training import run_test
from xnerf.training.train import run_training, run_validation
from xnerf.utils.config import load_config
from xnerf.zero_shot.build_prototypes import build_family_prototypes
from xnerf.zero_shot.evaluate_zero_shot import evaluate_zero_shot

_DEFAULT_CONFIG = "config.yaml"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def _manifests_exist(*paths: Path) -> bool:
    return all(p.exists() for p in paths)


def _resolve_paths(cfg: dict) -> dict[str, Path]:
    data_root = Path(cfg["data"]["root"])
    checkpoint_dir = Path(cfg["training"]["checkpoint_dir"])
    runs_dir = Path("runs")
    zero_shot_dir = runs_dir / "zero_shot"
    final_dir = runs_dir / "output"

    export_cfg  = cfg.get("export", {})
    outputs_cfg = cfg.get("outputs", {})

    return {
        "data_root":      data_root,
        "full_manifest":  Path(cfg["data"].get("full_manifest",  data_root / "processed" / "manifest.jsonl")),
        "train_manifest": Path(cfg["data"].get("train_manifest", data_root / "processed" / "train_manifest.jsonl")),
        "val_manifest":   Path(cfg["data"].get("val_manifest",   data_root / "processed" / "val_manifest.jsonl")),
        "test_manifest":  Path(cfg["data"].get("test_manifest",  data_root / "processed" / "test_manifest.jsonl")),
        "checkpoint_dir": checkpoint_dir,
        "checkpoint":     Path(export_cfg.get("checkpoint", checkpoint_dir / "best.pt")),
        "export_output":  Path(export_cfg.get("output",     "models/xnerf_local_inference.pt")),
        "train_metrics":  Path(outputs_cfg.get("train_metrics", str(runs_dir / "train_metrics.json"))),
        "test_dir":       Path(outputs_cfg.get("test_dir",      str(runs_dir / "test"))),
        "zero_shot_dir":  Path(outputs_cfg.get("zero_shot_dir", str(zero_shot_dir))),
        "prototype_bank": Path(outputs_cfg.get("prototype_bank", str(zero_shot_dir / "prototypes.pt"))),
        "final_dir":      Path(outputs_cfg.get("final_dir",     str(final_dir))),
        "train_done":     checkpoint_dir / "train_done.json",
    }


def _require_checkpoint(p: dict[str, Path], override: str | None = None) -> Path:
    ckpt = Path(override) if override else p["checkpoint"]
    if not ckpt.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt}\n"
            "Run the 'train' command first, or pass --checkpoint <path>."
        )
    return ckpt


def _collect_final_files(p: dict[str, Path]) -> dict[str, str]:
    copies = {
        "local_checkpoint": (p["export_output"],           p["final_dir"] / "xnerf_local_inference.pt"),
        "train_metrics":    (p["train_metrics"],           p["final_dir"] / "train_metrics.json"),
        "test_metrics":     (p["test_dir"] / "test_metrics.json",           p["final_dir"] / "test_metrics.json"),
        "zero_shot_metrics":(p["zero_shot_dir"] / "zero_shot_metrics.json", p["final_dir"] / "zero_shot_metrics.json"),
        "prototype_bank":   (p["prototype_bank"],          p["final_dir"] / "prototypes.pt"),
        "confusion_matrix": (p["test_dir"] / "confusion_matrix.png",        p["final_dir"] / "confusion_matrix.png"),
        "test_predictions": (p["test_dir"] / "test_predictions.npz",        p["final_dir"] / "test_predictions.npz"),
        "zero_shot_preds":  (p["zero_shot_dir"] / "zero_shot_predictions.npz",
                             p["final_dir"] / "zero_shot_predictions.npz"),
    }
    result: dict[str, str] = {}
    for key, (src, dst) in copies.items():
        _copy_if_exists(src, dst)
        if dst.exists():
            result[key] = str(dst)
    return result


# ---------------------------------------------------------------------------
# Individual stage functions (split mode)
# ---------------------------------------------------------------------------

def cmd_build_manifest(cfg: dict, p: dict[str, Path], args: argparse.Namespace) -> dict[str, Any]:
    if getattr(args, "rebuild_manifests", False) or not _manifests_exist(
        p["full_manifest"], p["train_manifest"], p["val_manifest"], p["test_manifest"]
    ):
        build_manifest(
            root=p["data_root"],
            out=p["full_manifest"],
            make_splits=True,
            train_ratio=float(cfg.get("splits", {}).get("train_ratio", 0.8)),
            val_ratio=float(cfg.get("splits", {}).get("val_ratio", 0.1)),
            seed=int(cfg.get("seed", 1337)),
            manifest_only=getattr(args, "manifest_only", False),
        )
        result = {"rebuilt": True}
    else:
        result = {"reused": True}

    result.update({
        "full":  str(p["full_manifest"]),
        "train": str(p["train_manifest"]),
        "val":   str(p["val_manifest"]),
        "test":  str(p["test_manifest"]),
    })
    print(json.dumps(result, indent=2))
    return result


def cmd_train(cfg: dict, p: dict[str, Path], args: argparse.Namespace) -> dict[str, Any]:
    if p["train_done"].exists() and not getattr(args, "force", False):
        print(f"[train] Checkpoint already exists at {p['checkpoint']}.")
        print(f"        Delete {p['train_done']} or pass --force to retrain.")
        return _read_json(p["train_done"])

    metrics = run_training(args.config)

    sentinel = {"status": "done", "checkpoint": str(p["checkpoint"]), **metrics}
    _write_json(p["train_done"], sentinel)
    _write_json(p["train_metrics"], metrics)

    print(json.dumps(sentinel, indent=2))
    return sentinel


def cmd_validate(cfg: dict, p: dict[str, Path], args: argparse.Namespace) -> dict[str, Any]:
    ckpt = _require_checkpoint(p, getattr(args, "checkpoint", None))
    metrics = run_validation(args.config, checkpoint_path=str(ckpt))
    _write_json(p["final_dir"] / "val_metrics.json", metrics)
    print(json.dumps(metrics, indent=2))
    return metrics


def cmd_test(cfg: dict, p: dict[str, Path], args: argparse.Namespace) -> dict[str, Any]:
    ckpt = _require_checkpoint(p, getattr(args, "checkpoint", None))
    metrics = run_test(config_path=args.config, checkpoint_path=str(ckpt), out_dir=p["test_dir"])
    _copy_if_exists(p["test_dir"] / "test_metrics.json",    p["final_dir"] / "test_metrics.json")
    _copy_if_exists(p["test_dir"] / "confusion_matrix.png", p["final_dir"] / "confusion_matrix.png")
    print(json.dumps(metrics, indent=2))
    return metrics


def cmd_zero_shot(cfg: dict, p: dict[str, Path], args: argparse.Namespace) -> dict[str, Any]:
    ckpt = _require_checkpoint(p, getattr(args, "checkpoint", None))
    build_family_prototypes(
        config_path=args.config,
        checkpoint_path=str(ckpt),
        manifest_path=str(p["train_manifest"]),
        output_path=str(p["prototype_bank"]),
    )
    metrics = evaluate_zero_shot(
        config_path=args.config,
        checkpoint_path=str(ckpt),
        manifest_path=str(p["test_manifest"]),
        prototype_path=str(p["prototype_bank"]),
        out_dir=p["zero_shot_dir"],
    )
    _copy_if_exists(p["zero_shot_dir"] / "zero_shot_metrics.json", p["final_dir"] / "zero_shot_metrics.json")
    _copy_if_exists(p["prototype_bank"], p["final_dir"] / "prototypes.pt")
    print(json.dumps(metrics, indent=2))
    return metrics


def cmd_export(cfg: dict, p: dict[str, Path], args: argparse.Namespace) -> dict[str, Any]:
    ckpt = _require_checkpoint(p, getattr(args, "checkpoint", None))
    exported = export_checkpoint(ckpt, Path(args.config), p["export_output"])
    result = {"exported": str(exported)}
    print(json.dumps(result, indent=2))
    return result


# ---------------------------------------------------------------------------
# Monolithic pipeline
# ---------------------------------------------------------------------------

def run_local_pipeline(
    config_path: str = _DEFAULT_CONFIG,
    rebuild_manifests: bool = False,
) -> dict[str, Any]:
    """Run the complete local workflow in one call.

    Steps:
        1. Build full/train/val/test manifests (skipped if already present).
        2. Train and checkpoint best.pt.
        3. Validate on val split.
        4. Evaluate the held-out test split.
        5. Build and evaluate zero-shot prototype bank.
        6. Export local inference checkpoint.
        7. Write combined summary JSON.
    """
    cfg = load_config(config_path)
    p = _resolve_paths(cfg)
    p["final_dir"].mkdir(parents=True, exist_ok=True)

    class _Args:
        config = config_path
        rebuild_manifests = False
        force = False
        checkpoint = None

    fake_args = _Args()
    fake_args.rebuild_manifests = rebuild_manifests

    summary: dict[str, Any] = {"config": config_path, "steps": {}}
    summary["steps"]["build_manifest"] = cmd_build_manifest(cfg, p, fake_args)
    summary["steps"]["train"]          = cmd_train(cfg, p, fake_args)
    summary["steps"]["validate"]       = cmd_validate(cfg, p, fake_args)
    summary["steps"]["test"]           = cmd_test(cfg, p, fake_args)
    summary["steps"]["zero_shot"]      = cmd_zero_shot(cfg, p, fake_args)
    summary["steps"]["export"]         = cmd_export(cfg, p, fake_args)

    test_metrics      = summary["steps"]["test"]
    zero_shot_metrics = summary["steps"]["zero_shot"]
    summary["metrics"] = {
        "accuracy":                    test_metrics.get("accuracy"),
        "precision":                   test_metrics.get("precision"),
        "recall":                      test_metrics.get("recall"),
        "f1":                          test_metrics.get("f1"),
        "roc_auc":                     test_metrics.get("roc_auc"),
        "architecture_malware_accuracy": test_metrics.get("architecture_malware_accuracy"),
        "cross_architecture_accuracy": test_metrics.get("cross_architecture_accuracy"),
        "per_architecture_accuracy":   test_metrics.get("per_architecture_accuracy"),
        "zero_shot_accuracy":          zero_shot_metrics.get("zero_shot_accuracy"),
        "zero_shot_f1":                zero_shot_metrics.get("zero_shot_f1"),
    }

    summary["final_output_dir"] = str(p["final_dir"])
    summary["final_files"] = _collect_final_files(p)
    _write_json(p["final_dir"] / "summary.json", summary)
    print(json.dumps(summary, indent=2))
    return summary


def cmd_pipeline(cfg: dict, p: dict[str, Path], args: argparse.Namespace) -> dict[str, Any]:
    return run_local_pipeline(
        config_path=args.config,
        rebuild_manifests=getattr(args, "rebuild_manifests", False),
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _add_common(sub: argparse.ArgumentParser) -> None:
    sub.add_argument("--config", default=_DEFAULT_CONFIG)


def _add_ckpt(sub: argparse.ArgumentParser) -> None:
    sub.add_argument("--checkpoint", default=None, help="Override checkpoint path")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="X-NERF++ local pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    subs = parser.add_subparsers(dest="command", required=True)

    # --- monolithic --------------------------------------------------------
    p_pipe = subs.add_parser("pipeline", help="Run all stages end-to-end")
    _add_common(p_pipe)
    p_pipe.add_argument("--rebuild-manifests", action="store_true",
                        help="Force manifest rebuild even if split files already exist")

    # --- split stages ------------------------------------------------------
    p_bm = subs.add_parser("build-manifest", help="Build train/val/test manifests")
    _add_common(p_bm)
    p_bm.add_argument("--rebuild-manifests", action="store_true")
    p_bm.add_argument(
        "--manifest-only",
        action="store_true",
        help="Build manifests and metadata without generating tensor cache files",
    )

    p_tr = subs.add_parser("train", help="Train the model")
    _add_common(p_tr)
    p_tr.add_argument("--force", action="store_true",
                      help="Retrain even if train_done.json already exists")

    p_va = subs.add_parser("validate", help="Validate on the val split")
    _add_common(p_va)
    _add_ckpt(p_va)

    p_te = subs.add_parser("test", help="Evaluate on the test split")
    _add_common(p_te)
    _add_ckpt(p_te)

    p_zs = subs.add_parser("zero-shot", help="Build prototypes and evaluate zero-shot")
    _add_common(p_zs)
    _add_ckpt(p_zs)

    p_ex = subs.add_parser("export", help="Export local-inference checkpoint")
    _add_common(p_ex)
    _add_ckpt(p_ex)

    args = parser.parse_args()
    cfg = load_config(args.config)
    p = _resolve_paths(cfg)
    p["final_dir"].mkdir(parents=True, exist_ok=True)

    dispatch = {
        "pipeline":       cmd_pipeline,
        "build-manifest": cmd_build_manifest,
        "train":          cmd_train,
        "validate":       cmd_validate,
        "test":           cmd_test,
        "zero-shot":      cmd_zero_shot,
        "export":         cmd_export,
    }
    dispatch[args.command](cfg, p, args)


if __name__ == "__main__":
    main()
