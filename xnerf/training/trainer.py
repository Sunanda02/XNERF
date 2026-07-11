from __future__ import annotations

import gc
from pathlib import Path
from typing import Any
from uuid import uuid4

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import traceback


from xnerf.datasets.validation import validate_family_batch
from xnerf.training.losses import classification_losses, total_loss
from xnerf.utils.base import Trainer, move_to_device
from xnerf.utils.base import collate_dicts

class XNerfTrainer(Trainer):
    """Production training loop.

    Inputs:
        model, train/val DatasetLoader, optimizer config.
    Outputs:
        checkpoint files and metrics dict.
    Tensor dimensions:
        consumes batch tensors documented by MalwareManifestDataset.
    Usage:
        trainer = XNerfTrainer(model, train_ds, val_ds)
        trainer.fit()
    """

    def __init__(
        self,
        model: torch.nn.Module,
        train_dataset,
        val_dataset=None,
        batch_size: int = 8,
        lr: float = 3e-4,
        epochs: int = 10,
        grad_accum: int = 1,
        num_workers: int = 2,
        checkpoint_dir: str | Path = "checkpoints",
        patience: int = 3,
        device: str | None = None,
        resume_from: str | Path | None = None,
        grad_clip: float = 1.0,
        use_amp: bool = True,
        debug_max_batches: int | None = None,
        family_names: list[str] | None = None,
    ):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = model.to(self.device)
        if torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model)
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_dicts)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_dicts) if val_dataset else None
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        self.use_amp = bool(use_amp) and self.device.type == "cuda"
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        self.epochs = epochs
        self.grad_accum = grad_accum
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.patience = patience
        self.grad_clip = grad_clip
        self.batch_size = batch_size
        self.lr = lr
        self.debug_max_batches = int(debug_max_batches) if debug_max_batches else None
        self.family_names = list(family_names or getattr(train_dataset, "family_names", []))
        self.start_epoch = 1
        self.best_val_loss = float("inf")
        self.bad_epochs = 0
        if resume_from:
            self._load_resume_checkpoint(Path(resume_from))
        self._validate_family_labels()
        self._print_startup_diagnostics(train_dataset, val_dataset, num_workers)

    def _validate_family_labels(self) -> None:
        sample_batch = next(iter(self.train_loader))
        validate_family_batch(sample_batch)
        if not torch.is_tensor(sample_batch["family_label"]):
            raise RuntimeError("training batch family_label must be a tensor")

    def _load_resume_checkpoint(self, checkpoint_path: Path) -> None:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"resume checkpoint not found: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model"])
        if "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        if "scaler" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler"])
        epoch = int(checkpoint.get("epoch", 0))
        self.start_epoch = epoch + 1
        self.best_val_loss = float(checkpoint.get("best_val_loss", checkpoint.get("val_loss", float("inf"))))
        self.bad_epochs = int(checkpoint.get("bad_epochs", 0))
        del checkpoint
        gc.collect()
        print(f"Resuming training from {checkpoint_path} at epoch {self.start_epoch}")

    def _save_checkpoint(self, path: Path, epoch: int, val_loss: float, best_val_loss: float, bad_epochs: int) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scaler": self.scaler.state_dict(),
                "epoch": epoch,
                "val_loss": val_loss,
                "best_val_loss": best_val_loss,
                "bad_epochs": bad_epochs,
                "family_names": self.family_names,
            },
            tmp_path,
        )
        try:
            tmp_path.replace(path)
        except PermissionError:
            fallback_path = path.with_name(f"{path.stem}_epoch_{epoch}_{uuid4().hex[:8]}{path.suffix}")
            tmp_path.replace(fallback_path)
            print(
                f"warning: could not replace locked checkpoint {path}; saved {fallback_path} instead",
                flush=True,
            )

    def _diagnostic_value(self, value: Any, limit: int = 8) -> Any:
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().flatten()[:limit].tolist()
        if isinstance(value, (list, tuple)):
            return list(value[:limit])
        return value

    def _batch_diagnostics(self, batch: dict[str, Any], batch_idx: int | None) -> dict[str, Any]:
        keys = ("dataset", "label", "arch_id", "family_label", "family", "sha256", "sample_id", "row_index", "path")
        diagnostics: dict[str, Any] = {"batch_idx": batch_idx}
        for key in keys:
            if key in batch:
                diagnostics[key] = self._diagnostic_value(batch[key])
        return diagnostics

    def _raise_nonfinite(self, message: str, batch: dict[str, Any], batch_idx: int | None) -> None:
        raise RuntimeError(f"{message}; batch={self._batch_diagnostics(batch, batch_idx)}")

    def _tensor_stats(self, value: torch.Tensor) -> dict[str, Any]:
        detached = value.detach()
        finite = detached[torch.isfinite(detached)]
        stats: dict[str, Any] = {
            "shape": tuple(detached.shape),
            "dtype": str(detached.dtype),
            "device": str(detached.device),
            "nonfinite_count": int((~torch.isfinite(detached)).sum().item()),
        }
        if finite.numel():
            finite_cpu = finite.float().cpu()
            stats.update(
                {
                    "min": float(finite_cpu.min()),
                    "max": float(finite_cpu.max()),
                    "mean": float(finite_cpu.mean()),
                }
            )
        return stats

    def _check_finite_tensor(self, name: str, value: torch.Tensor, batch: dict[str, Any], batch_idx: int | None, kind: str) -> None:
        if not torch.isfinite(value).all():
            self._raise_nonfinite(f"non-finite {kind} detected: {name}, stats={self._tensor_stats(value)}", batch, batch_idx)

    def _print_startup_diagnostics(self, train_dataset, val_dataset, num_workers: int) -> None:
        train_manifest = getattr(train_dataset, "manifest_path", "unknown")
        val_manifest = getattr(val_dataset, "manifest_path", None) if val_dataset is not None else None
        print(
            "[train-start] "
            f"device={self.device} "
            f"batch_size={self.batch_size} "
            f"lr={self.lr} "
            f"epochs={self.epochs} "
            f"grad_accum={self.grad_accum} "
            f"grad_clip={self.grad_clip} "
            f"use_amp={self.use_amp} "
            f"grad_scaler={self.scaler.is_enabled()} "
            f"debug_max_batches={self.debug_max_batches} "
            f"num_workers={num_workers} "
            f"train_manifest={train_manifest} "
            f"train_size={len(train_dataset)} "
            f"val_manifest={val_manifest} "
            f"val_size={len(val_dataset) if val_dataset is not None else 0}",
            flush=True,
        )

    def _check_gradients(self, batch: dict[str, Any], batch_idx: int | None) -> None:
        for name, param in self.model.named_parameters():
            if param.grad is not None and not torch.isfinite(param.grad).all():
                self._raise_nonfinite(
                    f"non-finite gradient detected: {name}, stats={self._tensor_stats(param.grad)}",
                    batch,
                    batch_idx,
                )

    def _check_parameters(self, batch: dict[str, Any], batch_idx: int | None) -> None:
        for name, param in self.model.named_parameters():
            if not torch.isfinite(param).all():
                self._raise_nonfinite(
                    f"non-finite parameter detected after optimizer step: {name}, stats={self._tensor_stats(param)}",
                    batch,
                    batch_idx,
                )

    def _check_optimizer_state(self, batch: dict[str, Any], batch_idx: int | None) -> None:
        for param_idx, state in enumerate(self.optimizer.state.values()):
            for name, value in state.items():
                if torch.is_tensor(value) and not torch.isfinite(value).all():
                    self._raise_nonfinite(
                        f"non-finite optimizer state detected after optimizer step: param_index={param_idx} state={name}, stats={self._tensor_stats(value)}",
                        batch,
                        batch_idx,
                    )

    def _optimizer_step(self, batch: dict[str, Any], batch_idx: int | None) -> None:
        self.scaler.unscale_(self.optimizer)
        self._check_gradients(batch, batch_idx)
        if self.grad_clip and self.grad_clip > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            if torch.is_tensor(grad_norm) and not torch.isfinite(grad_norm):
                self._raise_nonfinite(
                    f"non-finite gradient norm detected before optimizer step: {float(grad_norm.detach().cpu())}",
                    batch,
                    batch_idx,
                )
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self._check_parameters(batch, batch_idx)
        self._check_optimizer_state(batch, batch_idx)
        self.optimizer.zero_grad(set_to_none=True)

    def _step(self, batch: dict[str, Any], train: bool, batch_idx: int | None = None) -> float:
        batch = move_to_device(batch, self.device)
        with torch.set_grad_enabled(train), torch.cuda.amp.autocast(enabled=self.use_amp):
            outputs = self.model(batch)
            for name, value in outputs.items():
                if torch.is_tensor(value):
                    self._check_finite_tensor(name, value, batch, batch_idx, "model output")
            losses = classification_losses(outputs, batch)
            if "family_label" in batch and "family_ce" not in losses:
                self._raise_nonfinite("family_label present but family_ce loss was not computed", batch, batch_idx)
            for name, value in losses.items():
                if torch.is_tensor(value):
                    self._check_finite_tensor(name, value, batch, batch_idx, "loss term")
            loss = total_loss(losses)
            if batch_idx > 9180:
              print(batch_idx, {k: float(v.detach().cpu()) for k, v in losses.items()})
        if not torch.isfinite(loss):
            details = {
                name: float(value.detach().cpu())
                for name, value in losses.items()
                if value.numel() == 1
            }
            self._raise_nonfinite(
                f"non-finite loss detected: total={float(loss.detach().cpu())}, terms={details}",
                batch,
                batch_idx,
            )
        if batch_idx > 9180:
               print("TOTAL LOSS =", float(loss.detach().cpu()))
        if train:
            try:
               self.scaler.scale(loss / self.grad_accum).backward()
            except Exception as e:
                  print("\nFAILED AT TRAIN BATCH:", batch_idx)
                  print("LOSS =", float(loss.detach().cpu()))
                  print("LOSSES =", {k: float(v.detach().cpu()) for k, v in losses.items()})
                  print(self._batch_diagnostics(batch, batch_idx))
                  traceback.print_exc()
                  raise
        return float(loss.detach().cpu())

    def fit(self) -> dict[str, Any]:
        best = self.best_val_loss
        bad_epochs = self.bad_epochs
        history = []
        if self.start_epoch > self.epochs:
            return {"best_val_loss": best, "history": history, "resumed_from_epoch": self.start_epoch - 1}
        for epoch in range(self.start_epoch, self.epochs + 1):
            self.model.train()
            self.optimizer.zero_grad(set_to_none=True)
            train_loss = 0.0
            train_batches = 0
            last_train_batch = None
            for i, batch in enumerate(tqdm(self.train_loader, desc=f"epoch {epoch} train")):
                if self.debug_max_batches is not None and i >= self.debug_max_batches:
                    break
                train_loss += self._step(batch, train=True, batch_idx=i)
                train_batches += 1
                last_train_batch = batch
                if (i + 1) % self.grad_accum == 0:
                    self._optimizer_step(batch, i)
            if train_batches and train_batches % self.grad_accum != 0:
                self._optimizer_step(last_train_batch, train_batches - 1)
            val_loss = self.validate() if self.val_loader else train_loss / max(1, train_batches)
            history.append({"epoch": epoch, "train_loss": train_loss / max(1, train_batches), "val_loss": val_loss})
            if val_loss < best:
                best = val_loss
                bad_epochs = 0
                self._save_checkpoint(self.checkpoint_dir / "best.pt", epoch, val_loss, best, bad_epochs)
            else:
                bad_epochs += 1
                if bad_epochs >= self.patience:
                    self._save_checkpoint(self.checkpoint_dir / "last.pt", epoch, val_loss, best, bad_epochs)
                    print(
                        f"epoch {epoch}: "
                        f"train_loss={history[-1]['train_loss']:.4f} "
                        f"val_loss={val_loss:.4f} "
                        f"best_val_loss={best:.4f}",
                        flush=True,
                    )
                    break
            print(
                f"epoch {epoch}: "
                f"train_loss={history[-1]['train_loss']:.4f} "
                f"val_loss={val_loss:.4f} "
                f"best_val_loss={best:.4f}",
                flush=True,
            )
            self._save_checkpoint(self.checkpoint_dir / "last.pt", epoch, val_loss, best, bad_epochs)
        return {"best_val_loss": best, "history": history}

    @torch.no_grad()
    def validate(self) -> float:
        self.model.eval()
        total = 0.0
        batches = 0
        for i, batch in enumerate(tqdm(self.val_loader, desc="validate")):
            if self.debug_max_batches is not None and i >= self.debug_max_batches:
                break
            total += self._step(batch, train=False, batch_idx=i)
            batches += 1
        return total / max(1, batches)

