from __future__ import annotations

import hashlib
from importlib.resources import path
import json
import re
from pathlib import Path
from typing import Any

import torch

from xnerf.datasets.family_cleaning import build_family_vocabulary, family_placeholder_reason, load_family_normalization_rules, normalize_family_name
from xnerf.preprocessing.ontology import ARCH_TO_ID
from xnerf.preprocessing.static_features import binary_image_from_file, load_edgelist_graph
from xnerf.utils.base import DatasetLoader
from xnerf.utils.io import read_jsonl, sha256_file


def _cache_root_for_source(path: Path) -> Path | None:
    for parent in path.parents:
        if parent.name == "raw":
            return parent.parent / "cache" / "isr"
    return None


def _safe_cache_name(value: str, fallback: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return (cleaned or fallback)[:80]


def load_family_vocab(
    manifest_path: str | Path,
    rows: list[dict[str, Any]] | None = None,
    family_vocab_path: str | Path | None = None,
    family_rules_path: str | Path | None = None,
) -> tuple[list[str], dict[str, int]]:
    manifest_path = Path(manifest_path)
    rules = load_family_normalization_rules(family_rules_path)
    candidates = [
        Path(family_vocab_path) if family_vocab_path else None,
        manifest_path.parent / "family_vocab.json",
        manifest_path.with_name("family_vocab.json"),
    ]
    for candidate in candidates:
        if candidate is None:
            continue
        if not candidate.exists():
            continue
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except Exception:
            continue
        family_names = payload.get("family_names") or payload.get("id_to_family") or payload.get("families")
        if isinstance(family_names, list) and family_names:
            names = [normalize_family_name(name, rules=rules) for name in family_names]
            family_to_id = {name: idx for idx, name in enumerate(names)}
            return names, family_to_id
    if rows is None:
        try:
            rows = read_jsonl(manifest_path)
        except Exception:
            rows = []
    names = build_family_vocabulary(rows, rules=rules)
    family_to_id = {name: idx for idx, name in enumerate(names)}
    return names, family_to_id


class MalwareManifestDataset(DatasetLoader):
    """JSONL-backed unified malware dataset.

    Inputs:
        manifest_path with fields path, label, family, arch, optional *_path tensors.
    Outputs:
        dict with tensors: binary_image [1,H,W], api_ids [T], memory_trace [T,C],
        network_ids [T], isr [T,4], arch_id [], label [].
    Tensor dimensions:
        binary image [1,256,256], api/network [256], memory [512,8], isr [1024,4].
    Usage:
        ds = MalwareManifestDataset("data/processed/manifest.jsonl")
        item = ds[0]
    """

    def __init__(
        self,
        manifest_path: str | Path,
        family_vocab_path: str | Path | None = None,
        family_rules_path: str | Path | None = None,
        image_size: int = 256,
        seq_len: int = 256,
        memory_len: int = 512,
        isr_len: int = 1024,
        require_cache: bool = False,
    ):
        self.manifest_path = Path(manifest_path)
        self.rows = read_jsonl(self.manifest_path)
        self.family_rules = load_family_normalization_rules(family_rules_path)
        self.family_names, self.family_to_id = load_family_vocab(
            self.manifest_path,
            self.rows,
            family_vocab_path=family_vocab_path,
            family_rules_path=family_rules_path,
        )
        self.id_to_family = {idx: name for name, idx in self.family_to_id.items()}
        self.family_rules_path = Path(family_rules_path) if family_rules_path else None
        self.image_size = image_size
        self.seq_len = seq_len
        self.memory_len = memory_len
        self.isr_len = isr_len
        self.require_cache = require_cache
        self._source_hash_cache: dict[str, str] = {}

    def __len__(self) -> int:
        return len(self.rows)

    def _binary_image(self, path: Path) -> torch.Tensor:
        try:
            return binary_image_from_file(path, image_size=self.image_size)
        except Exception as e:
            print("\n========== BINARY IMAGE LOAD FAILED ==========")
            print("Path      :", repr(path))
            print("Exists    :", path.exists())
            print("Is file   :", path.is_file())
            print("Absolute  :", path.resolve())
            print("Exception :", repr(e))
            print("==============================================")
            raise
   
    def _load_graph(self, path):
        return load_edgelist_graph(path)
    
    
    def _source_file_key(self, path: Path) -> str:
        key = str(path)
        if key not in self._source_hash_cache:
            self._source_hash_cache[key] = sha256_file(path)[:12]
        return self._source_hash_cache[key]

    def _derived_feature_path(self, row: dict[str, Any]) -> Path | None:
        path = Path(row["path"])
        cache_root = _cache_root_for_source(path)
        if cache_root is None:
            return None
        row_index = int(row.get("row_index", 0))
        sample_id = str(row.get("sample_id") or f"{path.stem}_{row_index}")
        file_key = self._source_file_key(path)
        source = _safe_cache_name(path.stem, "features")[:32]
        sample_key = hashlib.sha256(sample_id.encode("utf-8", errors="ignore")).hexdigest()[:12]
        shard = f"{file_key}_{row_index // 10_000:05d}"
        return cache_root / "features" / file_key[:2] / shard / f"{source}_{row_index}_{sample_key}.pt"

    def _derived_isr_path(self, row: dict[str, Any]) -> Path | None:
        path = Path(row["path"])
        cache_root = _cache_root_for_source(path)
        if cache_root is None:
            return None
        sha = row.get("sha256")
        if not sha:
            sha = sha256_file(path)
        return cache_root / f"{sha}.pt"

    def _memory_trace(self, row: dict[str, Any]) -> torch.Tensor:
        out = torch.zeros(self.memory_len, 8, dtype=torch.float32)
        feature_path = row.get("feature_path")
        if not feature_path and row.get("data_type") in {"feature_csv", "feature_parquet"}:
            derived = self._derived_feature_path(row)
            if derived and derived.exists():
                feature_path = str(derived)
        if row.get("data_type") in {"feature_csv", "feature_parquet"} and self.require_cache:
            if not feature_path:
                raise FileNotFoundError(
                    f"Manifest row requires a feature tensor cache but feature_path is empty: "
                    f"manifest={self.manifest_path} path={row.get('path')} row_index={row.get('row_index')}. "
                    "Run `python -m xnerf.datasets.build_dataset --root data "
                    f"--generate-cache-from-manifest {self.manifest_path}` first."
                )
            if not Path(feature_path).exists():
                raise FileNotFoundError(
                    f"Feature tensor cache is missing: {feature_path} "
                    f"(manifest={self.manifest_path}, path={row.get('path')}, row_index={row.get('row_index')})."
                )
        if feature_path and Path(feature_path).exists():
            loaded = torch.load(feature_path, map_location="cpu").float()
            loaded = torch.nan_to_num(loaded, nan=0.0, posinf=0.0, neginf=0.0)
            if loaded.dim() == 1:
                loaded = torch.nn.functional.pad(loaded, (0, max(0, self.memory_len * 8 - loaded.numel())))[: self.memory_len * 8].view(self.memory_len, 8)
            out[: min(self.memory_len, loaded.shape[0]), : min(8, loaded.shape[1])] = loaded[: self.memory_len, :8]
        return out

    def _load_ids(self, row: dict[str, Any], key: str) -> torch.Tensor:
        values = row.get(key, [])
        out = torch.zeros(self.seq_len, dtype=torch.long)
        if values:
            vals = torch.tensor(values[: self.seq_len], dtype=torch.long)
            out[: vals.numel()] = vals
        return out

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        path = Path(row["path"])
        
        graph_x = torch.zeros((0, 4), dtype=torch.float32)
        graph_edge_index = torch.zeros((2, 0), dtype=torch.long)

        if str(path).lower().endswith(".edgelist"):
            graph_x, graph_edge_index = self._load_graph(path)


        raw_family = row.get("family", "unknown")
        label = int(row.get("label", 0))
        family = normalize_family_name(raw_family, label=label, rules=self.family_rules)
        invalid_family = label == 1 and family_placeholder_reason(raw_family, rules=self.family_rules) is not None
        if not invalid_family and family not in self.family_to_id:
            raise KeyError(f"family '{family}' missing from vocabulary for {self.manifest_path}")
        isr = torch.zeros(self.isr_len, 4, dtype=torch.long)
        isr_path = row.get("isr_path")
        if self.require_cache:
            suffix = path.suffix.lower()
            likely_binary = row.get("data_type") not in {"feature_csv", "feature_parquet", "api_sequence_csv", "api_sequence_txt"} and suffix in {".bin", ".exe", ".dll", ".so", ".elf", ""}
            unknown_arch = str(row.get("arch", "unknown")).strip().lower() == "unknown"
            likely_binary = likely_binary and not unknown_arch
            if likely_binary and not isr_path:
                derived = self._derived_isr_path(row)
                if derived and derived.exists():
                    isr_path = str(derived)
            if likely_binary and not isr_path:
                raise FileNotFoundError(
                    f"Manifest row requires an ISR tensor cache but isr_path is empty: "
                    f"manifest={self.manifest_path} path={row.get('path')}. "
                    "Run `python -m xnerf.datasets.build_dataset --root data "
                    f"--generate-cache-from-manifest {self.manifest_path}` first."
                )
            if isr_path and not Path(isr_path).exists():
                raise FileNotFoundError(
                    f"ISR tensor cache is missing: {isr_path} "
                    f"(manifest={self.manifest_path}, path={row.get('path')})."
                )
        if isr_path and Path(isr_path).exists():
            loaded = torch.load(isr_path, map_location="cpu")
            if loaded.is_floating_point():
                loaded = torch.nan_to_num(loaded, nan=0.0, posinf=0.0, neginf=0.0)
            isr[: min(self.isr_len, loaded.shape[0])] = loaded[: self.isr_len]
        data_type = row.get("data_type")
        return {
            "binary_image": torch.zeros(1, self.image_size, self.image_size, dtype=torch.float32)
            if data_type in {"feature_csv", "feature_parquet"}
            else self._binary_image(path),
            "graph_x": graph_x,
            "graph_edge_index": graph_edge_index,
            "api_ids": self._load_ids(row, "api_ids"),
            "network_ids": self._load_ids(row, "network_ids"),
            "memory_trace": self._memory_trace(row),
            "isr": isr,
            "arch_id": torch.tensor(ARCH_TO_ID.get(str(row.get("arch", "unknown")).strip().lower(), ARCH_TO_ID["unknown"]), dtype=torch.long),
            "label": torch.tensor(label, dtype=torch.long),
            "family_label": torch.tensor(-1 if invalid_family else self.family_to_id[family], dtype=torch.long),
            "dataset": row.get("dataset", "unknown"),
            "family": row.get("family", "unknown"),
            "path": row.get("path", ""),
            "row_index": row.get("row_index", index),
            "sample_id": row.get("sample_id", ""),
            "sha256": row.get("sha256", ""),
        }


class UnifiedMalwareDataset(MalwareManifestDataset):
    """Alias for the production multimodal loader."""
