from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import re
from pathlib import Path
from typing import Callable
from collections import defaultdict

from xnerf.datasets.family_cleaning import build_family_vocabulary, load_family_normalization_rules
from xnerf.datasets.family_cleaning import normalize_family_name
from xnerf.preprocessing.pipeline import ArchitectureNormalizationPipeline
from xnerf.preprocessing.static_features import build_memory_trace as shared_build_memory_trace
from xnerf.sandbox.cape_parser import parse_cape_report
from xnerf.utils.io import read_jsonl, sha256_file, write_jsonl
from xnerf.utils.tokenization import tokens_to_ids

# ------------------------------------------------------------------
# CIC-YNU IoTMal sampling
# ------------------------------------------------------------------

IOTMAL_FAMILY_SAMPLE_PROBS = {
    "Mirai": 0.002,
    "Benign": 0.01,
    "DarkNexus": 0.10,
    "Unknown": 0.10,
    "Gafgyt": 0.20,
    "Generic": 0.40,
}

IOTMAL_RANDOM_SEED = 1337
random.seed(IOTMAL_RANDOM_SEED)

def infer_arch(path: Path) -> str:
    text = str(path).lower()
    for arch in ("arm64", "arm","mipsel", "mips", "riscv", "x64", "x86"):
        if arch in text:
            return arch
    return "unknown"

def should_keep_iotmal_sample(family: str) -> bool:
    family = str(family).strip()

    keep_prob = IOTMAL_FAMILY_SAMPLE_PROBS.get(
        family,
        1.0,  # keep all rare families
    )

    return random.random() <= keep_prob

def infer_label(path: Path) -> int:
    text = str(path).lower()
    if any(token in text for token in ("benign", "goodware", "clean")):
        return 0
    return 1


def parse_label_value(value: str | int | float | None, default: int = 1) -> int:
    if value is None:
        return default
    text = str(value).strip().lower()
    if text in {"0", "benign", "goodware", "clean", "false", "normal"}:
        return 0
    if text in {"1", "malware", "malicious", "true", "infected"}:
        return 1
    return default


def enrich_dynamic_report(record: dict, path: Path) -> dict:
    if path.suffix.lower() != ".json":
        return record
    parts = {part.lower() for part in path.parts}
    if not ({"cape", "avast", "andmal2020", "cicmaldroid2020"} & parts):
        return record
    try:
        parsed = parse_cape_report(path)
    except Exception as exc:
        record["parse_error"] = f"{type(exc).__name__}: {exc}"
        return record
    record["api_ids"] = tokens_to_ids(parsed["api_calls"], vocab_size=8192, max_len=256, prefix="api")
    record["network_ids"] = tokens_to_ids(parsed["network_events"], vocab_size=4096, max_len=256, prefix="net")
    record["memory_event_count"] = len(parsed.get("memory_events", []))
    record["process_event_count"] = len(parsed.get("process_events", []))
    record["api_call_count"] = len(parsed.get("api_calls", []))
    record["network_event_count"] = len(parsed.get("network_events", []))
    score = parsed.get("summary", {}).get("score")
    if score is not None:
        record["sandbox_score"] = score
    return record


def _safe_float(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _looks_like_header(row: list[str]) -> bool:
    if not row:
        return True
    numeric = sum(_safe_float(cell) is not None for cell in row[1:])
    return numeric < max(1, len(row[1:]) // 2)


def norm_col(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")


ID_COLUMNS = {
    "sha",
    "sha1",
    "sha256",
    "md5",
    "hash",
    "sample",
    "sample_id",
    "sampleid",
    "file",
    "filename",
    "apk",
    "apk_name",
    "id",
}

LABEL_COLUMNS = {"label", "labels", "class", "category", "verdict", "is_malware", "malware", "type"}
FAMILY_COLUMNS = {"family", "malware_family", "class_name", "category_name","malwarefamily","classification_family"}

SPLIT_ALIASES = {
    "train": "train",
    "training": "train",
    "test": "test",
    "val": "val",
    "valid": "val",
    "validation": "val",
}
SPLIT_PATTERN = re.compile(r"(?:^|[^a-z])(train|training|test|val|valid|validation)(?:[^a-z]|$)")


def column_index(headers: list[str], candidates: set[str]) -> int | None:
    normalized = [norm_col(h) for h in headers]
    for i, name in enumerate(normalized):
        if name in candidates:
            return i
    return None


def normalize_split(value: str | None) -> str | None:
    if not value:
        return None
    key = value.strip().lower()
    return SPLIT_ALIASES.get(key)


def top_level_dataset(path: Path, raw: Path) -> str:
    relative = path.relative_to(raw)
    return relative.parts[0] if len(relative.parts) else "unknown"


def print_parse_folder(path: Path, raw: Path, current: Path | None) -> Path:
    folder = path.parent
    if folder != current:
        try:
            display = folder.relative_to(raw)
        except ValueError:
            display = folder
        print(f"Parsing folder: {display}")
    return folder


def should_skip_manifest_path(path: Path) -> bool:
    if path.name.startswith(".extracted_"):
        return True
    return path.suffix.lower() in {".zip", ".7z", ".rar"}


def infer_split_from_path(path: Path) -> str | None:
    for part in (*path.parts, path.name):
        match = SPLIT_PATTERN.search(part.lower())
        if match:
            return SPLIT_ALIASES.get(match.group(1))
    return None


def is_label_map_csv(path: Path, first_row: list[str]) -> bool:
    name = path.name.lower()
    if "public_labels" in name and _looks_like_header(first_row):
        return True
    headers = [norm_col(x) for x in first_row]
    header_set = set(headers)
    metadata = ID_COLUMNS | LABEL_COLUMNS | FAMILY_COLUMNS
    non_metadata = [h for h in headers if h not in metadata]
    return bool(header_set & ID_COLUMNS) and bool(header_set & (LABEL_COLUMNS | FAMILY_COLUMNS)) and len(non_metadata) <= 1


def load_label_maps(raw: Path) -> dict[str, dict]:
    labels: dict[str, dict] = {}
    for path in raw.rglob("*.csv"):
        try:
            with open(path, "r", encoding="utf-8", errors="ignore", newline="") as f:
                reader = csv.reader(f)
                first = next(reader, [])
                if not is_label_map_csv(path, first):
                    continue
                id_idx = column_index(first, ID_COLUMNS)
                label_idx = column_index(first, LABEL_COLUMNS)
                family_idx = column_index(first, FAMILY_COLUMNS)
                if id_idx is None:
                    continue
                for row in reader:
                    if id_idx >= len(row):
                        continue
                    sample_id = row[id_idx].strip()
                    if not sample_id:
                        continue
                    item = labels.setdefault(sample_id, {})
                    if label_idx is not None and label_idx < len(row):
                        item["label"] = parse_label_value(row[label_idx], default=item.get("label", infer_label(path)))
                    if family_idx is not None and family_idx < len(row) and row[family_idx].strip():
                        item["family"] = row[family_idx].strip()
        except OSError:
            continue
    return labels


def collect_family_names(rows: list[dict], family_rules_path: str | Path | None = None) -> list[str]:
    rules = load_family_normalization_rules(family_rules_path)
    return build_family_vocabulary(rows, rules=rules)


def write_family_vocab(out_dir: Path, family_names: list[str], family_rules_path: str | Path | None = None) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    rules = load_family_normalization_rules(family_rules_path)
    family_names = build_family_vocabulary(({"family": name, "label": 1} for name in family_names), rules=rules)
    family_to_id = {name: idx for idx, name in enumerate(family_names)}
    payload = {
        "family_names": family_names,
        "family_to_id": family_to_id,
        "id_to_family": family_names,
    }
    path = out_dir / "family_vocab.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def feature_family_from_path(path: Path) -> str:
    stem = path.stem
    if stem.lower() in {"csv", "features", "feature_vectors_static", "feature_vectors"}:
        return path.parent.name
    return stem


def build_memory_trace(features: list[float], rows: int = 512, cols: int = 8):
    return shared_build_memory_trace(features, rows=rows, cols=cols)


def feature_tensor_path(feature_cache: Path, file_key: str, source_name: str, row_index: int, sample_id: str) -> Path:
    """Return a short, sharded cache path for per-row feature tensors."""

    source = safe_name(source_name, "features")[:32]
    sample_key = hashlib.sha256(sample_id.encode("utf-8", errors="ignore")).hexdigest()[:12]
    shard = f"{file_key}_{row_index // 10_000:05d}"
    folder = feature_cache / file_key[:2] / shard
    folder.mkdir(parents=True, exist_ok=True)
    return folder / f"{source}_{row_index}_{sample_key}.pt"


def safe_name(value: str, fallback: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return (cleaned or fallback)[:80]


def row_features(row: list[str], numeric_indexes: list[int]) -> list[float]:
    features = []
    for i in numeric_indexes:
        if i < len(row):
            value = _safe_float(row[i])
            if value is not None:
                features.append(value)
    return features


def api_sequence_column_indexes(headers: list[str]) -> list[int]:
    normalized = [norm_col(h) for h in headers]
    skip = ID_COLUMNS | LABEL_COLUMNS | FAMILY_COLUMNS | {"split", "subset"}
    indexes = []
    for i, name in enumerate(normalized):
        if name in skip:
            continue
        if re.fullmatch(r"(?:t_)?\d+", name) or re.fullmatch(r"api(?:_call)?_?\d+", name):
            indexes.append(i)
    return indexes


def is_api_sequence_csv(path: Path, first_row: list[str]) -> bool:
    header_names = {norm_col(h) for h in first_row}
    has_metadata = bool(header_names & ID_COLUMNS) and bool(header_names & (LABEL_COLUMNS | {"malware"}))
    indexes = api_sequence_column_indexes(first_row)
    if has_metadata and len(indexes) >= 3:
        return True
    if not _looks_like_header(first_row):
        return False
    text = str(path).lower()
    return "apicallsequence" in text.replace("_", "") and bool(indexes)


def process_api_sequence_csv(
    path: Path,
    raw: Path,
    max_rows_per_csv: int | None = None,
    row_sink: Callable[[dict], None] | None = None,
) -> tuple[list[dict], int]:
    rows: list[dict] = []
    count = 0
    dataset = top_level_dataset(path, raw)
    default_label = infer_label(path)
    default_family = feature_family_from_path(path)
    split_hint = infer_split_from_path(path)
    file_hash = sha256_file(path)
    with open(path, "r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.reader(f)
        headers = next(reader, [])
        if not headers:
            return rows, count
        seq_indexes = api_sequence_column_indexes(headers)
        if not seq_indexes:
            return rows, count
        id_idx = column_index(headers, ID_COLUMNS)
        label_idx = column_index(headers, LABEL_COLUMNS)
        family_idx = column_index(headers, FAMILY_COLUMNS)
        for idx, row in enumerate(reader):
            if max_rows_per_csv is not None and count >= max_rows_per_csv:
                break
            if not row:
                continue
            calls = [row[i].strip() for i in seq_indexes if i < len(row) and row[i].strip()]
            if not calls:
                continue
            sample_id = row[id_idx].strip() if id_idx is not None and id_idx < len(row) else f"{path.stem}_{idx}"
            sample_id = sample_id or f"{path.stem}_{idx}"
            label = default_label
            if label_idx is not None and label_idx < len(row):
                label = parse_label_value(row[label_idx], default=label)
            family = default_family
            if family_idx is not None and family_idx < len(row) and row[family_idx].strip():
                family = row[family_idx].strip()
            record = {
                "path": str(path),
                "row_index": idx,
                "sample_id": sample_id,
                "sha256": sample_id if len(sample_id) >= 16 else f"{file_hash}:{idx}",
                "dataset": dataset,
                "data_type": "api_sequence_csv",
                "api_ids": tokens_to_ids(calls, vocab_size=8192, max_len=256, prefix="api"),
                "api_call_count": len(calls),
                "network_ids": [],
                "network_event_count": 0,
                "label": label,
                "family": family,
                "arch": infer_arch(path),
            }
            if split_hint:
                record["split"] = split_hint
            if row_sink:
                row_sink(record)
            else:
                rows.append(record)
            count += 1
    return rows, count


def load_line_labels(path: Path) -> list[str]:
    if not path.exists():
        return []
    labels = []
    with open(path, "r", encoding="utf-8", errors="ignore", newline="") as f:
        for row in csv.reader(f):
            if row and row[0].strip():
                labels.append(row[0].strip())
    return labels


def process_api_sequence_txt(
    path: Path,
    raw: Path,
    max_rows: int | None = None,
    row_sink: Callable[[dict], None] | None = None,
) -> tuple[list[dict], int]:
    rows: list[dict] = []
    count = 0
    dataset = top_level_dataset(path, raw)
    split_hint = infer_split_from_path(path)
    labels = load_line_labels(path.parent.parent / "labels.csv") or load_line_labels(path.parent / "labels.csv")
    file_hash = sha256_file(path)
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for idx, line in enumerate(f):
            if max_rows is not None and count >= max_rows:
                break
            calls = [token for token in line.strip().split() if token]
            if not calls:
                continue
            family = labels[idx] if idx < len(labels) else path.stem
            sample_id = f"{path.stem}_{idx}"
            record = {
                "path": str(path),
                "row_index": idx,
                "sample_id": sample_id,
                "sha256": f"{file_hash}:{idx}",
                "dataset": dataset,
                "data_type": "api_sequence_txt",
                "api_ids": tokens_to_ids(calls, vocab_size=8192, max_len=256, prefix="api"),
                "api_call_count": len(calls),
                "network_ids": [],
                "network_event_count": 0,
                "label": 1,
                "family": family,
                "arch": infer_arch(path),
            }
            if split_hint:
                record["split"] = split_hint
            if row_sink:
                row_sink(record)
            else:
                rows.append(record)
            count += 1
    return rows, count


def process_feature_csv(
    path: Path,
    raw: Path,
    cache: Path,
    label_maps: dict[str, dict] | None = None,
    max_rows_per_csv: int | None = None,
    row_sink: Callable[[dict], None] | None = None,
    manifest_only: bool = False,
) -> tuple[list[dict], int]:
    """Convert headerless or headered numeric feature-vector CSV rows.

    Headerless expected row shape:
        sample_id, f1, f2, f3, ...

    Headered tables use known id/label/family columns when present and all
    numeric non-label columns as features. Label-map CSVs such as
    public_labels.csv return no training samples.

    Outputs one dataset sample per CSV row. Numeric features are cached as a
    [512,8] tensor consumed by MalwareManifestDataset.memory_trace.
    """

    rows: list[dict] = []
    count = 0
    feature_cache = cache / "features"
    if not manifest_only:
        feature_cache.mkdir(parents=True, exist_ok=True)
    dataset = path.relative_to(raw).parts[0] if len(path.relative_to(raw).parts) else "unknown"
    default_label = infer_label(path)
    default_family = feature_family_from_path(path)
    split_hint = infer_split_from_path(path)
    label_maps = label_maps or {}
    file_key = sha256_file(path)[:12]
    with open(path, "r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.reader(f)
        first = next(reader, [])
        if not first:
            return rows, count
        has_header = _looks_like_header(first)
        if has_header and is_label_map_csv(path, first):
            print(f"Loaded label-map CSV metadata, not training samples: {path}")
            return rows, count
        id_idx = 0
        label_idx = None
        family_idx = None
        numeric_indexes = list(range(1, len(first)))
        if has_header:
            headers = first
            id_idx = column_index(headers, ID_COLUMNS)
            label_idx = column_index(headers, LABEL_COLUMNS)
            family_idx = column_index(headers, FAMILY_COLUMNS)
            normalized = [norm_col(h) for h in headers]
            skip_cols = {idx for idx in (id_idx, label_idx, family_idx) if idx is not None}
            numeric_indexes = [
                i
                for i, name in enumerate(normalized)
                if i not in skip_cols and name not in LABEL_COLUMNS and name not in FAMILY_COLUMNS
            ]
            id_idx = 0 if id_idx is None else id_idx

        def _row_iter():
            if not has_header:
                yield first
            for row in reader:
                yield row

        for idx, row in enumerate(_row_iter()):
            if max_rows_per_csv is not None and len(rows) >= max_rows_per_csv:
                break
            if not row:
                continue
            sample_id = row[id_idx].strip() if id_idx < len(row) else f"{path.stem}_{idx}"
            sample_id = sample_id or f"{path.stem}_{idx}"
            features = row_features(row, numeric_indexes)
            if not features:
                continue
            feature_path = None
            if not manifest_only:
                import torch

                feature_path = feature_tensor_path(feature_cache, file_key, path.stem, idx, sample_id)
                torch.save(build_memory_trace(features), feature_path)
            mapped = label_maps.get(sample_id, {})
            label = mapped.get("label", default_label)
            if label_idx is not None and label_idx < len(row):
                label = parse_label_value(row[label_idx], default=label)
            family = mapped.get("family", default_family)
            if family_idx is not None and family_idx < len(row) and row[family_idx].strip():
                family = row[family_idx].strip()
            row = {
                "path": str(path),
                "row_index": idx,
                "sample_id": sample_id,
                "sha256": sample_id if len(sample_id) >= 16 else f"{sha256_file(path)}:{idx}",
                "dataset": dataset,
                "data_type": "feature_csv",
                "feature_path": str(feature_path) if feature_path else "",
                "feature_dim": len(features),
                "label": label,
                "family": family,
                "arch": infer_arch(path),
            }
            if split_hint:
                row["split"] = split_hint
            if row_sink:
                row_sink(row)
            else:
                rows.append(row)
            count += 1
    return rows, count


def _parquet_table_to_pydict(path: Path) -> dict[str, list]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("Reading parquet requires pyarrow. Install pyarrow to parse parquet datasets.") from exc
    table = pq.read_table(path)
    return table.to_pydict()


def _feature_column_name(columns: list[str]) -> str | None:
    for name in ("features", "feature", "feature_vector", "x"):
        if name in columns:
            return name
    return None


def process_feature_parquet(
    path: Path,
    raw: Path,
    cache: Path,
    label_maps: dict[str, dict] | None = None,
    max_rows_per_parquet: int | None = None,
    row_sink: Callable[[dict], None] | None = None,
    manifest_only: bool = False,
) -> tuple[list[dict], int]:
    """Convert parquet feature tables into manifest rows.

    Supports EMBER-style parquet files with a `features` column or wide numeric
    tables with id/label metadata columns.
    """

    rows: list[dict] = []
    count = 0
    feature_cache = cache / "features"
    if not manifest_only:
        feature_cache.mkdir(parents=True, exist_ok=True)
    dataset = path.relative_to(raw).parts[0] if len(path.relative_to(raw).parts) else "unknown"
    default_label = infer_label(path)
    default_family = feature_family_from_path(path)
    split_hint = infer_split_from_path(path)
    label_maps = label_maps or {}

    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("Reading parquet requires pyarrow. Install pyarrow to parse parquet datasets.") from exc

    parquet = pq.ParquetFile(path)
    columns = parquet.schema.names
    feature_col = _feature_column_name(columns)
    split_col = "split" if "split" in columns else ("subset" if "subset" in columns else None)
    file_key = sha256_file(path)[:12]
    id_col = None
    label_col = None
    family_col = None
    if feature_col is None:
        id_col = column_index(columns, ID_COLUMNS)
        label_col = column_index(columns, LABEL_COLUMNS)
        family_col = column_index(columns, FAMILY_COLUMNS)
        normalized = [norm_col(h) for h in columns]
        skip_cols = {idx for idx in (id_col, label_col, family_col) if idx is not None}
        numeric_indexes = [
            i
            for i, name in enumerate(normalized)
            if i not in skip_cols and name not in LABEL_COLUMNS and name not in FAMILY_COLUMNS
        ]
    else:
        numeric_indexes = []

    max_rows = max_rows_per_parquet or parquet.metadata.num_rows
    row_index = 0
    for batch in parquet.iter_batches(batch_size=2048, columns=columns):
        data = batch.to_pydict()
        batch_rows = batch.num_rows
        for i in range(batch_rows):
            if row_index >= max_rows:
                return rows, count
            sample_id = None
            if id_col is not None:
                sample_id = str(data[columns[id_col]][i]).strip()
            if not sample_id:
                sample_id = str(data.get("sha256", [""] * batch_rows)[i]).strip() if "sha256" in data else f"{path.stem}_{row_index}"
            
            mapped = label_maps.get(sample_id, {})
            label = mapped.get("label", default_label)
            if label_col is not None:
                label = parse_label_value(data[columns[label_col]][i], default=label)
            family = mapped.get("family", default_family)
            if family_col is not None:
                family_val = str(data[columns[family_col]][i]).strip()
                if family_val:
                    family = family_val
            if dataset == "CIC-YNU_IoTMal":
                 family_name = str(family).strip()
                 label = 0 if family_name.lower() == "benign" else 1
                 if not should_keep_iotmal_sample(family_name):
                    row_index += 1
                    continue
            
            if feature_col:
                features = data[feature_col][i] or []
            else:
                features = [
                    _safe_float(data[columns[idx]][i])
                    for idx in numeric_indexes
                    if _safe_float(data[columns[idx]][i]) is not None
                ]
            if not features:
                row_index += 1
                continue

            feature_path = None
            if not manifest_only:
                import torch

                feature_path = feature_tensor_path(feature_cache, file_key, path.stem, row_index, sample_id)
                torch.save(build_memory_trace(list(features)), feature_path)
            
            arch = infer_arch(path)

            if "Arch" in data and data["Arch"][i] is not None:
                arch_val = str(data["Arch"][i]).strip().lower()
                if arch_val :
                   arch = arch_val
            row = {
                "path": str(path),
                "row_index": row_index,
                "sample_id": sample_id,
                "sha256": sample_id if len(sample_id) >= 16 else f"{sha256_file(path)}:{row_index}",
                "dataset": dataset,
                "data_type": "feature_parquet",
                "feature_path": str(feature_path) if feature_path else "",
                "feature_dim": len(features),
                "label": label,
                "family": family,
                "arch": arch ,
            }
            row_split = normalize_split(str(data[split_col][i])) if split_col else split_hint
            if row_split:
                row["split"] = row_split
            if row_sink:
                row_sink(row)
            else:
                rows.append(row)
            row_index += 1
            count += 1
    return rows, count


def _hash_split(value: str, train_ratio: float, val_ratio: float, seed: int) -> str:
    digest = hashlib.sha256(f"{seed}:{value}".encode("utf-8")).hexdigest()
    ratio = int(digest[:8], 16) / 0xFFFFFFFF
    if ratio < train_ratio:
        return "train"
    if ratio < train_ratio + val_ratio:
        return "val"
    return "test"


def _progress_key(path: Path) -> str:
    stat = path.stat()
    payload = f"{path.as_posix()}:{stat.st_size}:{int(stat.st_mtime)}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def split_rows(
    rows: list[dict],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 1337,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Deterministic train/val/test split.

    Stratifies coarsely by label and family where available. If a bucket is too
    small, it is still assigned deterministically so every sample appears in
    exactly one split.
    """

    buckets: dict[tuple, list[dict]] = {}
    for row in rows:
        key = (row.get("label", 0), row.get("family", "unknown"))
        buckets.setdefault(key, []).append(row)

    rng = random.Random(seed)
    train, val, test = [], [], []
    for bucket in buckets.values():
        rng.shuffle(bucket)
        n = len(bucket)
        if n < 3:
            train.extend(bucket)
            continue
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        n_train = max(1, n_train)
        n_val = max(1, n_val)
        if n_train + n_val >= n:
            n_train = n - 2
            n_val = 1
        train.extend(bucket[:n_train])
        val.extend(bucket[n_train : n_train + n_val])
        test.extend(bucket[n_train + n_val :])

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    if not test and len(train) > 2:
        test.append(train.pop())
    if not val and len(train) > 2:
        val.append(train.pop())
    return train, val, test


def split_train_val(rows: list[dict], val_ratio: float = 0.1, seed: int = 1337) -> tuple[list[dict], list[dict]]:
    if not rows or val_ratio <= 0:
        return rows, []
    buckets: dict[tuple, list[dict]] = {}
    for row in rows:
        key = (row.get("label", 0), row.get("family", "unknown"))
        buckets.setdefault(key, []).append(row)
    rng = random.Random(seed)
    train, val = [], []
    for bucket in buckets.values():
        rng.shuffle(bucket)
        n = len(bucket)
        if n < 2:
            train.extend(bucket)
            continue
        n_val = max(1, int(n * val_ratio))
        if n_val >= n:
            n_val = 1
        val.extend(bucket[:n_val])
        train.extend(bucket[n_val:])
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def write_splits(out: Path, rows: list[dict], train_ratio: float, val_ratio: float, seed: int) -> dict[str, int]:
    split_rows_map = {"train": [], "val": [], "test": []}
    unknown: list[dict] = []
    for row in rows:
        split = normalize_split(row.get("split"))
        if split in split_rows_map:
            split_rows_map[split].append(row)
        else:
            unknown.append(row)
    has_explicit = any(split_rows_map.values())
    if not has_explicit:
        train, val, test = split_rows(rows, train_ratio=train_ratio, val_ratio=val_ratio, seed=seed)
    else:
        train = split_rows_map["train"] + unknown
        val = split_rows_map["val"]
        test = split_rows_map["test"]
        if not train:
            train, val, test = split_rows(rows, train_ratio=train_ratio, val_ratio=val_ratio, seed=seed)
        elif not val:
            train, val = split_train_val(train, val_ratio=val_ratio, seed=seed)
    split_dir = out.parent
    write_jsonl(split_dir / "train_manifest.jsonl", train)
    write_jsonl(split_dir / "val_manifest.jsonl", val)
    write_jsonl(split_dir / "test_manifest.jsonl", test)
    return {"train": len(train), "val": len(val), "test": len(test)}


def build_manifest(
    root: Path,
    out: Path,
    max_binary_bytes: int = 2_000_000,
    max_rows_per_csv: int | None = None,
    max_rows_per_parquet: int | None = None,
    split_mode: str = "stratified",
    resume: bool = False,
    only_dataset: str | None = None,
    make_splits: bool = True,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 1337,
    manifest_only: bool = False,
) -> None:
    raw = root / "raw"
    cache = root / "cache" / "isr"
    if not manifest_only:
        cache.mkdir(parents=True, exist_ok=True)
    split_mode = split_mode.lower().strip()
    if split_mode not in {"stratified", "hash"}:
        raise ValueError(f"Unknown split mode: {split_mode}")
    if manifest_only and resume:
        raise ValueError("--resume cannot be used with --manifest-only because resume markers are cache files.")
    rows: list[dict] = []
    if not raw.exists():
        raise FileNotFoundError(f"raw dataset folder not found: {raw}")
    label_maps = load_label_maps(raw)
    if label_maps:
        print(f"Loaded {len(label_maps)} CSV label-map entries")
    normalizers: dict[str, ArchitectureNormalizationPipeline] = {}
    scan_root = raw
    if only_dataset:
        scan_root = raw / only_dataset
        if not scan_root.exists():
            raise FileNotFoundError(f"dataset folder not found: {scan_root}")

    if split_mode == "hash":
        if resume and not out.exists():
            resume = False
        if resume and not make_splits:
            raise ValueError("Resume is only supported when splits are enabled in hash mode.")
        out.parent.mkdir(parents=True, exist_ok=True)
        split_dir = out.parent
        train_path = split_dir / "train_manifest.jsonl"
        val_path = split_dir / "val_manifest.jsonl"
        test_path = split_dir / "test_manifest.jsonl"
        counts = {"full": 0, "train": 0, "val": 0, "test": 0}
        family_names: set[str] = set()
        progress_dir = root / "cache" / "manifest_progress"
        if not manifest_only:
            progress_dir.mkdir(parents=True, exist_ok=True)
        mode = "a" if resume else "w"
        with open(out, mode, encoding="utf-8") as manifest_f, open(
            train_path, mode, encoding="utf-8"
        ) as train_f, open(val_path, mode, encoding="utf-8") as val_f, open(
            test_path, mode, encoding="utf-8"
        ) as test_f:
            def emit_row(row: dict) -> None:
                manifest_f.write(json.dumps(row, sort_keys=True) + "\n")
                counts["full"] += 1
                family_names.add(normalize_family_name(row.get("family", "unknown")))
                if not make_splits:
                    return
                split = normalize_split(row.get("split"))
                if not split:
                    split_key = row.get("sha256") or row.get("sample_id") or row.get("path")
                    split = _hash_split(str(split_key), train_ratio=train_ratio, val_ratio=val_ratio, seed=seed)
                if split == "train":
                    train_f.write(json.dumps(row, sort_keys=True) + "\n")
                elif split == "val":
                    val_f.write(json.dumps(row, sort_keys=True) + "\n")
                else:
                    split = "test"
                    test_f.write(json.dumps(row, sort_keys=True) + "\n")
                counts[split] += 1

            current_folder = None
            for path in sorted(scan_root.rglob("*")):
                if not path.is_file() or should_skip_manifest_path(path):
                    continue
                current_folder = print_parse_folder(path, raw, current_folder)
                marker = progress_dir / f"{_progress_key(path)}.done"
                if resume and marker.exists():
                    continue
                if path.suffix.lower() == ".csv":
                    with open(path, "r", encoding="utf-8", errors="ignore", newline="") as f:
                        first = next(csv.reader(f), [])
                    if is_api_sequence_csv(path, first):
                        _, count = process_api_sequence_csv(
                            path,
                            raw=raw,
                            max_rows_per_csv=max_rows_per_csv,
                            row_sink=emit_row,
                        )
                        print(f"Parsed {count} API sequence rows from {path}")
                        if not manifest_only:
                            marker.write_text(str(path), encoding="utf-8")
                        continue
                    _, count = process_feature_csv(
                        path,
                        raw=raw,
                        cache=cache,
                        label_maps=label_maps,
                        max_rows_per_csv=max_rows_per_csv,
                        row_sink=emit_row,
                        manifest_only=manifest_only,
                    )
                    print(f"Parsed {count} feature rows from {path}")
                    if not manifest_only:
                        marker.write_text(str(path), encoding="utf-8")
                    continue
                if path.suffix.lower() == ".txt" and path.name.lower() == "all_analysis_data.txt":
                    _, count = process_api_sequence_txt(
                        path,
                        raw=raw,
                        max_rows=max_rows_per_csv,
                        row_sink=emit_row,
                    )
                    print(f"Parsed {count} API sequence rows from {path}")
                    if not manifest_only:
                        marker.write_text(str(path), encoding="utf-8")
                    continue
                if path.suffix.lower() == ".parquet":
                    _, count = process_feature_parquet(
                        path,
                        raw=raw,
                        cache=cache,
                        label_maps=label_maps,
                        max_rows_per_parquet=max_rows_per_parquet,
                        row_sink=emit_row,
                        manifest_only=manifest_only,
                    )
                    print(f"Parsed {count} feature rows from {path}")
                    if not manifest_only:
                        marker.write_text(str(path), encoding="utf-8")
                    continue
                record = {
                    "path": str(path),
                    "sha256": sha256_file(path),
                    "dataset": path.relative_to(raw).parts[0] if len(path.relative_to(raw).parts) else "unknown",
                    "label": infer_label(path),
                    "family": path.parent.name,
                    "arch": infer_arch(path),
                }
                record = enrich_dynamic_report(record, path)
                is_binary_candidate = path.stat().st_size <= max_binary_bytes and path.suffix.lower() in {".bin", ".exe", ".dll", ".so", ".elf", ""}
                if manifest_only and is_binary_candidate:
                    record["isr_path"] = ""
                if not manifest_only and is_binary_candidate:
                    arch = record["arch"]
                    if arch != "unknown":
                        normalizers.setdefault(arch, ArchitectureNormalizationPipeline(arch=arch))
                        blob = path.read_bytes()
                        isr = normalizers[arch].process({"bytes": blob, "arch": arch})
                        isr_path = cache / f"{record['sha256']}.pt"
                        import torch

                        torch.save(isr, isr_path)
                        record["isr_path"] = str(isr_path)
                    else:
                        record["isr_path"] = ""
                emit_row(record)
                if not manifest_only:
                    marker.write_text(str(path), encoding="utf-8")

        if counts["full"] == 0:
            if resume:
                print("No new dataset files found to process.")
                return
            raise RuntimeError(f"No dataset files found under {scan_root}. Missing datasets are allowed, but at least one usable dataset is required.")
        write_family_vocab(split_dir, sorted(family_names) or ["unknown"])
        print(f"Wrote {counts['full']} manifest rows to {out}")
        if make_splits:
            print(f"Wrote split manifests to {out.parent}: {{'train': {counts['train']}, 'val': {counts['val']}, 'test': {counts['test']}}}")
        return

    current_folder = None
    for path in sorted(scan_root.rglob("*")):
        if not path.is_file() or should_skip_manifest_path(path):
            continue
        current_folder = print_parse_folder(path, raw, current_folder)
        if path.suffix.lower() == ".csv":
            with open(path, "r", encoding="utf-8", errors="ignore", newline="") as f:
                first = next(csv.reader(f), [])
            if is_api_sequence_csv(path, first):
                api_rows, count = process_api_sequence_csv(
                    path,
                    raw=raw,
                    max_rows_per_csv=max_rows_per_csv,
                )
                rows.extend(api_rows)
                print(f"Parsed {count} API sequence rows from {path}")
                continue
            csv_rows, count = process_feature_csv(
                path,
                raw=raw,
                cache=cache,
                label_maps=label_maps,
                max_rows_per_csv=max_rows_per_csv,
                manifest_only=manifest_only,
            )
            rows.extend(csv_rows)
            print(f"Parsed {count} feature rows from {path}")
            continue
        if path.suffix.lower() == ".txt" and path.name.lower() == "all_analysis_data.txt":
            api_rows, count = process_api_sequence_txt(
                path,
                raw=raw,
                max_rows=max_rows_per_csv,
            )
            rows.extend(api_rows)
            print(f"Parsed {count} API sequence rows from {path}")
            continue
        if path.suffix.lower() == ".parquet":
            parquet_rows, count = process_feature_parquet(
                path,
                raw=raw,
                cache=cache,
                label_maps=label_maps,
                max_rows_per_parquet=max_rows_per_parquet,
                manifest_only=manifest_only,
            )
            rows.extend(parquet_rows)
            print(f"Parsed {count} feature rows from {path}")
            continue
        record = {
            "path": str(path),
            "sha256": sha256_file(path),
            "dataset": path.relative_to(raw).parts[0] if len(path.relative_to(raw).parts) else "unknown",
            "label": infer_label(path),
            "family": path.parent.name,
            "arch": infer_arch(path),
        }
        record = enrich_dynamic_report(record, path)
        is_binary_candidate = path.stat().st_size <= max_binary_bytes and path.suffix.lower() in {".bin", ".exe", ".dll", ".so", ".elf", ""}
        if manifest_only and is_binary_candidate:
            record["isr_path"] = ""
        if not manifest_only and is_binary_candidate:
            arch = record["arch"]
            if arch != "unknown":
                normalizers.setdefault(arch, ArchitectureNormalizationPipeline(arch=arch))
                blob = path.read_bytes()
                isr = normalizers[arch].process({"bytes": blob, "arch": arch})
                isr_path = cache / f"{record['sha256']}.pt"
                import torch

                torch.save(isr, isr_path)
                record["isr_path"] = str(isr_path)
            else:
                record["isr_path"] = ""
        rows.append(record)
    if not rows:
        raise RuntimeError(f"No dataset files found under {raw}. Missing datasets are allowed, but at least one usable dataset is required.")
    write_family_vocab(out.parent, collect_family_names(rows))
    write_jsonl(out, rows)
    print(f"Wrote {len(rows)} manifest rows to {out}")
    if make_splits:
        counts = write_splits(out, rows, train_ratio=train_ratio, val_ratio=val_ratio, seed=seed)
        print(f"Wrote split manifests to {out.parent}: {counts}")


def _feature_csv_values_for_row(path: Path, row_index: int) -> list[float]:
    with open(path, "r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.reader(f)
        first = next(reader, [])
        if not first:
            return []
        has_header = _looks_like_header(first)
        id_idx = 0
        label_idx = None
        family_idx = None
        numeric_indexes = list(range(1, len(first)))
        if has_header:
            headers = first
            id_idx = column_index(headers, ID_COLUMNS)
            label_idx = column_index(headers, LABEL_COLUMNS)
            family_idx = column_index(headers, FAMILY_COLUMNS)
            normalized = [norm_col(h) for h in headers]
            skip_cols = {idx for idx in (id_idx, label_idx, family_idx) if idx is not None}
            numeric_indexes = [
                i
                for i, name in enumerate(normalized)
                if i not in skip_cols and name not in LABEL_COLUMNS and name not in FAMILY_COLUMNS
            ]

        def _row_iter():
            if not has_header:
                yield first
            for row in reader:
                yield row

        for idx, row in enumerate(_row_iter()):
            if idx == row_index:
                return row_features(row, numeric_indexes)
    return []


def _feature_parquet_values_for_row(path: Path, row_index: int) -> list[float]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("Reading parquet requires pyarrow. Install pyarrow to parse parquet datasets.") from exc

    parquet = pq.ParquetFile(path)
    columns = parquet.schema.names
    feature_col = _feature_column_name(columns)
    if feature_col is None:
        id_col = column_index(columns, ID_COLUMNS)
        label_col = column_index(columns, LABEL_COLUMNS)
        family_col = column_index(columns, FAMILY_COLUMNS)
        normalized = [norm_col(h) for h in columns]
        skip_cols = {idx for idx in (id_col, label_col, family_col) if idx is not None}
        numeric_indexes = [
            i
            for i, name in enumerate(normalized)
            if i not in skip_cols and name not in LABEL_COLUMNS and name not in FAMILY_COLUMNS
        ]
    else:
        numeric_indexes = []

    current = 0
    for batch in parquet.iter_batches(batch_size=2048, columns=columns):
        if current + batch.num_rows <= row_index:
            current += batch.num_rows
            continue
        local_index = row_index - current
        data = batch.to_pydict()
        if feature_col:
            return list(data[feature_col][local_index] or [])
        return [
            _safe_float(data[columns[idx]][local_index])
            for idx in numeric_indexes
            if _safe_float(data[columns[idx]][local_index]) is not None
        ]
    return []


def _is_binary_cache_candidate(path: Path, max_binary_bytes: int) -> bool:
    return (
        path.exists()
        and path.is_file()
        and path.stat().st_size <= max_binary_bytes
        and path.suffix.lower() in {".bin", ".exe", ".dll", ".so", ".elf", ""}
    )


def _save_tensor_without_overwrite(torch_module, tensor, path: Path) -> bool:
    if path.exists():
        return False
    tmp = path.with_name(f"{path.name}.tmp")
    torch_module.save(tensor, tmp)
    tmp.replace(path)
    return True


def generate_cache_from_manifest(
    root: Path,
    manifest_path: Path,
    max_binary_bytes: int = 2_000_000,
) -> dict[str, int]:
    """Generate tensor caches for only the samples listed in an existing manifest.

    The manifest is rewritten in place with populated feature_path/isr_path fields.
    Split manifests can reuse these cache files because cache paths are
    deterministic from the source path, row_index, sample_id, and sha256.
    """

    rows = read_jsonl(manifest_path)
    if not rows:
        raise RuntimeError(f"Manifest is empty: {manifest_path}")

    cache = root / "cache" / "isr"
    feature_cache = cache / "features"
    cache.mkdir(parents=True, exist_ok=True)
    feature_cache.mkdir(parents=True, exist_ok=True)

    import torch

    normalizers: dict[str, ArchitectureNormalizationPipeline] = {}
    counts = {
        "rows": len(rows),
        "generated_feature_tensors": 0,
        "generated_isr_tensors": 0,
        "skipped_existing": 0,
        "skipped_invalid": 0,
        "failed": 0,
    }
    for row in rows:
        path = Path(row.get("path", ""))
        data_type = row.get("data_type")
        if data_type == "feature_csv":
            sample_id = str(row.get("sample_id") or f"{path.stem}_{row.get('row_index', 0)}")
            try:
                features = _feature_csv_values_for_row(path, int(row.get("row_index", 0)))
                if not features:
                    counts["skipped_invalid"] += 1
                    continue
                file_key = sha256_file(path)[:12]
                feature_path = feature_tensor_path(feature_cache, file_key, path.stem, int(row.get("row_index", 0)), sample_id)
                if feature_path.exists():
                    row["feature_path"] = str(feature_path)
                    row["feature_dim"] = len(features)
                    counts["skipped_existing"] += 1
                else:
                    generated = _save_tensor_without_overwrite(torch, build_memory_trace(features), feature_path)
                    row["feature_path"] = str(feature_path)
                    row["feature_dim"] = len(features)
                    if generated:
                        counts["generated_feature_tensors"] += 1
                    else:
                        counts["skipped_existing"] += 1
            except Exception as exc:
                counts["failed"] += 1
                print(f"Failed: {sample_id}: {exc}")
            continue
        if data_type == "feature_parquet":
            sample_id = str(row.get("sample_id") or f"{path.stem}_{row.get('row_index', 0)}")
            try:
                features = _feature_parquet_values_for_row(path, int(row.get("row_index", 0)))
                if not features:
                    counts["skipped_invalid"] += 1
                    continue
                file_key = sha256_file(path)[:12]
                feature_path = feature_tensor_path(feature_cache, file_key, path.stem, int(row.get("row_index", 0)), sample_id)
                if feature_path.exists():
                    row["feature_path"] = str(feature_path)
                    row["feature_dim"] = len(features)
                    counts["skipped_existing"] += 1
                else:
                    generated = _save_tensor_without_overwrite(torch, build_memory_trace(list(features)), feature_path)
                    row["feature_path"] = str(feature_path)
                    row["feature_dim"] = len(features)
                    if generated:
                        counts["generated_feature_tensors"] += 1
                    else:
                        counts["skipped_existing"] += 1
            except Exception as exc:
                counts["failed"] += 1
                print(f"Failed: {sample_id}: {exc}")
            continue
        if _is_binary_cache_candidate(path, max_binary_bytes):
            sample_id = str(row.get("sample_id") or row.get("sha256") or path.name)
            try:
                sha = row.get("sha256") or sha256_file(path)
                isr_path = cache / f"{sha}.pt"
                if isr_path.exists():
                    row["isr_path"] = str(isr_path)
                    counts["skipped_existing"] += 1
                else:
                    arch = row.get("arch", infer_arch(path))
                    normalizers.setdefault(arch, ArchitectureNormalizationPipeline(arch=arch))
                    isr = normalizers[arch].process({"bytes": path.read_bytes(), "arch": arch})
                    generated = _save_tensor_without_overwrite(torch, isr, isr_path)
                    row["isr_path"] = str(isr_path)
                    if generated:
                        counts["generated_isr_tensors"] += 1
                    else:
                        counts["skipped_existing"] += 1
            except Exception as exc:
                counts["failed"] += 1
                print(f"Failed: {sample_id}: {exc}")
            continue
        counts["skipped_invalid"] += 1

    tmp = manifest_path.with_name(f"{manifest_path.name}.tmp")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(manifest_path)
    print(json.dumps(counts, indent=2, sort_keys=True))
    return counts


def validate_manifest(path: Path) -> None:
    count = 0
    missing = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            count += 1
            if not Path(row["path"]).exists():
                missing.append(row["path"])
    if missing:
        raise FileNotFoundError(f"{len(missing)} manifest paths are missing, first={missing[0]}")
    if count == 0:
        raise RuntimeError(f"Manifest is empty: {path}")
    print(f"Validated {count} manifest rows")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build X-NERF++ unified manifest and optional tensor caches",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--root", type=Path, default=Path("data"), help="Dataset root containing raw/ and cache/")
    parser.add_argument("--out", type=Path, default=Path("data/processed/manifest.jsonl"), help="Output manifest JSONL path")
    parser.add_argument("--only-dataset", type=str, default=None, help="Only scan data/raw/<dataset_name>")
    parser.add_argument("--max-rows-per-csv", type=int, default=None, help="Limit rows parsed per CSV file")
    parser.add_argument("--max-rows-per-parquet", type=int, default=None, help="Limit rows parsed per parquet file")
    parser.add_argument("--split-mode", choices=["stratified", "hash"], default="stratified", help="Split strategy")
    parser.add_argument("--resume", action="store_true", help="Resume hash-mode manifest build using progress markers")
    parser.add_argument("--validate", action="store_true", help="Validate that manifest source paths exist")
    parser.add_argument("--no-split", action="store_true", help="Write only --out, not train/val/test manifests")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Training split ratio")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation split ratio")
    parser.add_argument("--seed", type=int, default=1337, help="Deterministic split seed")
    parser.add_argument(
        "--manifest-only",
        action="store_true",
        help="Build manifest and split metadata only; do not create ISR, feature, tensor, or progress cache files",
    )
    parser.add_argument(
        "--generate-cache-from-manifest",
        type=Path,
        default=None,
        metavar="MANIFEST_PATH",
        help="Read an existing manifest, generate tensor caches only for its rows, and update cache paths in place",
    )
    args = parser.parse_args()
    if args.generate_cache_from_manifest:
        generate_cache_from_manifest(args.root, args.generate_cache_from_manifest)
    elif args.validate:
        validate_manifest(args.out)
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        build_manifest(
            args.root,
            args.out,
            only_dataset=args.only_dataset,
            max_rows_per_csv=args.max_rows_per_csv,
            max_rows_per_parquet=args.max_rows_per_parquet,
            split_mode=args.split_mode,
            resume=args.resume,
            make_splits=not args.no_split,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            seed=args.seed,
            manifest_only=args.manifest_only,
        )


if __name__ == "__main__":
    main()
