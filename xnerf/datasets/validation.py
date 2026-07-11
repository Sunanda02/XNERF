from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

from xnerf.datasets.family_cleaning import FamilyNormalizationRules, build_family_vocabulary, is_placeholder_family, load_family_normalization_rules, normalize_family_name


def family_names_from_metadata(metadata: Mapping[str, Any] | None) -> list[str]:
    if not isinstance(metadata, Mapping):
        return []
    family_names = metadata.get("family_names")
    if isinstance(family_names, list):
        return [str(name).strip() for name in family_names if str(name).strip()]
    id_to_family = metadata.get("id_to_family")
    if isinstance(id_to_family, list):
        return [str(name).strip() for name in id_to_family if str(name).strip()]
    if isinstance(id_to_family, Mapping):
        def _key(item: Any) -> tuple[int, str]:
            key = item[0]
            try:
                return int(key), str(key)
            except (TypeError, ValueError):
                return 0, str(key)

        return [str(value).strip() for _, value in sorted(id_to_family.items(), key=_key) if str(value).strip()]
    families = metadata.get("families")
    if isinstance(families, list):
        return [str(name).strip() for name in families if str(name).strip()]
    return []


def validate_family_rows(rows: Iterable[Mapping[str, Any]], family_vocab: Iterable[str] | None = None, rules: FamilyNormalizationRules | str | Path | None = None) -> list[str]:
    rules = load_family_normalization_rules(rules) if isinstance(rules, (str, Path)) else (rules or FamilyNormalizationRules())
    known_vocab = {str(name).strip() for name in (family_vocab or []) if str(name).strip()}
    issues: list[str] = []
    for index, row in enumerate(rows):
        family = str(row.get("family", "")).strip()
        label = row.get("label")
        if not family:
            issues.append(f"row {index}: missing family value")
            continue
        normalized = normalize_family_name(family, label=label, rules=rules)
        if not normalized:
            issues.append(f"row {index}: empty normalized family for {family!r}")
        if is_placeholder_family(family, rules=rules):
            issues.append(f"row {index}: placeholder family value {family!r}")
        if known_vocab and normalized not in known_vocab:
            issues.append(f"row {index}: missing vocab entry for family {normalized!r}")
    return issues


def validate_family_vocab(rows: Iterable[Mapping[str, Any]], family_vocab: Iterable[str], rules: FamilyNormalizationRules | str | Path | None = None) -> list[str]:
    rules = load_family_normalization_rules(rules) if isinstance(rules, (str, Path)) else (rules or FamilyNormalizationRules())
    normalized_vocab = [str(name).strip() for name in family_vocab if str(name).strip()]
    issues: list[str] = []
    if not normalized_vocab:
        issues.append("family vocabulary is empty")
        return issues
    row_families = {normalize_family_name(row.get("family", ""), label=row.get("label"), rules=rules) for row in rows}
    missing = sorted(name for name in row_families if name not in normalized_vocab)
    for name in missing:
        issues.append(f"missing vocab entry for family {name!r}")
    return issues


def validate_checkpoint_family_metadata(checkpoint_payload: Mapping[str, Any], family_vocab: Iterable[str] | None = None) -> list[str]:
    issues: list[str] = []
    checkpoint_names = family_names_from_metadata(checkpoint_payload)
    if not checkpoint_names:
        issues.append("checkpoint is missing family_names metadata")
        return issues
    if family_vocab is not None:
        vocab_names = [str(name).strip() for name in family_vocab if str(name).strip()]
        if len(vocab_names) != len(checkpoint_names):
            issues.append(f"checkpoint family count {len(checkpoint_names)} does not match vocab count {len(vocab_names)}")
        missing = sorted(name for name in vocab_names if name not in checkpoint_names)
        extra = sorted(name for name in checkpoint_names if name not in vocab_names)
        for name in missing:
            issues.append(f"checkpoint missing family name {name!r}")
        for name in extra:
            issues.append(f"checkpoint has extra family name {name!r}")
    return issues


def validate_family_batch(batch: Mapping[str, Any]) -> None:
    if "family_label" not in batch:
        raise RuntimeError("training batch is missing family_label")
    family_label = batch["family_label"]
    if family_label is None:
        raise RuntimeError("training batch family_label is empty")


def family_vocab_from_rows(rows: Iterable[Mapping[str, Any]], rules: FamilyNormalizationRules | str | Path | None = None) -> list[str]:
    rules = load_family_normalization_rules(rules) if isinstance(rules, (str, Path)) else (rules or FamilyNormalizationRules())
    cleaned_rows = [dict(row) for row in rows]
    return build_family_vocabulary(cleaned_rows, rules=rules)
