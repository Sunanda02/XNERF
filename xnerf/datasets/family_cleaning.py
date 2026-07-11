from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

DEFAULT_PLACEHOLDER_PATTERNS: tuple[str, ...] = (
    r"^(train|test)_ember_2018_v2_features$",
    r"^public_.*_reports$",
    r"^dynamic_api_call_sequence_",
    r"^drebin-\d+-dataset-",
    r"^malbehavd-v1-dataset$",
    r"^feature_vectors_",
    r"^cicmaldroid2020$",
    r"^malnet_tiny$",
    r"^ember$",
    r"^andmal2020$",
    r"^malapi2019$",
    r"^malwareanalysisdatasetsapicallsequences$",
    r"^<unknown>$",
    r"^no_category$",
    r"^nocategory$",
)

DEFAULT_BENIGN_MARKERS: frozenset[str] = frozenset({"ben0", "ben1", "ben2", "ben3", "ben4", "benign", "goodware", "clean", "false", "normal"})

DEFAULT_FAMILY_ALIASES: dict[str, str] = {
    "spy": "Spyware",
    "spyware": "Spyware",
    "worms": "Worm",
    "worm": "Worm",
    "virus": "Virus",
    "viruses": "Virus",
    "trojan": "Trojan",
    "trojans": "Trojan",
    "adware": "Adware",
    "ransomware": "Ransomware",
    "backdoor": "Backdoor",
    "downloader": "Downloader",
    "banker": "Banker",
    "dropper": "Dropper",
    "scareware": "Scareware",
    "pua": "PUA",
    "genpua": "PUA",
    "fileinfector": "FileInfector",
    "fileinfectors": "FileInfector",
}


@dataclass(frozen=True)
class FamilyNormalizationRules:
    """Configurable family normalization rules.

    Rules can be loaded from JSON/YAML or constructed in code.
    """

    aliases: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_FAMILY_ALIASES))
    benign_markers: frozenset[str] = DEFAULT_BENIGN_MARKERS
    placeholder_patterns: tuple[str, ...] = DEFAULT_PLACEHOLDER_PATTERNS
    unknown_tokens: frozenset[str] = frozenset({"unknown", "<unknown>", "other", "generic", "no_category", "nocategory"})

    def compiled_placeholders(self) -> list[re.Pattern[str]]:
        return [re.compile(pattern, re.IGNORECASE) for pattern in self.placeholder_patterns]


def load_family_normalization_rules(path: str | Path | None = None) -> FamilyNormalizationRules:
    if path is None:
        return FamilyNormalizationRules()
    path = Path(path)
    if not path.exists():
        return FamilyNormalizationRules()

    raw: Any
    try:
        if path.suffix.lower() in {".json", ".jsonl"}:
            raw = json.loads(path.read_text(encoding="utf-8"))
        else:
            try:
                import yaml
            except ImportError:
                return FamilyNormalizationRules()
            raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception:
        return FamilyNormalizationRules()

    if not isinstance(raw, Mapping):
        return FamilyNormalizationRules()

    aliases = raw.get("aliases")
    benign_markers = raw.get("benign_markers")
    placeholder_patterns = raw.get("placeholder_patterns")
    unknown_tokens = raw.get("unknown_tokens")
    return FamilyNormalizationRules(
        aliases=dict(aliases) if isinstance(aliases, Mapping) else dict(DEFAULT_FAMILY_ALIASES),
        benign_markers=frozenset(str(item).strip().lower() for item in benign_markers) if isinstance(benign_markers, list) else DEFAULT_BENIGN_MARKERS,
        placeholder_patterns=tuple(str(item) for item in placeholder_patterns) if isinstance(placeholder_patterns, list) else DEFAULT_PLACEHOLDER_PATTERNS,
        unknown_tokens=frozenset(str(item).strip().lower() for item in unknown_tokens) if isinstance(unknown_tokens, list) else frozenset({"unknown", "<unknown>", "other", "generic", "no_category", "nocategory"}),
    )


def _normalize_text(value: object) -> str:
    text = str(value).strip()
    return re.sub(r"\s+", " ", text) if text else "unknown"


def _normalized_key(value: object) -> str:
    text = _normalize_text(value)
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def family_placeholder_reason(family: object, rules: FamilyNormalizationRules | None = None) -> str | None:
    rules = rules or FamilyNormalizationRules()
    text = _normalize_text(family)
    lowered = text.lower()
    if not lowered:
        return "empty"
    if lowered in rules.benign_markers:
        return None
    if lowered in rules.unknown_tokens:
        return "unknown-token"
    for pattern in rules.compiled_placeholders():
        if pattern.search(text):
            return "placeholder-pattern"
    return None


def is_placeholder_family(family: object, rules: FamilyNormalizationRules | None = None) -> bool:
    return family_placeholder_reason(family, rules=rules) is not None


def normalize_family_name(family: object, label: int | str | None = None, rules: FamilyNormalizationRules | None = None) -> str:
    rules = rules or FamilyNormalizationRules()
    text = _normalize_text(family)
    lowered = text.lower()

    if label is not None and str(label).strip() in {"0", "benign", "goodware", "clean", "false", "normal"}:
        return "benign"
    if lowered in rules.benign_markers:
        return "benign"
    placeholder_reason = family_placeholder_reason(text, rules=rules)
    if placeholder_reason is not None:
        return "unknown"

    alias_key = _normalized_key(text)
    if alias_key in rules.aliases:
        return rules.aliases[alias_key]

    if text.lower().startswith("spy"):
        return "Spyware"
    if text.lower().startswith("worm"):
        return "Worm"
    if text.lower().startswith("virus"):
        return "Virus"
    if text.lower().startswith("trojan"):
        return "Trojan"
    if text.lower().startswith("adware"):
        return "Adware"
    if text.lower().startswith("backdoor"):
        return "Backdoor"
    if text.lower().startswith("downloader"):
        return "Downloader"
    if text.lower().startswith("banker"):
        return "Banker"
    if text.lower().startswith("ransomware"):
        return "Ransomware"
    if text.lower().startswith("scareware"):
        return "Scareware"
    if text.lower().startswith("pua"):
        return "PUA"
    if text.lower().startswith("dropper"):
        return "Dropper"
    if text.lower().startswith("fileinfector"):
        return "FileInfector"
    if lowered in rules.unknown_tokens:
        return "unknown"
    return text or "unknown"


def normalize_family_rows(rows: Iterable[dict[str, Any]], rules: FamilyNormalizationRules | None = None) -> list[dict[str, Any]]:
    rules = rules or FamilyNormalizationRules()
    cleaned: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        label = int(item.get("label", 0))
        item["family_raw"] = _normalize_text(item.get("family", "unknown"))
        item["family"] = normalize_family_name(item.get("family", "unknown"), label=label, rules=rules)
        cleaned.append(item)
    return cleaned


def build_family_vocabulary(rows: Iterable[dict[str, Any]], rules: FamilyNormalizationRules | None = None) -> list[str]:
    rules = rules or FamilyNormalizationRules()
    names = {normalize_family_name(row.get("family", "unknown"), label=row.get("label", 0), rules=rules) for row in rows}
    ordered = sorted(_normalize_text(name) for name in names if _normalize_text(name))
    if "benign" in ordered:
        ordered.remove("benign")
        ordered.insert(0, "benign")
    if "unknown" in ordered:
        ordered.remove("unknown")
        ordered.insert(1 if ordered and ordered[0] == "benign" else 0, "unknown")
    return ordered or ["unknown"]


def family_to_id_map(family_names: Iterable[str]) -> dict[str, int]:
    ordered = [_normalize_text(name) for name in family_names]
    return {name: idx for idx, name in enumerate(ordered)}


def id_to_family_map(family_names: Iterable[str]) -> dict[int, str]:
    ordered = [_normalize_text(name) for name in family_names]
    return {idx: name for idx, name in enumerate(ordered)}


def family_vocab_payload(rows_or_names: Iterable[dict[str, Any]] | Iterable[str], rules: FamilyNormalizationRules | None = None) -> dict[str, Any]:
    items = list(rows_or_names)
    if not items:
        family_names = ["unknown"]
    elif isinstance(items[0], Mapping):
        family_names = build_family_vocabulary(items, rules=rules)  # type: ignore[arg-type]
    else:
        family_names = build_family_vocabulary(({"family": name, "label": 1} for name in items), rules=rules)  # type: ignore[arg-type]
    return {
        "family_names": family_names,
        "family_to_id": family_to_id_map(family_names),
        "id_to_family": family_names,
    }
