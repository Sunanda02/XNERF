from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from xnerf.preprocessing.ontology import ARCH_TO_ID


KNOWN_ARCHES = frozenset(k for k in ARCH_TO_ID if k != "unknown")
ARCH_HINTS = {
    "x86": ("x86", "i386", "i686", "win32", "32bit"),
    "x64": ("x64", "amd64", "x86_64", "64bit", "x86-64"),
    "arm": ("arm", "armeabi", "armv7", "thumb"),
    "arm64": ("arm64", "aarch64", "armv8"),
    "mips": ("mips",),
    "riscv": ("riscv", "rv64", "rv32"),
}


def _normalize(value: object) -> str:
    return str(value).strip().lower()


def inspect_architecture_distribution(rows: Iterable[Mapping[str, Any]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for row in rows:
        counts[_normalize(row.get("arch", "unknown")) or "unknown"] += 1
    return counts


def detect_single_architecture_dataset(rows: Iterable[Mapping[str, Any]], dominance_threshold: float = 0.95, min_rows: int = 100) -> dict[str, Any]:
    rows = list(rows)
    counts = inspect_architecture_distribution(rows)
    total = sum(counts.values())
    dominant_arch, dominant_count = counts.most_common(1)[0] if counts else ("unknown", 0)
    dominant_share = dominant_count / total if total else 0.0
    return {
        "total_rows": total,
        "distribution": counts,
        "dominant_arch": dominant_arch,
        "dominant_count": dominant_count,
        "dominant_share": dominant_share,
        "is_single_architecture": total >= min_rows and dominant_share >= dominance_threshold,
    }


def suspicious_architecture_assignments(rows: Iterable[Mapping[str, Any]], known_arches: Iterable[str] | None = None) -> list[dict[str, Any]]:
    known = {str(item).strip().lower() for item in (known_arches or KNOWN_ARCHES)}
    issues: list[dict[str, Any]] = []
    for row in rows:
        arch = _normalize(row.get("arch", "unknown")) or "unknown"
        path = _normalize(row.get("path", ""))
        if arch not in known:
            issues.append({"issue": "unknown-arch", "arch": arch, "path": row.get("path", "")})
            continue
        path_hits = [name for name, hints in ARCH_HINTS.items() if any(hint in path for hint in hints)]
        if path_hits and arch not in path_hits:
            issues.append({"issue": "path-arch-mismatch", "arch": arch, "path": row.get("path", ""), "path_hints": path_hits})
    return issues


def architecture_audit_report(rows: Iterable[Mapping[str, Any]], known_arches: Iterable[str] | None = None, dominance_threshold: float = 0.95, min_rows: int = 100) -> dict[str, Any]:
    rows = list(rows)
    distribution = inspect_architecture_distribution(rows)
    single_arch = detect_single_architecture_dataset(rows, dominance_threshold=dominance_threshold, min_rows=min_rows)
    suspicious = suspicious_architecture_assignments(rows, known_arches=known_arches)
    by_dataset: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        by_dataset[str(row.get("dataset", "unknown"))][_normalize(row.get("arch", "unknown")) or "unknown"] += 1
    return {
        "distribution": distribution,
        "single_architecture": single_arch,
        "suspicious_assignments": suspicious,
        "by_dataset": {name: dict(counter) for name, counter in by_dataset.items()},
    }
