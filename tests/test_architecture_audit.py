from __future__ import annotations

from xnerf.datasets.audit import (
    architecture_audit_report,
    detect_single_architecture_dataset,
    inspect_architecture_distribution,
    suspicious_architecture_assignments,
)


def test_architecture_distribution_and_single_arch_detection():
    rows = [{"arch": "x86", "dataset": "a"} for _ in range(95)] + [{"arch": "x64", "dataset": "a"} for _ in range(5)]
    distribution = inspect_architecture_distribution(rows)
    report = detect_single_architecture_dataset(rows, dominance_threshold=0.9, min_rows=50)

    assert distribution["x86"] == 95
    assert report["is_single_architecture"] is True
    assert report["dominant_arch"] == "x86"


def test_architecture_audit_flags_mismatch_and_unknown():
    rows = [
        {"arch": "arm", "path": "sample_x86.exe", "dataset": "a"},
        {"arch": "weird", "path": "sample.bin", "dataset": "b"},
    ]
    issues = suspicious_architecture_assignments(rows)
    report = architecture_audit_report(rows)

    assert any(item["issue"] == "path-arch-mismatch" for item in issues)
    assert any(item["issue"] == "unknown-arch" for item in issues)
    assert report["distribution"]["arm"] == 1
