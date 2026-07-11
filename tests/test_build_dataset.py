from __future__ import annotations

import json
from pathlib import Path

from xnerf.datasets.build_dataset import build_manifest
from xnerf.utils.io import read_jsonl


def test_build_manifest_enriches_cape_json(tmp_path: Path):
    report_dir = tmp_path / "raw" / "cape" / "reports" / "malicious"
    report_dir.mkdir(parents=True)
    report = {
        "behavior": {
            "processes": [{"calls": [{"api": "CreateFileW"}]}],
            "apistats": {"1": {"RegSetValueExW": 3}},
        },
        "network": {"dns": [{"request": "evil.test"}]},
    }
    (report_dir / "report.json").write_text(json.dumps(report), encoding="utf-8")
    out = tmp_path / "processed" / "manifest.jsonl"

    build_manifest(tmp_path, out, make_splits=True)

    rows = read_jsonl(out)
    assert len(rows) == 1
    assert rows[0]["label"] == 1
    assert rows[0]["api_call_count"] == 4
    assert rows[0]["network_event_count"] == 1
    assert rows[0]["api_ids"]
    assert (tmp_path / "processed" / "train_manifest.jsonl").exists()

