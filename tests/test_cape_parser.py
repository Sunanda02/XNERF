from __future__ import annotations

import json
import zipfile
from pathlib import Path

from xnerf.sandbox.cape_parser import iter_cape_reports, parse_cape_report


def sample_report():
    return {
        "info": {"score": 7.5},
        "target": {"file": {"name": "sample.apk"}},
        "behavior": {
            "processes": [
                {
                    "pid": 123,
                    "process_name": "malware",
                    "calls": [{"api": "CreateFileW"}, {"api": "RegSetValueExW"}, {}],
                }
            ],
            "apistats": {"123": {"CreateFileW": 10, "InternetOpenA": 2}},
        },
        "network": {
            "dns": [{"request": "bad.example"}],
            "http": [{"uri": "http://bad.example/a"}],
        },
        "signatures": [{"name": "persistence"}],
    }


def test_parse_json_report(tmp_path: Path):
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(sample_report()), encoding="utf-8")

    parsed = parse_cape_report(report_path)

    assert "CreateFileW" in parsed["api_calls"]
    assert "InternetOpenA" in parsed["api_calls"]
    assert parsed["api_calls"].count("CreateFileW") == 6
    assert parsed["network_events"][0]["type"] == "dns"
    assert parsed["memory_events"][0]["type"] == "signatures"
    assert parsed["summary"]["score"] == 7.5


def test_parse_zip_member(tmp_path: Path):
    zip_path = tmp_path / "reports.zip"
    with zipfile.ZipFile(zip_path, "w") as z:
        z.writestr("nested/report.json", json.dumps(sample_report()))

    parsed = parse_cape_report(zip_path, "nested/report.json")

    assert parsed["summary"]["target"] == "sample.apk"
    assert len(list(iter_cape_reports(zip_path))) == 1

