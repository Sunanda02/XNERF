from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any


def _load_report(path: str | Path, file_name: str | None = None) -> dict[str, Any]:
    path = Path(path)
    if file_name:
        with zipfile.ZipFile(path, "r") as z:
            with z.open(file_name) as f:
                return json.load(f)
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def parse_cape_report(path: str | Path, file_name: str | None = None) -> dict:
    """Parse CAPE/Avast-style sandbox JSON reports.

    Inputs:
        path: .json report path, or .zip path when file_name is provided.
        file_name: optional JSON member inside zip.
    Outputs:
        {
          "api_calls": list[str],
          "network_events": list[dict],
          "memory_events": list[dict],
          "process_events": list[dict],
          "summary": dict
        }
    Usage:
        parse_cape_report("report.json")
        parse_cape_report("reports.zip", "sample/report.json")
    """

    report = _load_report(path, file_name=file_name)
    behavior = report.get("behavior", {})
    api_calls = []
    process_events = []

    for proc in behavior.get("processes", []):
        proc_name = proc.get("process_name") or proc.get("name") or proc.get("module_path") or "unknown_process"
        pid = proc.get("pid") or proc.get("process_id")
        process_events.append({"type": "process", "pid": pid, "name": proc_name})
        for call in proc.get("calls", []):
            api = call.get("api")
            if api:
                api_calls.append(api)

    apistats = behavior.get("apistats", {})
    for _pid, data in apistats.items():
        if not isinstance(data, dict):
            continue
        for api, count in data.items():
            try:
                repeats = min(int(count), 5)
            except (TypeError, ValueError):
                repeats = 1
            api_calls.extend([api] * repeats)

    summary = behavior.get("summary", {})
    if isinstance(summary, dict):
        for api in summary.get("resolved_apis", []) or []:
            if api:
                api_calls.append(str(api))

    network = report.get("network", {})
    network_events = []
    for dns in network.get("dns", []):
        network_events.append({"type": "dns", "value": dns.get("request") or dns.get("hostname") or dns})
    for http in network.get("http", []):
        network_events.append({"type": "http", "value": http.get("uri") or http.get("host") or http})
    for key in ("tcp", "udp", "icmp", "smtp", "irc"):
        for item in network.get(key, []):
            network_events.append({"type": key, "value": item})
    if isinstance(summary, dict):
        for key in ("executed_commands",):
            for item in summary.get(key, []) or []:
                network_events.append({"type": key, "value": item})

    memory_events = []
    for key in ("memory", "procmemory", "dropped", "signatures"):
        values = report.get(key, [])
        if isinstance(values, dict):
            values = values.values()
        for item in values or []:
            memory_events.append({"type": key, "value": item})
    if isinstance(summary, dict):
        for key in (
            "keys",
            "read_keys",
            "write_keys",
            "delete_keys",
            "files",
            "read_files",
            "write_files",
            "delete_files",
            "mutexes",
            "started_services",
            "created_services",
        ):
            for item in summary.get(key, []) or []:
                memory_events.append({"type": key, "value": item})

    target = report.get("target", {})
    info = report.get("info", {})
    return {
        "api_calls": api_calls,
        "network_events": network_events,
        "memory_events": memory_events,
        "process_events": process_events,
        "summary": {
            "target": target.get("file", {}).get("name") if isinstance(target.get("file"), dict) else target,
            "score": info.get("score") or report.get("score"),
            "package": target.get("package") if isinstance(target, dict) else None,
        },
    }


def iter_cape_reports(path: str | Path):
    """Yield (source, parsed_report) for a JSON report or ZIP of JSON reports."""

    path = Path(path)
    if path.suffix.lower() == ".json":
        yield str(path), parse_cape_report(path)
        return
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path, "r") as z:
            for name in z.namelist():
                if name.lower().endswith(".json") and not name.endswith("/"):
                    yield f"{path}!{name}", parse_cape_report(path, name)
