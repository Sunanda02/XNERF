from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, Mapping

import yaml


def load_config(path: str | Path = "config.yaml") -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def instantiate(spec: Mapping[str, Any], **overrides: Any) -> Any:
    """Dependency-injection helper.

    spec:
        target: package.module.ClassName
        params: optional constructor kwargs
    """

    target = spec["target"]
    module_name, class_name = target.rsplit(".", 1)
    cls = getattr(importlib.import_module(module_name), class_name)
    params = dict(spec.get("params", {}))
    params.update(overrides)
    return cls(**params)

