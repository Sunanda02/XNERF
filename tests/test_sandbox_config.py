from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sandbox"))

from config import load_sandbox_config


def test_load_sandbox_config_falls_back_to_bundled_checkpoint(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "best.pt").write_bytes(b"checkpoint")
    (tmp_path / "config.yaml").write_text(
        "local_inference:\n"
        "  checkpoint: models/xnerf_local_inference.pt\n",
        encoding="utf-8",
    )

    config = load_sandbox_config(tmp_path / "config.yaml")

    assert config.checkpoint == Path("models/best.pt")