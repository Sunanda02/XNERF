from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sandbox"))

from inference import family_name


def test_family_name_uses_checkpoint_metadata_when_present():
    assert family_name(2, {"family_names": ["Trojan", "Worm", "Botnet"]}) == "Botnet"


def test_family_name_uses_sidecar_vocab_when_checkpoint_metadata_is_missing(tmp_path: Path, monkeypatch):
    processed = tmp_path / "data" / "processed"
    processed.mkdir(parents=True)
    (processed / "family_vocab.json").write_text(
        "{\n"
        '  "family_names": ["Trojan", "Worm", "Botnet"]\n'
        "}\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    assert family_name(1, {}) == "Worm"