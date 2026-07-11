from __future__ import annotations

from xnerf.datasets.validation import (
    family_names_from_metadata,
    validate_checkpoint_family_metadata,
    validate_family_batch,
    validate_family_rows,
    validate_family_vocab,
)


def test_validation_detects_missing_vocab_entries():
    rows = [
        {"family": "Trojan", "label": 1},
        {"family": "Spy", "label": 1},
        {"family": "Ben0", "label": 0},
    ]
    issues = validate_family_rows(rows, family_vocab=["benign", "Trojan"])
    assert any("missing vocab entry" in issue for issue in issues)


def test_checkpoint_metadata_validation_handles_list_and_dict_forms():
    meta = {"family_names": ["benign", "Trojan"]}
    issues = validate_checkpoint_family_metadata(meta, family_vocab=["benign", "Trojan"])
    assert issues == []
    assert family_names_from_metadata({"id_to_family": {0: "benign", 1: "Trojan"}}) == ["benign", "Trojan"]


def test_validate_family_batch_requires_family_label():
    validate_family_batch({"family_label": [0, 1]})
    try:
        validate_family_batch({"label": [0, 1]})
    except RuntimeError as exc:
        assert "family_label" in str(exc)
    else:
        raise AssertionError("expected RuntimeError")
