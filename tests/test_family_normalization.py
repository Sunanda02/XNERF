from __future__ import annotations

from xnerf.datasets.family_cleaning import (
    FamilyNormalizationRules,
    build_family_vocabulary,
    is_placeholder_family,
    normalize_family_name,
)


def test_family_normalization_maps_common_aliases():
    assert normalize_family_name("Spy") == "Spyware"
    assert normalize_family_name("Worms") == "Worm"
    assert normalize_family_name("Trojan") == "Trojan"
    assert normalize_family_name("Ben0", label=0) == "benign"


def test_placeholder_detection_flags_dataset_names():
    assert is_placeholder_family("train_ember_2018_v2_features")
    assert is_placeholder_family("public_small_reports")
    assert not is_placeholder_family("Trojan")


def test_family_vocab_is_deterministic_and_stable():
    rows = [
        {"family": "Worms", "label": 1},
        {"family": "Spy", "label": 1},
        {"family": "Trojan", "label": 1},
        {"family": "Ben0", "label": 0},
    ]
    vocab = build_family_vocabulary(rows, rules=FamilyNormalizationRules())
    assert vocab == ["benign", "Spyware", "Trojan", "Worm"]
