"""Inlined ONNX column-order tables (no JSON on the inference path)."""

from __future__ import annotations

from pathlib import Path

import pytest

from xenosite.predict.features import load_names
from xenosite.predict.v1.features.name_tables import TABLES

FEATURES_DIR = (
    Path(__file__).resolve().parents[1] / "src/xenosite/predict/v1/features"
)

# Trained input widths (must stay aligned with ONNX graphs).
_EXPECTED_LEN = {
    ("bioactivation", "mol"): 20,
    ("bioactivation", "path"): 20,
    ("epoxidation", "bond"): 379,
    ("epoxidation", "mol"): 35,
    ("ndealk", "bond"): 386,
    ("phase1", "mol"): 41,
    ("phase1", "site"): 404,
    ("quinone", "atom"): 390,
    ("quinone", "mol"): 21,
    ("quinone", "pair"): 4,
    ("reactivity", "atom"): 209,
    ("reactivity", "mol"): 35,
    ("ugt", "atom"): 82,
}


def test_no_feature_name_json_in_package():
    leftovers = sorted(FEATURES_DIR.glob("*_names.json"))
    assert leftovers == [], f"JSON name files should be gone: {leftovers}"


def test_tables_cover_every_onnx_head():
    assert set(TABLES) == set(_EXPECTED_LEN)
    for key, n in _EXPECTED_LEN.items():
        names = TABLES[key]
        assert isinstance(names, tuple)
        assert len(names) == n
        assert all(isinstance(s, str) and s for s in names)
        assert len(set(names)) == n  # unique columns


def test_load_names_returns_copy_not_json():
    got = load_names("ugt", "atom")
    assert got is not None
    assert got[0] == "normalized_chance"
    assert got[-1] == "abonds"
    # Mutating the returned list must not change the frozen table.
    got[0] = "mutated"
    assert load_names("ugt", "atom")[0] == "normalized_chance"


def test_load_names_unknown_is_none():
    assert load_names("not-a-model", "atom") is None
    assert load_names("ugt", "nope") is None


@pytest.mark.parametrize(
    "model,head,first,last",
    [
        ("epoxidation", "bond", "Atom1_NC_sp1_0", None),
        ("quinone", "pair", "Atom1_Pred", "AtomPair__Distance_Is_Odd"),
        ("ndealk", "bond", "otherN_C", None),
        ("phase1", "site", "Atom1_NC_sp1_0", None),
    ],
)
def test_load_names_known_endpoints(model, head, first, last):
    names = load_names(model, head)
    assert names
    assert names[0] == first
    if last is not None:
        assert names[-1] == last
