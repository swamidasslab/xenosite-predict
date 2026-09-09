"""Bioactivation path/mol head parity on frozen feature matrices.

Sources (see ``tests/fixtures/bioactivation_parity.json``):
  - Training TSV feature columns (not TARGET — that is a 0/1 label)
  - Sibling doctest styrene path/mol descriptor rows

Reference ``y`` is py2 ``model.output`` via ``xenosite-predict-py2:dump``.
Regenerate with ``uv run python tools/gather_bioactivation_parity.py``.

Descriptor *builders* are not ported yet; styrene ``descriptor_rows`` freeze the
wider legacy feature contract for when the pipeline lands. This module checks
heads + that ONNX name tables ⊆ those frozen rows.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from xenosite.predict.backends.onnx import OnnxBackend
from xenosite.predict.v0_legacy.features import load_names

from tests.support import ROOT, onnx_weights_present, onnx_root

PARITY = ROOT / "tests" / "fixtures" / "bioactivation_parity.json"
ATOL = 1e-4


def _fixture() -> dict:
    assert PARITY.is_file(), f"missing {PARITY} (run tools/gather_bioactivation_parity.py)"
    return json.loads(PARITY.read_text())


@pytest.fixture(scope="module")
def parity() -> dict:
    return _fixture()


@pytest.mark.skipif(not onnx_weights_present("bioactivation"), reason="no bioactivation ONNX")
@pytest.mark.parametrize(
    "case",
    ["path_tsv", "mol_tsv", "styrene_path", "styrene_mol"],
)
def test_bioactivation_onnx_matches_py2_on_frozen_x(parity, case):
    rec = parity[case]
    head = rec["head"]
    names = list(rec["names"])
    assert names == list(load_names("bioactivation", head))
    x = np.asarray(rec["x"], dtype=np.float32)
    y_ref = np.asarray(rec["y"], dtype=np.float64).reshape(-1)
    y = np.asarray(
        OnnxBackend(onnx_root()).run_head("bioactivation", head, x),
        dtype=np.float64,
    ).reshape(-1)
    assert y.shape == y_ref.shape
    assert x.shape[0] >= (64 if case.endswith("_tsv") else 1)
    np.testing.assert_allclose(y, y_ref, atol=ATOL, rtol=0)


def test_parity_fixture_covers_trained_columns(parity):
    """TSV / styrene matrices use exactly the ONNX name-table columns (no TARGET)."""
    for case in ("path_tsv", "mol_tsv", "styrene_path", "styrene_mol"):
        names = parity[case]["names"]
        assert "TARGET" not in names
        assert "ID" not in names
        assert len(names) == 20


@pytest.mark.skipif(not onnx_weights_present("bioactivation"), reason="no bioactivation ONNX")
def test_bioactivation_name_tables_match_parity_fixture(parity):
    assert list(parity["path_tsv"]["names"]) == list(load_names("bioactivation", "path"))
    assert list(parity["mol_tsv"]["names"]) == list(load_names("bioactivation", "mol"))
    assert len(load_names("bioactivation", "path")) == 20
    assert len(load_names("bioactivation", "mol")) == 20


def test_styrene_path_descriptor_rows_cover_onnx_columns(parity):
    """Frozen sibling doctest path matrix includes every ONNX path feature."""
    rows = parity["styrene_path"]["descriptor_rows"]
    assert rows
    names = load_names("bioactivation", "path")
    missing = [n for n in names if n not in rows[0]]
    assert missing == [], missing
    # Wider legacy contract still present (not fed to ONNX).
    assert "Score__Formation" in rows[0]
    assert "Indicator__Epoxidation" in rows[0]
    assert "TARGET" in rows[0]


def test_styrene_mol_descriptor_rows_cover_onnx_columns(parity):
    rows = parity["styrene_mol"]["descriptor_rows"]
    assert rows
    names = load_names("bioactivation", "mol")
    missing = [n for n in names if n not in rows[0]]
    assert missing == [], missing
    assert all(k.startswith("PBS_") for k in names if "PBS" in k)


@pytest.mark.live
@pytest.mark.skipif(not onnx_weights_present("bioactivation"), reason="no bioactivation ONNX")
@pytest.mark.parametrize("case", ["styrene_path", "styrene_mol", "path_tsv"])
def test_bioactivation_live_nn_matches_onnx(legacy_api_url, parity, case):
    """Legacy Docker ``/nn/bioactivation/{path,mol}`` vs ONNX on frozen X."""
    from xenosite.predict.backends.legacy import LegacyTestBackend

    rec = parity[case]
    head = rec["head"]
    x = np.asarray(rec["x"], dtype=np.float32)
    # Keep live calls small.
    if x.shape[0] > 8:
        x = x[:8]
    be = LegacyTestBackend(legacy_api_url)
    try:
        raw = be.nn("bioactivation", head, x.tolist())
    except Exception as exc:
        pytest.skip(f"legacy /nn/bioactivation/{head} failed: {exc}")
    y0 = np.asarray(raw["y"] if isinstance(raw, dict) else raw, dtype=np.float64).reshape(-1)
    y1 = np.asarray(
        OnnxBackend(onnx_root()).run_head("bioactivation", head, x),
        dtype=np.float64,
    ).reshape(-1)
    n = min(y0.size, y1.size)
    np.testing.assert_allclose(y0[:n], y1[:n], atol=ATOL, rtol=0)
