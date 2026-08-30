"""Replay dumped py2 NN outputs against ONNX (no live Docker)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from xenosite.predict.backends.onnx import OnnxBackend
from xenosite.predict.features import bond_rows, load_names, matrix_from_rows
from xenosite.predict.molecule import parse_smiles

from tests.support import ROOT, onnx_weights_present

REPLAY = ROOT / "tests" / "fixtures" / "random_vectors.json"
ASPIRIN = "O=C(C)Oc1ccccc1C(=O)O"


def _replay() -> dict:
    if not REPLAY.is_file() or REPLAY.stat().st_size == 0:
        return {}
    data = json.loads(REPLAY.read_text())
    return {k: v for k, v in data.items() if isinstance(v, dict) and "x" in v and "y" in v}


@pytest.mark.parametrize("key", sorted(_replay()) or ["_none"])
def test_onnx_matches_dumped_py2_nn(key):
    replay = _replay()
    if not replay:
        pytest.skip("no dumped random-vector fixtures (run make convert-onnx)")
    rec = replay[key]
    model, head = key.split("_", 1)
    if not onnx_weights_present(model):
        pytest.skip(f"no ONNX for {model}")
    be = OnnxBackend(ROOT / "weights" / "onnx")
    x = np.asarray(rec["x"], dtype=np.float32)
    y_ref = np.asarray(rec["y"], dtype=np.float64)
    y = np.asarray(be.run_head(model, head, x), dtype=np.float64)
    n = min(y.size, y_ref.size)
    np.testing.assert_allclose(y.reshape(-1)[:n], y_ref.reshape(-1)[:n], atol=1e-4, rtol=0)


def test_epoxidation_bond_names_cover_matrix():
    names = load_names("epoxidation", "bond")
    if not names:
        pytest.skip("no committed epoxidation bond names")
    mol, _ = parse_smiles(ASPIRIN)
    rows = bond_rows(mol, original_atom_ordering=True)
    missing = [n for n in names if n not in rows[0]]
    assert missing == [], missing[:12]
    x, _ = matrix_from_rows(rows, names)
    assert x.shape == (len(rows), len(names))
