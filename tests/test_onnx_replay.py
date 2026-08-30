"""Replay dumped py2 NN outputs against ONNX, plus Hypothesis random-vector fuzz.

The convert dump (`random_vectors.json`) is a regression: one seeded matrix per
head with the Python 2 net's ``y``. Hypothesis draws extra finite matrices and
checks ONNX stays finite at the declared shape. Live Docker parity is in
``test_live.py``.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from xenosite.predict.backends.onnx import OnnxBackend
from xenosite.predict.features import bond_rows, load_names, matrix_from_rows
from xenosite.predict.molecule import parse_smiles

from tests.strategies import feature_matrix
from tests.support import (
    ROOT,
    list_onnx_heads,
    onnx_io_dims,
    onnx_weights_present,
)

REPLAY = ROOT / "tests" / "fixtures" / "random_vectors.json"
ASPIRIN = "O=C(C)Oc1ccccc1C(=O)O"


def _replay() -> dict:
    if not REPLAY.is_file() or REPLAY.stat().st_size == 0:
        return {}
    data = json.loads(REPLAY.read_text())
    return {k: v for k, v in data.items() if isinstance(v, dict) and "x" in v and "y" in v}


def test_onnx_replay_fixture_present():
    assert _replay(), "missing tests/fixtures/random_vectors.json (run make convert-onnx)"


@pytest.mark.parametrize("key", sorted(_replay()))
def test_onnx_matches_dumped_py2_nn(key):
    rec = _replay()[key]
    model, head = key.split("_", 1)
    assert onnx_weights_present(model), f"no ONNX for {model} under weights/onnx"
    be = OnnxBackend(ROOT / "weights" / "onnx")
    x = np.asarray(rec["x"], dtype=np.float32)
    y_ref = np.asarray(rec["y"], dtype=np.float64)
    y = np.asarray(be.run_head(model, head, x), dtype=np.float64)
    n = min(y.size, y_ref.size)
    np.testing.assert_allclose(y.reshape(-1)[:n], y_ref.reshape(-1)[:n], atol=1e-4, rtol=0)


def test_epoxidation_bond_names_cover_matrix():
    names = load_names("epoxidation", "bond")
    assert names, "no committed epoxidation bond names"
    mol, _ = parse_smiles(ASPIRIN)
    rows = bond_rows(mol, original_atom_ordering=True)
    missing = [n for n in names if n not in rows[0]]
    assert missing == [], missing[:12]
    x, _ = matrix_from_rows(rows, names)
    assert x.shape == (len(rows), len(names))


def test_onnx_heads_present():
    assert list_onnx_heads(), "no ONNX weights under weights/onnx"


@pytest.mark.parametrize("model,head", list_onnx_heads())
@settings(max_examples=8, deadline=None)
@given(data=st.data())
def test_onnx_random_matrix_finite(model, head, data):
    """Random finite inputs of the declared feature width stay finite."""
    dims = onnx_io_dims(model, head)
    assert dims is not None, f"no I/O dims for {model}/{head}"
    n_in, n_out = dims
    x = data.draw(feature_matrix(n_in))
    be = OnnxBackend(ROOT / "weights" / "onnx")
    y = np.asarray(be.run_head(model, head, x), dtype=np.float64)
    assert np.all(np.isfinite(y))
    assert y.shape[0] == x.shape[0]
    if y.ndim == 2:
        assert y.shape[1] == n_out or y.size == x.shape[0] * n_out
    else:
        assert y.size == x.shape[0] * n_out
