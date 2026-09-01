"""Phase1 TF pickles converted to ONNX: numpy reconstruction vs onnxruntime."""

from __future__ import annotations

import sys

import numpy as np
import pytest

from xenosite.predict.backends.onnx import OnnxBackend
from xenosite.predict.errors import WeightsNotFound
from xenosite.predict import predict

from tests.support import ROOT, onnx_weights_present, onnx_root

sys.path.insert(0, str(ROOT / "tools"))
from convert_phase1 import (  # noqa: E402
    MOL_PICKLE,
    SITE_PICKLE,
    find_pickle,
    load_tf_pickle,
    mol_spec,
    numpy_forward_mol,
    numpy_forward_site,
    site_spec,
)


def test_phase1_onnx_heads_present():
    assert onnx_weights_present("phase1"), "no ONNX for phase1 (run make convert-onnx MODEL=phase1)"
    be = OnnxBackend(onnx_root())
    assert be.has_head("phase1", "site")
    assert be.has_head("phase1", "mol")


@pytest.mark.parametrize("head,pickle_name,spec_fn,fwd", [
    ("site", SITE_PICKLE, site_spec, numpy_forward_site),
    ("mol", MOL_PICKLE, mol_spec, numpy_forward_mol),
])
def test_phase1_onnx_matches_numpy_mlp(head, pickle_name, spec_fn, fwd):
    assert onnx_weights_present("phase1"), "no ONNX for phase1 (run make convert-onnx MODEL=phase1)"
    pkl = find_pickle(ROOT / "weights" / "legacy", pickle_name)
    if pkl is None:
        pytest.skip(
            f"missing {pickle_name} under weights/legacy "
            "(run make extract-weights)"
        )
    _g, _kw, params, trains = load_tf_pickle(pkl)
    spec = spec_fn(params, trains)
    rng = np.random.default_rng(20260830)
    x = rng.standard_normal((16, spec["I"])).astype(np.float32)
    y_ref = np.asarray(fwd(x, spec), dtype=np.float64)
    be = OnnxBackend(onnx_root())
    y = np.asarray(be.run_head("phase1", head, x), dtype=np.float64)
    np.testing.assert_allclose(y, y_ref, atol=1e-5, rtol=0)


def test_phase1_smiles_predicts():
    assert onnx_weights_present("phase1"), "no ONNX for phase1 (run make convert-onnx MODEL=phase1)"
    be = OnnxBackend(onnx_root())
    mol = predict("O=C(C)Oc1ccccc1C(=O)O", models=["phase1"], backend=be)
    assert len(mol.results) == 5
    assert all(r.model.startswith("phase1.") for r in mol.results)
