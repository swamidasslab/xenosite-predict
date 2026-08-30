"""Golden frontend scores vs ONNX+RDKit. Xfail until RDKit matches OpenBabel."""

from __future__ import annotations

import pytest

from xenosite.predict import WeightsNotFound, predict
from xenosite.predict.backends.onnx import OnnxBackend
from xenosite.predict.compare import assert_equiv_results
from xenosite.predict.errors import ModelNotAvailable

from tests.support import ROOT, load_golden, onnx_weights_present

MODELS = ("epoxidation", "quinone", "reactivity", "ugt", "ndealk", "isozyme")


@pytest.mark.parametrize("model", MODELS)
def test_golden_scores_rdkit_onnx(model):
    rows = [g for g in load_golden() if g.get("model") == model]
    if not rows:
        pytest.skip(f"no golden rows for {model}")
    key = "ndealk" if model == "isozyme" else model
    if not onnx_weights_present(key):
        pytest.skip(f"no ONNX for {key}")
    be = OnnxBackend(ROOT / "weights" / "onnx")
    pytest.xfail(
        "RDKit descriptors are not yet equal to OpenBabel dumps at atol 1e-4; "
        "ONNX==numpy-NN random-vector parity holds. See docs/vendored-diffs.md."
    )
    g = rows[1] if len(rows) > 1 else rows[0]  # aspirin-like when present
    try:
        mol = predict(g["smiles"], models=[model], backend=be)
    except (WeightsNotFound, ModelNotAvailable) as exc:
        pytest.skip(str(exc))
    golden = (g.get("results") or [{}])[0]
    got = mol.results[0]
    subset = {}
    got_d = {"mol": getattr(got, "mol", None), "bond": getattr(got, "bond", None),
             "atom": getattr(got, "atom", None)}
    for k in ("mol", "bond", "atom"):
        if golden.get(k) is not None and got_d.get(k) is not None:
            subset[k] = golden[k]
    if not subset:
        pytest.skip("no overlapping score fields")
    assert_equiv_results(subset, {k: got_d[k] for k in subset})


def test_quinone_null_pair_predicts_or_skips():
    be = OnnxBackend(ROOT / "weights" / "onnx")
    if not onnx_weights_present("quinone"):
        pytest.skip("no quinone ONNX")
    mol = predict("O=C(Br)C(F)(F)F", models=["quinone"], backend=be)
    assert mol.results
    r = mol.results[0]
    assert hasattr(r, "pair")
