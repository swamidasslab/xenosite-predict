"""Golden frontend scores vs ONNX + internal OpenBabel features. Xfail until dumps match."""

from __future__ import annotations

import pytest

from xenosite.predict import WeightsNotFound, predict
from xenosite.predict.backends.onnx import OnnxBackend
from xenosite.predict.compare import assert_equiv_results
from xenosite.predict.errors import ModelNotAvailable, OpenBabelNotAvailable

from tests.support import ROOT, load_golden, onnx_weights_present

MODELS = ("epoxidation", "quinone", "reactivity", "ugt", "ndealk", "isozyme")


@pytest.mark.parametrize("model", MODELS)
def test_golden_scores_onnx(model):
    rows = [g for g in load_golden() if g.get("model") == model]
    if not rows:
        pytest.skip(f"no golden rows for {model}")
    key = "ndealk" if model == "isozyme" else model
    if not onnx_weights_present(key):
        pytest.skip(f"no ONNX for {key}")
    be = OnnxBackend(ROOT / "weights" / "onnx")
    pytest.xfail(
        "internal OpenBabel features are not yet verified against golden frontend "
        "scores at atol 1e-4. Un-xfail after dump-vs-feature tests pass and ONNX "
        "is fed from those rows. See docs/vendored-diffs.md."
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
    try:
        mol = predict("O=C(Br)C(F)(F)F", models=["quinone"], backend=be)
    except OpenBabelNotAvailable as exc:
        pytest.skip(str(exc))
    assert mol.results
    r = mol.results[0]
    assert hasattr(r, "pair")
