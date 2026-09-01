"""Production ONNX path: ``predict()`` defaults to principled site/OMP modes.

Golden parity tests (``test_golden*.py``, ``test_onnx_legacy_parity.py``) pass
``GOLDEN_PARAMETER`` (legacy ``ndealk_site_mode`` / ``quinone_omp_mode``) so scores
match regathered fixtures. This module verifies the public API without
``_parameter``: same as explicit principled, and intentionally different from
legacy on molecules where the modes diverge.
"""

from __future__ import annotations

import numpy as np
import pytest

from xenosite.predict import predict
from xenosite.predict.backends.onnx import OnnxBackend
from xenosite.predict.compare import assert_equiv_results

from tests.support import (
    GOLDEN_PARAMETER,
    PRINCIPLED_PARAMETER,
    ROOT,
    golden_score_fields,
    onnx_weights_present,
)

BACKEND = OnnxBackend(ROOT / "weights" / "onnx")

NAPHTHALENE = "c1ccc2ccccc2c1"
NDEALK_PRINCIPLED_FEWER = "COc1ccc2nc(C)cc(NCCCN3CCOCC3)c2c1"
NDEALK_CN1CCN = "CN1CCN(c2ccc3nc(-c4cccc(C(F)(F)F)c4)[nH]c3c2)CC1"


def _onnx_predict(smiles: str, model: str, **kwargs):
    if not onnx_weights_present("ndealk" if model == "isozyme" else model):
        pytest.skip(f"no ONNX weights for {model}")
    return predict(smiles, models=[model], backend=BACKEND, **kwargs)


def _scores(mol, model: str) -> dict:
    by_name = {r.model: r for r in mol.results}
    if model == "isozyme":
        hit = next((r for r in mol.results if r.model.startswith("isozyme.")), None)
        assert hit is not None
        return golden_score_fields(hit)
    if model == "reactivity":
        hit = next((r for r in mol.results if r.model.startswith("reactivity.")), None)
        assert hit is not None
        return golden_score_fields(hit)
    assert model in by_name, f"missing {model}; got {sorted(by_name)}"
    return golden_score_fields(by_name[model])


def _active_bond_count(scores: dict, *, threshold: float = 0.001) -> int:
    bond = scores.get("bond") or []
    return sum(1 for v in bond if float(v) > threshold)


def _max_score_delta(a: dict, b: dict) -> float:
    delta = 0.0
    for key in ("mol", "atom", "bond", "pair"):
        va, vb = a.get(key), b.get(key)
        if va is None or vb is None:
            continue
        if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
            delta = max(delta, abs(float(va) - float(vb)))
        elif isinstance(va, list) and isinstance(vb, list):
            for x, y in zip(va, vb):
                delta = max(delta, abs(float(x) - float(y)))
    return delta


@pytest.mark.parametrize(
    "model,smiles",
    [
        pytest.param("quinone", NAPHTHALENE, id="quinone:naphthalene"),
        pytest.param("ndealk", NDEALK_PRINCIPLED_FEWER, id="ndealk:coc1"),
        pytest.param("isozyme", NDEALK_CN1CCN, id="isozyme:cn1ccn"),
    ],
)
def test_predict_default_matches_explicit_principled(model, smiles):
    """No ``_parameter`` must equal explicit ``principled`` site/OMP modes."""
    default = _onnx_predict(smiles, model)
    explicit = _onnx_predict(
        smiles,
        model,
        _parameter=PRINCIPLED_PARAMETER,
    )
    assert default.results and explicit.results
    assert_equiv_results(_scores(default, model), _scores(explicit, model), atol=0.0)


def test_predict_does_not_require_parameter():
    """Public calls leave ``molecule._parameter`` empty unless set."""
    mol = _onnx_predict(NAPHTHALENE, "quinone")
    assert mol.results
    assert mol._parameter == {}


@pytest.mark.parametrize(
    "model,smiles",
    [
        pytest.param("quinone", NAPHTHALENE, id="quinone:naphthalene"),
        pytest.param("ndealk", NDEALK_PRINCIPLED_FEWER, id="ndealk:coc1"),
    ],
)
def test_predict_default_differs_from_legacy(model, smiles):
    """Principled production path is not bitwise-identical to golden legacy modes."""
    principled = _onnx_predict(smiles, model)
    legacy = _onnx_predict(smiles, model, _parameter=GOLDEN_PARAMETER)
    assert principled.results and legacy.results
    assert _max_score_delta(_scores(principled, model), _scores(legacy, model)) > 1e-6


def test_ndealk_principled_fewer_active_bonds_than_legacy():
    """Topo-GID dedup in principled mode drops orphan legacy site keys."""
    principled = _onnx_predict(NDEALK_PRINCIPLED_FEWER, "ndealk")
    legacy = _onnx_predict(NDEALK_PRINCIPLED_FEWER, "ndealk", _parameter=GOLDEN_PARAMETER)
    pri_n = _active_bond_count(_scores(principled, "ndealk"))
    leg_n = _active_bond_count(_scores(legacy, "ndealk"))
    assert pri_n < leg_n


def test_quinone_principled_scores_finite():
    mol = _onnx_predict(NAPHTHALENE, "quinone")
    scores = _scores(mol, "quinone")
    for key in ("mol", "atom", "bond", "pair"):
        val = scores.get(key)
        if val is None:
            continue
        if isinstance(val, (int, float)):
            assert np.isfinite(val)
        else:
            assert all(np.isfinite(float(x)) for x in val)


def test_principled_onnx_deterministic():
    a = _onnx_predict(NAPHTHALENE, "quinone")
    b = _onnx_predict(NAPHTHALENE, "quinone")
    assert_equiv_results(_scores(a, "quinone"), _scores(b, "quinone"), atol=0.0)


@pytest.mark.parametrize(
    "model",
    ["epoxidation", "reactivity", "ugt"],
)
def test_models_without_mode_flags_ignore_parameter(model):
    """Site/OMP modes only affect ndealk/isozyme/quinone; others are unchanged."""
    smiles = "CC(=O)Oc1ccccc1C(=O)O"
    if not onnx_weights_present(model):
        pytest.skip(f"no ONNX for {model}")
    plain = predict(smiles, models=[model], backend=BACKEND)
    with_param = predict(
        smiles,
        models=[model],
        backend=BACKEND,
        _parameter=GOLDEN_PARAMETER,
    )
    assert plain.results and with_param.results
    plain_by = {r.model: golden_score_fields(r) for r in plain.results}
    param_by = {r.model: golden_score_fields(r) for r in with_param.results}
    assert set(plain_by) == set(param_by)
    for name in sorted(plain_by):
        assert_equiv_results(plain_by[name], param_by[name], atol=0.0)
