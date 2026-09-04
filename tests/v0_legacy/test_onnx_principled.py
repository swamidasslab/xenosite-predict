"""Production ONNX path: ``predict()`` defaults to scoring version ``"1"``.

Golden parity tests (``test_golden*.py``, ``test_onnx_legacy_parity.py``) pass
``GOLDEN_PARAMETER`` or ``models=[(name, "0")]`` so scores match regathered
fixtures. This module verifies the public API at version ``"1"``: same as
explicit principled, and intentionally different from version ``"0"`` on
molecules where the modes diverge.

For a human-readable walkthrough of each flag, see
``tests/v0_legacy/test_legacy_vs_principled_guide.py`` and ``docs/legacy-vs-principled.md``.
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
    onnx_root,
)

BACKEND = OnnxBackend(onnx_root())

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
    """Public v1 calls leave ``molecule._parameter`` empty unless set."""
    mol = _onnx_predict(NAPHTHALENE, "quinone")
    assert mol.results
    assert mol._parameter == {}
    assert all(r.version == "1" for r in mol.results)


def test_scoring_version_0_matches_golden_parameter():
    """``models=[(name, "0")]`` applies legacy params (HTTP ``/v0``)."""
    if not onnx_weights_present("quinone"):
        pytest.skip("no ONNX weights for quinone")
    v0 = predict(NAPHTHALENE, models=[("quinone", "0")], backend=BACKEND)
    via_param = _onnx_predict(NAPHTHALENE, "quinone", _parameter=GOLDEN_PARAMETER)
    assert all(r.version == "0" for r in v0.results)
    assert v0._parameter["quinone_omp_mode"] == "legacy"
    assert_equiv_results(_scores(v0, "quinone"), _scores(via_param, "quinone"), atol=0.0)


def test_scoring_version_1_matches_default():
    """``models=[(name, "1")]`` is the same as omitting the version."""
    default = _onnx_predict(NAPHTHALENE, "quinone")
    v1 = predict(NAPHTHALENE, models=[("quinone", "1")], backend=BACKEND)
    assert all(r.version == "1" for r in default.results)
    assert all(r.version == "1" for r in v1.results)
    assert_equiv_results(_scores(default, "quinone"), _scores(v1, "quinone"), atol=0.0)


@pytest.mark.parametrize(
    "model,smiles",
    [
        pytest.param("quinone", NAPHTHALENE, id="quinone:naphthalene"),
        pytest.param("ndealk", NDEALK_PRINCIPLED_FEWER, id="ndealk:coc1"),
        pytest.param(
            "epoxidation",
            "c1ccc2c(c1)oc1ccccc12",
            id="epoxidation:dibenzofuran",
        ),
    ],
)
def test_predict_default_differs_from_legacy(model, smiles):
    """Principled production path is not bitwise-identical to golden legacy modes."""
    principled = _onnx_predict(smiles, model)
    legacy = _onnx_predict(smiles, model, _parameter=GOLDEN_PARAMETER)
    assert principled.results and legacy.results
    assert _max_score_delta(_scores(principled, model), _scores(legacy, model)) > 1e-6


def test_ndealk_rdkit_symmetry_pooling():
    """Production path pools one class score to all RDKit-symmetric bonds."""
    from xenosite.predict.molecule import parse_smiles
    from tests.v0_legacy.rdkit_equiv import assert_bond_scores_symmetric, bond_symmetry_groups

    rdmol, _ = parse_smiles(NDEALK_PRINCIPLED_FEWER)
    mol = _onnx_predict(NDEALK_PRINCIPLED_FEWER, "ndealk")
    groups = bond_symmetry_groups(rdmol)
    assert groups
    assert_bond_scores_symmetric(mol.results[0].bond, groups)


def test_epoxidation_rdkit_symmetry_pooling():
    """Production epoxidation pools within RDKit bond classes."""
    from xenosite.predict.molecule import parse_smiles
    from tests.v0_legacy.rdkit_equiv import assert_bond_scores_symmetric, bond_symmetry_groups

    smiles = "c1ccc2c(c1)oc1ccccc12"
    rdmol, _ = parse_smiles(smiles)
    mol = _onnx_predict(smiles, "epoxidation")
    groups = bond_symmetry_groups(rdmol)
    assert groups
    assert_bond_scores_symmetric(mol.results[0].bond, groups)


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
    ["reactivity", "ugt"],
)
def test_models_without_legacy_flags_ignore_golden_parameter(model):
    """Reactivity/UGT scores are unchanged by golden site/OMP/symmetry flags on aspirin."""
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
