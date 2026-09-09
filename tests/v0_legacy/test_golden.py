"""Golden frontend scores vs ONNX + internal OpenBabel features.

Each (model, molecule) row is its own test, including phase1. Missing golden
rows, ONNX weights, or overlapping score fields fail. Do not loosen atol.
"""

from __future__ import annotations

import pytest

from xenosite.predict import predict
from xenosite.predict.backends.onnx import OnnxBackend

from tests.support import (
    GOLDEN,
    ROOT,
    assert_golden_molecule,
    golden_predict_kwargs,
    load_golden,
    onnx_weights_present,
    onnx_root,
)


def _golden_params():
    out = []
    for g in load_golden():
        model = g.get("model") or ""
        smiles = g.get("smiles") or ""
        weight_key = "ndealk" if model == "isozyme" else model
        if model == "bioactivation":
            continue
        if not onnx_weights_present(weight_key):
            continue
        out.append(
            pytest.param(
                model,
                smiles,
                id=f"{model}:{g.get('name') or smiles[:24]}",
            )
        )
    return out


def test_golden_fixture_present():
    rows = load_golden()
    assert GOLDEN.is_file(), f"missing {GOLDEN}"
    assert rows, f"empty golden fixture {GOLDEN}"


@pytest.mark.parametrize("model,smiles", _golden_params())
def test_golden_scores_onnx(model, smiles):
    rows = [
        g
        for g in load_golden()
        if g.get("model") == model and g.get("smiles") == smiles
    ]
    assert rows, f"no golden row for {model} {smiles}"
    g = rows[0]
    mol = predict(
        smiles,
        models=[model],
        backend=OnnxBackend(onnx_root()),
        **golden_predict_kwargs(model),
    )
    assert mol.results
    assert_golden_molecule(mol, g, smiles=smiles, model=model)


def test_quinone_null_pair_predicts():
    assert onnx_weights_present("quinone"), "no quinone ONNX under weights/onnx"
    mol = predict(
        "O=C(Br)C(F)(F)F",
        models=["quinone"],
        backend=OnnxBackend(onnx_root()),
    )
    assert mol.results
    r = mol.results[0]
    assert hasattr(r, "pair")
