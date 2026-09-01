"""327-molecule golden score parity (``golden_descriptor_suite.json``).

Rows are captured from ONNX with ``GOLDEN_PARAMETER`` (legacy site/OMP modes).
Regather via ``make regather-golden-onnx`` after descriptor or mapping fixes.
"""

from __future__ import annotations

import pytest

from xenosite.predict import predict
from xenosite.predict.backends.onnx import OnnxBackend

from tests.support import (
    GOLDEN_PARAMETER,
    GOLDEN_SUITE,
    ROOT,
    assert_golden_molecule,
    load_descriptor_smiles,
    load_golden_suite,
    onnx_weights_present,
    parity_atol,
)


def _suite_params():
    params = []
    seen: set[tuple[str, str]] = set()
    for g in load_golden_suite(merge_smoke=True):
        model = g.get("model") or ""
        smiles = g.get("smiles") or ""
        if model == "bioactivation":
            continue
        key = (model, smiles)
        if key in seen:
            continue
        seen.add(key)
        weight_key = "ndealk" if model == "isozyme" else model
        if not onnx_weights_present(weight_key):
            continue
        label = g.get("name") or smiles[:24]
        params.append(pytest.param(model, smiles, id=f"{model}:{label}"))
    return params


def test_golden_suite_fixture_growing():
    """Suite file grows as gather_golden_suite runs; smoke golden always available."""
    rows = load_golden_suite(merge_smoke=False)
    desc = load_descriptor_smiles()
    assert desc, "missing descriptor_smiles.json"
    smoke = load_golden_suite(merge_smoke=True)
    assert smoke, "need golden_smiles.json or golden_descriptor_suite.json"
    if not rows:
        pytest.skip(
            f"no rows in {GOLDEN_SUITE}; run tools/gather_golden_suite.py (smoke golden only for now)"
        )
    assert len(rows) >= len(desc) * 6, (
        f"expected ~{len(desc)*6} suite rows (6 models × {len(desc)} SMILES), got {len(rows)}"
    )


@pytest.mark.parametrize("model,smiles", _suite_params())
def test_golden_suite_scores_onnx(model, smiles):
    rows = [
        g
        for g in load_golden_suite(merge_smoke=True)
        if g.get("model") == model and g.get("smiles") == smiles
    ]
    assert rows, f"no golden suite row for {model} {smiles}"
    g = rows[0]
    kwargs = {}
    if model in ("ndealk", "isozyme", "quinone"):
        kwargs["_parameter"] = GOLDEN_PARAMETER
    mol = predict(
        smiles,
        models=[model],
        backend=OnnxBackend(ROOT / "weights" / "onnx"),
        **kwargs,
    )
    assert mol.results
    assert_golden_molecule(mol, g, smiles=smiles, model=model)


def test_suite_omp_tolerance_is_standard():
    """Quinone legacy OMP descriptors match regathered dumps at ``PARITY_ATOL``."""
    from xenosite.predict.molecule import canonicalize_smiles
    from tests.support import PARITY_ATOL, descriptor_mismatch_columns

    smi = canonicalize_smiles(
        "CN1C(C(=O)NC2=NC=CS2)=C(O)C2=CC=CC=C2S1(=O)=O"
    )
    assert not descriptor_mismatch_columns(smi, "quinone")
    assert parity_atol(smi, "quinone") == PARITY_ATOL
