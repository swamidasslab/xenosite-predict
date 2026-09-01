"""327-molecule golden score parity (``golden_descriptor_suite.json``).

Rows are captured from legacy-test-api via ``tools/gather_golden_suite.py``.
Quinone OMP-only descriptor drift uses a looser score tolerance (``PARITY_ATOL_OMP``);
legacy scores remain authoritative — we do not refresh golden from ONNX.
"""

from __future__ import annotations

import pytest

from xenosite.predict import predict
from xenosite.predict.backends.onnx import OnnxBackend

from tests.support import (
    GOLDEN_NDEALK_PARAMETER,
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
    if model in ("ndealk", "isozyme"):
        kwargs["_parameter"] = GOLDEN_NDEALK_PARAMETER
    mol = predict(
        smiles,
        models=[model],
        backend=OnnxBackend(ROOT / "weights" / "onnx"),
        **kwargs,
    )
    assert mol.results
    assert_golden_molecule(mol, g, smiles=smiles, model=model)


def test_suite_omp_tolerance_covers_known_quinone():
    """Sudoxicam quinone uses OMP looser band when OMP columns drift."""
    from xenosite.predict.molecule import canonicalize_smiles
    from tests.support import PARITY_ATOL, PARITY_ATOL_OMP, descriptor_mismatch_columns

    smi = canonicalize_smiles(
        "CN1C(C(=O)NC2=NC=CS2)=C(O)C2=CC=CC=C2S1(=O)=O"
    )
    cols = descriptor_mismatch_columns(smi, "quinone")
    ompish = cols and all("Ortho_" in c or "Meta_" in c or "Para_" in c for c in cols)
    if ompish:
        assert parity_atol(smi, "quinone") == PARITY_ATOL_OMP
    else:
        assert parity_atol(smi, "quinone") == PARITY_ATOL
