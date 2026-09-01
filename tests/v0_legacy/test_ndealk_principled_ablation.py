"""Ablation tests for ndealk principled site collapse + RDKit symmetry pooling.

Ndealk legacy-vs-default drift is **not** like epoxidation (small descriptor fix).
Worst cases reach ~0.96 max bond delta because production:

1. Collapses BondTD rows to one site key per symmetry class (``ndealk_site_mode``).
2. Pools the class mean onto every RDKit-symmetric sibling bond (``symmetry_group_mode``).

Either step alone can leave scores unchanged on a molecule; both together (plus
principled NRings on ~12 edge cases) reproduce production exactly.

See ``docs/legacy-vs-principled.md`` chapters 2–3.
"""

from __future__ import annotations

import pytest

from xenosite.predict import predict
from xenosite.predict.backends.onnx import OnnxBackend
from xenosite.predict.molecule import parse_smiles

from tests.support import (
    GOLDEN_PARAMETER,
    PARITY_ATOL,
    PRINCIPLED_PARAMETER,
    ROOT,
    golden_score_fields,
    load_golden_suite,
    onnx_weights_present,
    onnx_root,
)

from tests.v0_legacy.rdkit_equiv import assert_bond_scores_symmetric, bond_symmetry_groups

BACKEND = OnnxBackend(onnx_root())

NDEALK_DIVERGENT = "COc1ccc2nc(C)cc(NCCCN3CCOCC3)c2c1"
TOP_OUTLIER = "O=C1c2ccccc2-c2c1cccc2N1CCNCC1"

_PRINCIPLED_FROM_GOLDEN = {
    **GOLDEN_PARAMETER,
    "ndealk_site_mode": "principled",
    "symmetry_group_mode": "rdkit",
    "bond_nrings_mode": "principled",
}


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


def _ndealk_scores(smiles: str, *, parameter: dict | None = None) -> dict:
    if not onnx_weights_present("ndealk"):
        pytest.skip("no ONNX weights for ndealk")
    kwargs: dict = {"models": ["ndealk"], "backend": BACKEND}
    if parameter is not None:
        kwargs["_parameter"] = parameter
    mol = predict(smiles, **kwargs)
    assert mol.results
    return golden_score_fields(mol.results[0])


def _ndealk_smiles_from_golden() -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for row in load_golden_suite():
        if row.get("model") != "ndealk":
            continue
        smi = row.get("smiles")
        if not smi or smi in seen:
            continue
        seen.add(smi)
        out.append(smi)
    return out


def test_ndealk_principled_bundle_from_golden_equals_production() -> None:
    """All three ndealk flags flipped from golden reproduces ``predict()`` default."""
    default = _ndealk_scores(NDEALK_DIVERGENT)
    partial = _ndealk_scores(NDEALK_DIVERGENT, parameter=_PRINCIPLED_FROM_GOLDEN)
    assert _max_score_delta(default, partial) <= PARITY_ATOL


def test_ndealk_site_alone_does_not_explain_top_outlier() -> None:
    """Worst drifter: site collapse is a no-op; symmetry pooling moves the full gap."""
    legacy = _ndealk_scores(TOP_OUTLIER, parameter=GOLDEN_PARAMETER)
    site_only = _ndealk_scores(
        TOP_OUTLIER,
        parameter={**GOLDEN_PARAMETER, "ndealk_site_mode": "principled"},
    )
    sym_fix = _ndealk_scores(
        TOP_OUTLIER,
        parameter={
            **GOLDEN_PARAMETER,
            "ndealk_site_mode": "principled",
            "symmetry_group_mode": "rdkit",
        },
    )
    default = _ndealk_scores(TOP_OUTLIER)

    assert _max_score_delta(site_only, legacy) <= PARITY_ATOL
    assert _max_score_delta(sym_fix, legacy) > 0.5
    assert _max_score_delta(default, legacy) > 0.5
    assert abs(
        _max_score_delta(sym_fix, legacy) - _max_score_delta(default, legacy)
    ) <= max(PARITY_ATOL, 0.05 * _max_score_delta(default, legacy))


def test_ndealk_top_outlier_symmetry_pooling_activates_sibling_bonds() -> None:
    """RDKit pooling copies a high class score onto bonds that legacy kept at zero."""
    legacy_bond = _ndealk_scores(TOP_OUTLIER, parameter=GOLDEN_PARAMETER)["bond"]
    default_bond = _ndealk_scores(TOP_OUTLIER)["bond"]
    assert sum(1 for x in legacy_bond if float(x) > 0.001) < sum(
        1 for x in default_bond if float(x) > 0.001
    )
    new_sites = [
        (i, float(l), float(d))
        for i, (l, d) in enumerate(zip(legacy_bond, default_bond))
        if float(l) <= 0.001 and float(d) > 0.1
    ]
    assert len(new_sites) >= 2
    assert max(d for _i, _l, d in new_sites) > 0.8


def test_ndealk_production_bond_vector_is_rdkit_symmetric() -> None:
    """Production path always yields identical scores within RDKit bond classes."""
    rdmol, _ = parse_smiles(TOP_OUTLIER)
    groups = bond_symmetry_groups(rdmol)
    bond = _ndealk_scores(TOP_OUTLIER)["bond"]
    assert_bond_scores_symmetric(bond, groups)


def test_ndealk_bond_nrings_is_minor_on_golden_suite() -> None:
    """NRings explains at most ~12 molecules; site+sym explains almost all drifters."""
    smiles_list = _ndealk_smiles_from_golden()
    nrings_needed = 0
    for smiles in smiles_list:
        default = _ndealk_scores(smiles)
        site_sym = _ndealk_scores(
            smiles,
            parameter={
                **GOLDEN_PARAMETER,
                "ndealk_site_mode": "principled",
                "symmetry_group_mode": "rdkit",
            },
        )
        if _max_score_delta(default, site_sym) > PARITY_ATOL:
            nrings_needed += 1
    assert nrings_needed <= 15


@pytest.mark.full
def test_ndealk_principled_ablation_explains_drift_on_golden_suite() -> None:
    """Among drifters, site+sym+nrings from golden matches production for ≥90%."""
    smiles_list = _ndealk_smiles_from_golden()
    drifters = 0
    explained = 0
    sym_primary = 0
    for smiles in smiles_list:
        default = _ndealk_scores(smiles)
        legacy = _ndealk_scores(smiles, parameter=GOLDEN_PARAMETER)
        partial = _ndealk_scores(smiles, parameter=_PRINCIPLED_FROM_GOLDEN)
        site_only = _ndealk_scores(
            smiles,
            parameter={**GOLDEN_PARAMETER, "ndealk_site_mode": "principled"},
        )
        sym_fix = _ndealk_scores(
            smiles,
            parameter={
                **GOLDEN_PARAMETER,
                "ndealk_site_mode": "principled",
                "symmetry_group_mode": "rdkit",
            },
        )
        full = _max_score_delta(default, legacy)
        if full <= PARITY_ATOL:
            continue
        drifters += 1
        if _max_score_delta(site_only, legacy) <= PARITY_ATOL:
            sym_primary += 1
        if _max_score_delta(partial, default) <= PARITY_ATOL:
            explained += 1
        elif _max_score_delta(sym_fix, legacy) >= PARITY_ATOL:
            assert abs(_max_score_delta(sym_fix, legacy) - full) <= max(
                PARITY_ATOL, 0.05 * full
            )
    assert drifters >= 50
    assert explained / drifters >= 0.95
    assert sym_primary / drifters >= 0.25
