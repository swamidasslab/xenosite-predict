"""Ablation tests for ``bond_nrings_mode`` semantics and score attribution.

When epoxidation legacy-vs-default drift is attributed to NRings, these tests
verify we mean what we document:

- ``Atom1_NRings`` / ``Atom2_NRings`` count rings containing each **endpoint
  atom** (legacy: DFS back-edge cycles; principled: RDKit ``NumAtomRings``).
- Flipping only ``bond_nrings_mode`` changes **only** those two ONNX columns on
  fused polycyclics; every other bond descriptor stays identical.
- For epoxidation, that single-flag flip from golden legacy explains essentially
  the full production-vs-legacy score gap (symmetry pooling is a no-op here).

See ``docs/legacy-vs-principled.md`` and ``tests/test_legacy_vs_principled_guide.py``.
"""

from __future__ import annotations

import numpy as np
import pytest

from xenosite.predict import predict
from xenosite.predict.backends.onnx import OnnxBackend
from xenosite.predict.features import bond_rows, load_names
from xenosite.predict.features.molgraph import MolGraph
from xenosite.predict.molecule import parse_smiles

from tests.support import (
    GOLDEN_PARAMETER,
    PARITY_ATOL,
    ROOT,
    golden_score_fields,
    load_golden_suite,
    onnx_weights_present,
    rows_for_model,
    onnx_root,
)

BACKEND = OnnxBackend(onnx_root())

NRINGS_COLS = frozenset({"Atom1_NRings", "Atom2_NRings"})

POLYCYCLIC = (
    pytest.param("c1ccc2ccccc2c1", id="naphthalene"),
    pytest.param("c1ccc2c(c1)oc1ccccc12", id="dibenzofuran"),
    pytest.param("c1ccc2c(c1)ccc1ccccc12", id="phenanthrene"),
)

ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"


def _ob_endpoints(row: dict) -> tuple[int, int]:
    _mol, a, b = str(row["_index"]).split(".", 2)
    return int(a), int(b)


def _legacy_nrings_for_atom(pymol, ob_idx: int) -> int:
    cycles = MolGraph(pymol).dfs_cycles()
    return sum(1 for cyc in cycles if ob_idx in cyc)


def _principled_nrings_for_atom(rdmol, ob_idx: int) -> float:
    from xenosite.predict.features import _ob

    return float(rdmol.GetRingInfo().NumAtomRings(_ob.ob_idx_to_rdkit(ob_idx)))


def _bond_rows_by_atoms(rows: list[dict]) -> dict[frozenset[int], dict]:
    return {frozenset(r["_atoms"]): r for r in rows}


def _select_vectors(rows: list[dict], names: list[str]) -> list[list[float]]:
    from xenosite.predict.features.names import select_columns

    return [select_columns(r, names) for r in rows]


def _assert_only_nrings_columns_differ(
    legacy_rows: list[dict],
    principled_rows: list[dict],
    names: list[str],
    *,
    require_nrings_change: bool,
) -> None:
    by_legacy = _bond_rows_by_atoms(legacy_rows)
    by_principled = _bond_rows_by_atoms(principled_rows)
    assert set(by_legacy) == set(by_principled)
    saw_nrings_diff = False
    for key in by_legacy:
        l_vec = _select_vectors([by_legacy[key]], names)[0]
        p_vec = _select_vectors([by_principled[key]], names)[0]
        diff_cols = [
            names[i]
            for i in range(len(names))
            if not np.isclose(l_vec[i], p_vec[i], atol=0.0, rtol=0.0)
        ]
        if diff_cols:
            assert set(diff_cols) <= NRINGS_COLS, (
                f"bond {sorted(key)}: unexpected column drift {diff_cols}"
            )
            saw_nrings_diff = True
    if require_nrings_change:
        assert saw_nrings_diff, "expected legacy vs principled NRings to differ on this scaffold"


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


def _epoxidation_scores(smiles: str, *, parameter: dict | None = None) -> dict:
    if not onnx_weights_present("epoxidation"):
        pytest.skip("no ONNX weights for epoxidation")
    kwargs: dict = {"models": ["epoxidation"], "backend": BACKEND}
    if parameter is not None:
        kwargs["_parameter"] = parameter
    mol = predict(smiles, **kwargs)
    assert mol.results
    return golden_score_fields(mol.results[0])


def _epoxidation_smiles_from_golden() -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for row in load_golden_suite():
        if row.get("model") != "epoxidation":
            continue
        smi = row.get("smiles")
        if not smi or smi in seen:
            continue
        seen.add(smi)
        out.append(smi)
    return out


# ---------------------------------------------------------------------------
# Semantics oracles (descriptor layer, no ONNX)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("smiles", POLYCYCLIC)
def test_legacy_nrings_matches_dfs_cycle_membership(smiles: str) -> None:
    """Legacy ``Atom*_NRings`` equals how many DFS back-edge cycles contain the endpoint."""
    from xenosite.predict.features import _ob

    rdmol, _ = parse_smiles(smiles)
    pymol = _ob.from_rdkit_mol(rdmol)
    rows = bond_rows(rdmol, bond_nrings_mode="legacy")
    for row in rows:
        a_ob, b_ob = _ob_endpoints(row)
        assert float(row["Atom1_NRings"]) == float(_legacy_nrings_for_atom(pymol, a_ob))
        assert float(row["Atom2_NRings"]) == float(_legacy_nrings_for_atom(pymol, b_ob))


@pytest.mark.parametrize("smiles", POLYCYCLIC)
def test_principled_nrings_matches_rdkit_num_atom_rings(smiles: str) -> None:
    """Principled ``Atom*_NRings`` equals RDKit ``RingInfo.NumAtomRings`` per endpoint."""
    rdmol, _ = parse_smiles(smiles)
    rows = bond_rows(rdmol, bond_nrings_mode="principled")
    for row in rows:
        a_ob, b_ob = _ob_endpoints(row)
        assert float(row["Atom1_NRings"]) == _principled_nrings_for_atom(rdmol, a_ob)
        assert float(row["Atom2_NRings"]) == _principled_nrings_for_atom(rdmol, b_ob)


def test_nrings_is_endpoint_atom_count_not_bond_ring_count() -> None:
    """``NumBondRings`` counts rings containing the bond; ``Atom*_NRings`` count endpoint atoms."""
    rdmol, _ = parse_smiles("c1ccc2c(c1)ccc1ccccc12")
    ri = rdmol.GetRingInfo()
    rows = bond_rows(rdmol, bond_nrings_mode="principled")
    row = next(r for r in rows if tuple(r["_atoms"]) == (2, 3))
    a_rdk, b_rdk = row["_atoms"]
    bond_idx = rdmol.GetBondBetweenAtoms(a_rdk, b_rdk).GetIdx()
    assert float(row["Atom1_NRings"]) == float(ri.NumAtomRings(a_rdk))
    assert float(row["Atom2_NRings"]) == float(ri.NumAtomRings(b_rdk))
    assert ri.NumBondRings(bond_idx) == 1
    assert (float(row["Atom1_NRings"]), float(row["Atom2_NRings"])) == (1.0, 2.0)


def test_aspirin_legacy_and_principled_nrings_agree() -> None:
    """Acyclic / simple rings: legacy DFS and principled RDKit agree on every bond."""
    rdmol, _ = parse_smiles(ASPIRIN)
    legacy = bond_rows(rdmol, bond_nrings_mode="legacy")
    principled = bond_rows(rdmol, bond_nrings_mode="principled")
    names = load_names("epoxidation", "bond")
    assert names
    _assert_only_nrings_columns_differ(
        legacy, principled, names, require_nrings_change=False
    )


# ---------------------------------------------------------------------------
# Descriptor ablation: only NRings columns move
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("smiles", POLYCYCLIC)
def test_epoxidation_descriptor_ablation_only_nrings_columns_differ(smiles: str) -> None:
    """Legacy vs principled epoxidation bond rows differ only in ``Atom*_NRings``."""
    rdmol, _ = parse_smiles(smiles)
    legacy = rows_for_model("epoxidation", rdmol, _parameter={"bond_nrings_mode": "legacy"})
    principled = rows_for_model(
        "epoxidation", rdmol, _parameter={"bond_nrings_mode": "principled"}
    )
    names = load_names("epoxidation", "bond")
    assert names
    _assert_only_nrings_columns_differ(
        legacy, principled, names, require_nrings_change=True
    )


def test_phenanthrene_legacy_nrings_diff_on_symmetric_bonds() -> None:
    """Fusion atoms inflate legacy counts on one symmetric bond class only."""
    rdmol, _ = parse_smiles("c1ccc2c(c1)ccc1ccccc12")
    legacy = bond_rows(rdmol, bond_nrings_mode="legacy")
    principled = bond_rows(rdmol, bond_nrings_mode="principled")
    by_l = _bond_rows_by_atoms(legacy)
    by_p = _bond_rows_by_atoms(principled)
    l01 = (float(by_l[frozenset({0, 1})]["Atom1_NRings"]), float(by_l[frozenset({0, 1})]["Atom2_NRings"]))
    l1011 = (
        float(by_l[frozenset({10, 11})]["Atom1_NRings"]),
        float(by_l[frozenset({10, 11})]["Atom2_NRings"]),
    )
    p01 = (float(by_p[frozenset({0, 1})]["Atom1_NRings"]), float(by_p[frozenset({0, 1})]["Atom2_NRings"]))
    p1011 = (
        float(by_p[frozenset({10, 11})]["Atom1_NRings"]),
        float(by_p[frozenset({10, 11})]["Atom2_NRings"]),
    )
    assert l01 == (1.0, 1.0)
    assert l1011 == (2.0, 2.0)
    assert p01 == p1011 == (1.0, 1.0)


# ---------------------------------------------------------------------------
# ONNX score ablation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("smiles", POLYCYCLIC)
def test_epoxidation_bond_nrings_flag_alone_explains_legacy_to_default_drift(smiles: str) -> None:
    """Golden + ``bond_nrings_mode=principled`` equals production default; explains full drift."""
    default = _epoxidation_scores(smiles)
    legacy = _epoxidation_scores(smiles, parameter=GOLDEN_PARAMETER)
    nrings_only = _epoxidation_scores(
        smiles,
        parameter={**GOLDEN_PARAMETER, "bond_nrings_mode": "principled"},
    )

    full_drift = _max_score_delta(default, legacy)
    nrings_drift = _max_score_delta(nrings_only, legacy)

    assert full_drift > PARITY_ATOL
    assert _max_score_delta(nrings_only, default) <= PARITY_ATOL
    assert abs(full_drift - nrings_drift) <= max(PARITY_ATOL, 0.05 * full_drift)


def test_epoxidation_symmetry_pooling_no_op_once_nrings_principled() -> None:
    """RDKit bond pooling only moves scores when legacy NRings breaks feature symmetry."""
    smiles = "c1ccc2c(c1)ccc1ccccc12"
    legacy = _epoxidation_scores(smiles, parameter=GOLDEN_PARAMETER)
    nrings_only = _epoxidation_scores(
        smiles,
        parameter={**GOLDEN_PARAMETER, "bond_nrings_mode": "principled"},
    )
    both = _epoxidation_scores(
        smiles,
        parameter={
            **GOLDEN_PARAMETER,
            "bond_nrings_mode": "principled",
            "symmetry_group_mode": "rdkit",
        },
    )
    symmetry_only = _epoxidation_scores(
        smiles,
        parameter={**GOLDEN_PARAMETER, "symmetry_group_mode": "rdkit"},
    )
    assert _max_score_delta(symmetry_only, legacy) > PARITY_ATOL
    assert _max_score_delta(both, nrings_only) <= PARITY_ATOL


def test_epoxidation_aspirin_bond_nrings_ablation_is_no_op() -> None:
    """Simple monocycle: legacy and principled NRings agree ⇒ zero score drift."""
    legacy = _epoxidation_scores(ASPIRIN, parameter=GOLDEN_PARAMETER)
    nrings_only = _epoxidation_scores(
        ASPIRIN,
        parameter={**GOLDEN_PARAMETER, "bond_nrings_mode": "principled"},
    )
    assert _max_score_delta(nrings_only, legacy) <= PARITY_ATOL


@pytest.mark.full
def test_epoxidation_bond_nrings_ablation_explains_drift_on_golden_suite() -> None:
    """Across golden epoxidation molecules, NRings-only fix matches full drift when drift > atol."""
    smiles_list = _epoxidation_smiles_from_golden()
    assert smiles_list, "need golden suite epoxidation rows"
    explained = 0
    drifters = 0
    for smiles in smiles_list:
        default = _epoxidation_scores(smiles)
        legacy = _epoxidation_scores(smiles, parameter=GOLDEN_PARAMETER)
        nrings_only = _epoxidation_scores(
            smiles,
            parameter={**GOLDEN_PARAMETER, "bond_nrings_mode": "principled"},
        )
        full = _max_score_delta(default, legacy)
        partial = _max_score_delta(nrings_only, legacy)
        if full <= PARITY_ATOL:
            continue
        drifters += 1
        if abs(full - partial) <= max(PARITY_ATOL, 0.05 * full):
            explained += 1
    assert drifters >= 10, "expected multiple epoxidation legacy drifters in golden suite"
    assert explained / drifters >= 0.95, (
        f"bond_nrings ablation explained {explained}/{drifters} drifters "
        f"(expected ≥95%; symmetry should not move epoxidation scores alone)"
    )
