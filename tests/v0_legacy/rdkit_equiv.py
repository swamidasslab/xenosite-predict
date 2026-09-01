"""RDKit symmetry oracle helpers for score invariance tests.

Grouping logic lives in ``xenosite.predict.symmetry``; this module adds test
assertions only.
"""

from __future__ import annotations

from collections import defaultdict

from rdkit import Chem

from xenosite.predict.symmetry import rdkit_atom_ranks, rdkit_bond_symmetry_key

from tests.support import PARITY_ATOL

ACTIVE_ATOL = 1e-12

STRICT_BOND_MODELS = frozenset({"epoxidation", "ndealk", "isozyme"})


def atom_symmetry_groups(rdmol: Chem.Mol) -> dict[int, list[int]]:
    """Map symmetry rank → equivalent 0-based RDKit atom indices (size ≥ 2 only)."""
    ranks = rdkit_atom_ranks(rdmol)
    groups: dict[int, list[int]] = defaultdict(list)
    for idx, rank in enumerate(ranks):
        groups[int(rank)].append(int(idx))
    return {rank: members for rank, members in groups.items() if len(members) > 1}


def bond_symmetry_groups(rdmol: Chem.Mol) -> dict[tuple[int, int, int], list[int]]:
    """Map bond symmetry key → 0-based RDKit bond indices (size ≥ 2 only)."""
    groups: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    for bond in rdmol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        key = rdkit_bond_symmetry_key(rdmol, i, j)
        groups[key].append(int(bond.GetIdx()))
    return {key: members for key, members in groups.items() if len(members) > 1}


def assert_scores_cover_indices(scores: list[float], indices: list[int], *, label: str) -> None:
    if not indices:
        return
    assert len(scores) > max(indices), (
        f"{label}: score vector length {len(scores)} missing indices through {max(indices)}"
    )


def assert_full_atom_vector(scores: list[float], n_atoms: int) -> None:
    assert len(scores) == n_atoms, f"expected {n_atoms} atom scores, got {len(scores)}"
    assert all(isinstance(float(v), float) for v in scores)


def assert_full_bond_vector(scores: list[float], n_bonds: int) -> None:
    assert len(scores) == n_bonds, f"expected {n_bonds} bond scores, got {len(scores)}"
    assert all(isinstance(float(v), float) for v in scores)


def assert_atom_scores_symmetric(
    scores: list[float],
    groups: dict[int, list[int]],
    *,
    atol: float = PARITY_ATOL,
) -> None:
    for rank, members in groups.items():
        assert_scores_cover_indices(scores, members, label=f"atom rank {rank}")
        vals = [float(scores[i]) for i in members]
        span = max(vals) - min(vals)
        assert span <= atol, f"atom symmetry rank {rank} members {members}: {vals}"


def assert_bond_scores_symmetric(
    scores: list[float],
    groups: dict[tuple[int, int, int], list[int]],
    *,
    atol: float = PARITY_ATOL,
) -> None:
    for key, members in groups.items():
        assert_scores_cover_indices(scores, members, label=f"bond {key}")
        vals = [float(scores[i]) for i in members]
        span = max(vals) - min(vals)
        assert span <= atol, f"bond symmetry {key} members {members}: {vals}"


def assert_bond_scores_openbabel_principled(
    scores: list[float],
    groups: dict[tuple[int, int, int], list[int]],
    *,
    atol: float = PARITY_ATOL,
) -> None:
    """OpenBabel-class dedup without RDKit broadcast: ≤1 active score per class."""
    for key, members in groups.items():
        assert_scores_cover_indices(scores, members, label=f"bond {key}")
        vals = [float(scores[i]) for i in members]
        active = [v for v in vals if abs(v) > ACTIVE_ATOL]
        if len(active) >= 2:
            span = max(active) - min(active)
            assert span <= atol, (
                f"openbabel principled bond symmetry {key} members {members}: {active}"
            )


def _descriptor_vectors(rows: list[dict], names: list[str]):
    from xenosite.predict.features.names import select_columns

    return [select_columns(r, names) for r in rows]


def assert_atom_descriptor_rows_symmetric(
    rows: list[dict],
    rdmol: Chem.Mol,
    names: list[str],
    *,
    atol: float = PARITY_ATOL,
) -> None:
    """ONNX atom-head columns must match within each multi-member RDKit atom class."""
    import numpy as np

    ranks = rdkit_atom_ranks(rdmol)
    by_rank: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        if "_atom" not in row:
            continue
        by_rank[int(ranks[int(row["_atom"])])].append(row)
    for rank, members in by_rank.items():
        if len(members) < 2:
            continue
        vecs = _descriptor_vectors(members, names)
        ref = vecs[0]
        atoms = [int(m["_atom"]) for m in members]
        for vec, atom in zip(vecs[1:], atoms[1:]):
            if not np.allclose(ref, vec, atol=atol, rtol=0):
                diff_cols = [
                    names[i]
                    for i in range(len(names))
                    if not np.isclose(ref[i], vec[i], atol=atol, rtol=0)
                ]
                raise AssertionError(
                    f"atom symmetry rank {rank} atoms {atoms}: "
                    f"{len(diff_cols)} columns differ (e.g. {diff_cols[:5]})"
                )


def assert_bond_descriptor_rows_symmetric(
    rows: list[dict],
    rdmol: Chem.Mol,
    names: list[str],
    *,
    pymol=None,
    atol: float = PARITY_ATOL,
) -> None:
    """ONNX bond-head columns match within each directed OpenBabel bond class."""
    import numpy as np

    from xenosite.predict.features import _ob
    from xenosite.predict.symmetry import directed_ob_bond_symmetry_key

    if pymol is None:
        pymol = _ob.from_rdkit_mol(rdmol)
    by_key: dict[tuple[int, int, int], list[dict]] = defaultdict(list)
    for row in rows:
        if "_atoms" not in row or "_index" not in row:
            continue
        key = directed_ob_bond_symmetry_key(pymol, row, rdmol)
        by_key[key].append(row)
    for key, members in by_key.items():
        if len(members) < 2:
            continue
        vecs = _descriptor_vectors(members, names)
        ref = vecs[0]
        bonds = [tuple(m["_atoms"]) for m in members]
        for vec, bond in zip(vecs[1:], bonds[1:]):
            if not np.allclose(ref, vec, atol=atol, rtol=0):
                diff_cols = [
                    names[i]
                    for i in range(len(names))
                    if not np.isclose(ref[i], vec[i], atol=atol, rtol=0)
                ]
                raise AssertionError(
                    f"directed OB bond symmetry {key} pairs {bonds}: "
                    f"{len(diff_cols)} columns differ (e.g. {diff_cols[:5]})"
                )
