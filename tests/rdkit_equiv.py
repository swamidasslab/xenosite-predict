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
