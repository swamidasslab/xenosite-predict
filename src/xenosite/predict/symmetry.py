"""Topological symmetry classes for score grouping (RDKit or OpenBabel).

Production ``predict()`` defaults to RDKit ``CanonicalRankAtoms`` classes and
pools scores within each class (mean of active members). Golden / legacy parity
passes ``symmetry_group_mode="openbabel"`` on ``_parameter`` to match OpenBabel
GID bond classes from the TF1 frontend without RDKit pooling.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Literal, Mapping, Any, Sequence

from rdkit import Chem

SymmetryGroupMode = Literal["rdkit", "openbabel"]

_ACTIVE_ATOL = 1e-12


def resolve_symmetry_group_mode(parameter: Mapping[str, Any] | None) -> SymmetryGroupMode:
    mode = (parameter or {}).get("symmetry_group_mode", "rdkit")
    return mode if mode in ("rdkit", "openbabel") else "rdkit"


def uses_rdkit_symmetry(parameter: Mapping[str, Any] | None) -> bool:
    return resolve_symmetry_group_mode(parameter) == "rdkit"


def rdkit_atom_ranks(rdmol: Chem.Mol) -> list[int]:
    return [int(r) for r in Chem.CanonicalRankAtoms(rdmol, breakTies=False)]


def rdkit_bond_symmetry_key(rdmol: Chem.Mol, atom_a: int, atom_b: int) -> tuple[int, int, int]:
    """Bond class from sorted endpoint atom symmetry ranks and bond order."""
    ranks = rdkit_atom_ranks(rdmol)
    ri, rj = ranks[int(atom_a)], ranks[int(atom_b)]
    bond = rdmol.GetBondBetweenAtoms(int(atom_a), int(atom_b))
    bt = int(bond.GetBondType()) if bond is not None else 0
    return (min(ri, rj), max(ri, rj), bt)


def rdkit_bond_symmetry_groups(rdmol: Chem.Mol) -> dict[tuple[int, int, int], list[int]]:
    """Map RDKit bond symmetry key → 0-based bond indices (all classes)."""
    groups: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    for bond in rdmol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        key = rdkit_bond_symmetry_key(rdmol, i, j)
        groups[key].append(int(bond.GetIdx()))
    return dict(groups)


def rdkit_atom_symmetry_groups(rdmol: Chem.Mol) -> dict[int, list[int]]:
    """Map RDKit atom symmetry rank → 0-based atom indices (all classes)."""
    groups: dict[int, list[int]] = defaultdict(list)
    for idx, rank in enumerate(rdkit_atom_ranks(rdmol)):
        groups[int(rank)].append(int(idx))
    return dict(groups)


def _pool_scores_within_index_groups(
    scores: Sequence[float],
    groups: Mapping[Any, Sequence[int]],
) -> list[float]:
    """Set each multi-member index group to the mean of its active scores.

    Active means ``abs(score) > _ACTIVE_ATOL``. When a symmetry class maps to
    bonds with different descriptors (and thus different raw ONNX scores), pooling
    averages those scores instead of copying an arbitrary first member. When only
    one bond in the class is active (typical ndealk principled dedup), the mean
    equals that score and all siblings receive the same value.
    """
    out = [float(v) for v in scores]
    for members in groups.values():
        if len(members) < 2:
            continue
        vals = [out[i] for i in members if 0 <= i < len(out)]
        if not vals:
            continue
        active = [v for v in vals if abs(v) > _ACTIVE_ATOL]
        pooled = sum(active) / len(active) if active else 0.0
        for i in members:
            if 0 <= i < len(out):
                out[i] = pooled
    return out


def broadcast_atom_scores_within_rdkit_groups(
    rdmol: Chem.Mol,
    atom_scores: Sequence[float],
) -> list[float]:
    return _pool_scores_within_index_groups(
        atom_scores, rdkit_atom_symmetry_groups(rdmol)
    )


def broadcast_bond_scores_within_rdkit_groups(
    rdmol: Chem.Mol,
    bond_scores: Sequence[float],
) -> list[float]:
    return _pool_scores_within_index_groups(
        bond_scores, rdkit_bond_symmetry_groups(rdmol)
    )


def apply_atom_symmetry(
    rdmol: Chem.Mol,
    atom_scores: Sequence[float],
    parameter: Mapping[str, Any] | None,
) -> list[float]:
    """Return *atom_scores*, pooled within RDKit classes when configured."""
    scores = [float(v) for v in atom_scores]
    if uses_rdkit_symmetry(parameter):
        return broadcast_atom_scores_within_rdkit_groups(rdmol, scores)
    return scores


def apply_bond_symmetry(
    rdmol: Chem.Mol,
    bond_scores: Sequence[float],
    parameter: Mapping[str, Any] | None,
) -> list[float]:
    """Return *bond_scores*, pooled within RDKit classes when configured."""
    scores = [float(v) for v in bond_scores]
    if uses_rdkit_symmetry(parameter):
        return broadcast_bond_scores_within_rdkit_groups(rdmol, scores)
    return scores
