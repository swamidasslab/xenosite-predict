"""Topological symmetry classes for score grouping (RDKit or OpenBabel).

Production ``predict()`` defaults to RDKit ``CanonicalRankAtoms`` classes.
Golden / legacy parity passes ``symmetry_group_mode="openbabel"`` on
``_parameter`` to match OpenBabel GID bond classes from the TF1 frontend.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Literal, Mapping, Any

from rdkit import Chem

SymmetryGroupMode = Literal["rdkit", "openbabel"]


def resolve_symmetry_group_mode(parameter: Mapping[str, Any] | None) -> SymmetryGroupMode:
    mode = (parameter or {}).get("symmetry_group_mode", "rdkit")
    return mode if mode in ("rdkit", "openbabel") else "rdkit"


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


def broadcast_bond_scores_within_rdkit_groups(
    rdmol: Chem.Mol,
    bond_scores: list[float],
) -> list[float]:
    """Set every bond in an RDKit symmetry class to the class representative score."""
    out = [float(v) for v in bond_scores]
    for members in rdkit_bond_symmetry_groups(rdmol).values():
        if len(members) < 2:
            continue
        vals = [out[i] for i in members if i < len(out)]
        if not vals:
            continue
        active = [v for v in vals if abs(v) > 1e-12]
        rep = active[0] if active else 0.0
        for i in members:
            if i < len(out):
                out[i] = rep
    return out
