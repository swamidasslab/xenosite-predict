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
BondNringsMode = Literal["legacy", "principled"]

_ACTIVE_ATOL = 1e-12


def directed_bondtd_pair(row: Mapping[str, Any]) -> tuple[int, int]:
    """Directed BondTD / Bond_and_LonePair endpoints from ``_index`` ``…a.b`` (1-based OB)."""
    parts = str(row["_index"]).split(".")
    return int(parts[-2]), int(parts[-1])


def collapse_opposite_direction_rows(
    rows: list[dict],
    scores: Sequence[float] | Any,
) -> tuple[list[dict], Any]:
    """Max-merge scores when both ``(a, b)`` and ``(b, a)`` descriptor rows exist.

    Phase1 and ndealk BondTD tables can list the same heavy-atom bond in both
    directions; only one direction may appear for a given bond. When both exist,
    take the per-column max, then keep one representative row.
    """
    import numpy as np

    if not rows:
        return [], np.asarray(scores, dtype=float)
    arr = np.asarray(scores, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    elif arr.shape[0] != len(rows) and arr.shape[1] == len(rows):
        arr = arr.T
    if arr.shape[0] != len(rows):
        raise ValueError(f"scores length {arr.shape[0]} != rows {len(rows)}")

    groups: list[list[int]] = []
    seen: set[int] = set()
    dir_to_idx = {directed_bondtd_pair(row): i for i, row in enumerate(rows)}
    for i, row in enumerate(rows):
        if i in seen:
            continue
        a, b = directed_bondtd_pair(row)
        if a == b:
            groups.append([i])
            seen.add(i)
            continue
        rev = dir_to_idx.get((b, a))
        if rev is not None and rev not in seen:
            groups.append([i, rev])
            seen.add(i)
            seen.add(rev)
        else:
            groups.append([i])
            seen.add(i)

    out_rows: list[dict] = []
    out_scores: list[np.ndarray] = []
    for members in groups:
        block = arr[members]
        merged = block.max(axis=0)
        rep = members[int(np.argmax(block[:, 0]))]
        out_rows.append(rows[rep])
        out_scores.append(merged)
    out_arr = np.vstack(out_scores)
    if out_arr.shape[1] == 1:
        return out_rows, out_arr.reshape(-1)
    return out_rows, out_arr


def resolve_symmetry_group_mode(parameter: Mapping[str, Any] | None) -> SymmetryGroupMode:
    mode = (parameter or {}).get("symmetry_group_mode", "rdkit")
    return mode if mode in ("rdkit", "openbabel") else "rdkit"


def uses_rdkit_symmetry(parameter: Mapping[str, Any] | None) -> bool:
    return resolve_symmetry_group_mode(parameter) == "rdkit"


def resolve_bond_nrings_mode(parameter: Mapping[str, Any] | None) -> BondNringsMode:
    mode = (parameter or {}).get("bond_nrings_mode", "principled")
    return mode if mode in ("legacy", "principled") else "principled"


def directed_ob_bond_symmetry_key(pymol, row: Mapping[str, Any], rdmol: Chem.Mol) -> tuple[int, int, int]:
    """Directed OpenBabel bond class: (GID(Atom1), GID(Atom2), RDKit bond order).

    Atom1/Atom2 follow BondTD ``_index`` endpoint order (``mol.a.b``), not the
    sorted undirected GID pair used for legacy site dedup.
    """
    from .features import _ob

    ob, _ = _ob.load()
    vec = ob.vectorUnsignedInt()
    pymol.OBMol.GetGIDVector(vec)
    ranks = list(vec)
    _mol, a, b = str(row["_index"]).split(".", 2)
    ia, ib = int(a), int(b)
    g1 = int(ranks[ia - 1]) if ia - 1 < len(ranks) else 0
    g2 = int(ranks[ib - 1]) if ib - 1 < len(ranks) else 0
    i, j = int(row["_atoms"][0]), int(row["_atoms"][1])
    bond = rdmol.GetBondBetweenAtoms(i, j)
    bt = int(bond.GetBondType()) if bond is not None else 0
    return (g1, g2, bt)


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
