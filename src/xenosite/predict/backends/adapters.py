"""Shared adapters: backend-native scores → user-API :class:`Molecule` results.

Ported in spirit from ``xenosite-api`` ``v0/adapters.py``. One helper set, not
per-model copies. Numeric scores must match; 0-based RDKit indices; adapter
quirks (UGT replacing ``results``, quinone ``{}`` → 0.0) are not replayed
except insofar as scores stay the same.
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence

import numpy as np

from ..types import (
    AtomBondResult,
    AtomResult,
    BondResult,
    Metabolite,
    MolAtomPairResult,
    MolAtomResult,
    MolBondResult,
    Molecule,
)


def append_mol_bond(
    molecule: Molecule,
    *,
    model: str,
    version: str,
    mol: float,
    bond: Sequence[float],
) -> None:
    molecule.results.append(
        MolBondResult(model=model, version=version, mol=float(mol), bond=[float(x) for x in bond])
    )


def append_mol_atom(
    molecule: Molecule,
    *,
    model: str,
    version: str,
    mol: float,
    atom: Sequence[float],
    metabolite: Optional[list[Metabolite]] = None,
) -> None:
    molecule.results.append(
        MolAtomResult(
            model=model,
            version=version,
            mol=float(mol),
            atom=[float(x) for x in atom],
            metabolite=metabolite,
        )
    )


def append_atom(
    molecule: Molecule,
    *,
    model: str,
    version: str,
    atom: Sequence[float],
) -> None:
    molecule.results.append(
        AtomResult(model=model, version=version, atom=[float(x) for x in atom])
    )


def append_bond(
    molecule: Molecule,
    *,
    model: str,
    version: str,
    bond: Sequence[float],
) -> None:
    molecule.results.append(
        BondResult(model=model, version=version, bond=[float(x) for x in bond])
    )


def append_atom_bond(
    molecule: Molecule,
    *,
    model: str,
    version: str,
    atom: Sequence[float],
    bond: Sequence[float],
) -> None:
    molecule.results.append(
        AtomBondResult(
            model=model,
            version=version,
            atom=[float(x) for x in atom],
            bond=[float(x) for x in bond],
        )
    )


def append_atom_pair(
    molecule: Molecule,
    *,
    model: str,
    version: str,
    mol: float,
    atom: Sequence[float],
    pair: Sequence[float],
    pair_idx: Sequence[tuple[int, int]],
) -> None:
    pp = canonicalize_pair_idx(list(pair_idx), list(pair))
    molecule.results.append(
        MolAtomPairResult(
            model=model,
            version=version,
            mol=float(mol),
            atom=[float(x) for x in atom],
            pair=pp["pair"],
            pair_idx=pp["pair_idx"],
        )
    )


def canonicalize_pair_idx(
    pair_idx: Sequence[tuple[int, int]], pair: Sequence[float]
) -> dict:
    ix = [(tuple(sorted(idxs)), float(x)) for idxs, x in zip(pair_idx, pair)]
    ix.sort()
    return {"pair_idx": [i for i, _ in ix], "pair": [x for _, x in ix]}


def canonical_bond_site_pair(a: int, b: int) -> tuple[int, int]:
    """Legacy ndealk/isozyme site keys use ascending atom ids (``2-1`` → ``1-2``)."""
    if a <= b:
        return a, b
    return b, a


def reorder_by_bond(
    scores: Sequence[float],
    current: Sequence[Iterable[int]],
    new: Sequence[tuple[int, int]],
    *,
    fill: float = 0.0,
) -> list[float]:
    """Map bond scores onto ``molecule.bonds.idx`` (frozenset of atom ids)."""
    lookup = {frozenset(b): i for i, b in enumerate(current)}
    out: list[float] = []
    for b in new:
        key = frozenset(b)
        out.append(float(scores[lookup[key]]) if key in lookup else fill)
    return out


def or_combine(values: Sequence[float]) -> float:
    """``1 - prod(1 - p)`` aggregation used by quinone atoms and bioactivation."""
    if not values:
        return 0.0
    arr = np.asarray(values, dtype=float)
    return float(1.0 - np.prod(1.0 - arr))


def safe_atom_index(v) -> int:
    """OpenBabel 1-based atom id → 0-based RDKit index (``v0/adapters.py``)."""
    try:
        return int(v) - 1
    except (TypeError, ValueError):
        return v  # type: ignore[return-value]


def legacy_atom_vector(site_map: dict, n_atoms: int, *, one_based: bool = True) -> list[float]:
    """Per-atom scores from a legacy site map → 0-based RDKit vector.

    Delegates to :func:`xenosite.predict.numbering.legacy_site_to_atom_vector`
    (handles gapped legacy OB 2.4 keys on ``[nH]`` SMILES).
    """
    from ..numbering import legacy_site_to_atom_vector

    return legacy_site_to_atom_vector(site_map, n_atoms)
