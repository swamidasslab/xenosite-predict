"""Quinone atom-pair filtering and mol-head feature assembly.

Port of ``EligibleAtoms``, ``AddAtomPred``, and ``MolData`` from legacy quinone1.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
from rdkit import Chem

MOL_DESC_KEYS = (
    "bonds",
    "TPSA",
    "logP",
    "MW",
    "MR",
    "HBD",
    "HBA1",
    "HBA2",
    "sbonds",
    "dbonds",
    "tbonds",
    "abonds",
    "heavy_atoms",
    "hydrogens",
    "NumRings",
)


def _ob_idx(row: dict) -> int:
    return int(str(row["_index"]).split(".")[-1])


def eligible_atom_rows(atom_rows: list[dict]) -> list[dict]:
    """Atoms with ``NC_0 == 1`` and ``NRings > 0`` (legacy ``atom_descriptors`` filter)."""
    out = []
    for row in atom_rows:
        if row.get("NC_0") == 1 and (row.get("NRings") or 0) > 0:
            out.append(row)
    return out


def _logit(p: float) -> float:
    p = min(max(float(p), 1e-15), 1.0 - 1e-15)
    return math.log(p) - math.log(1.0 - p)


def quinone_pair_rows(
    rdkit_mol,
    atom_rows: list[dict],
    atom_scores_by_ob: dict[int, float],
) -> list[dict]:
    """Eligible atom pairs with distance + logit atom preds (legacy ``AddAtomPred``)."""
    eligible = eligible_atom_rows(atom_rows)
    if len(eligible) < 2:
        return []

    ob_to_rdkit = {_ob_idx(r): int(r["_atom"]) for r in atom_rows}
    dist = Chem.GetDistanceMatrix(rdkit_mol)
    pairs: list[dict] = []

    obs = sorted(_ob_idx(r) for r in eligible)
    for i, a1 in enumerate(obs):
        for a2 in obs[i + 1 :]:
            rd1, rd2 = ob_to_rdkit[a1], ob_to_rdkit[a2]
            d = float(dist[rd1, rd2])
            s1 = float(atom_scores_by_ob.get(a1, 0.0))
            s2 = float(atom_scores_by_ob.get(a2, 0.0))
            top, second = sorted((s1, s2), reverse=True)
            pairs.append(
                {
                    "Atom1_Pred": _logit(top),
                    "Atom2_Pred": _logit(second),
                    "AtomPair__Distance": d,
                    "AtomPair__Distance_Is_Odd": float(int(d) % 2),
                    "_atoms": (min(rd1, rd2), max(rd1, rd2)),
                    "_ob_atoms": (a1, a2),
                }
            )

    pairs.sort(key=lambda r: (r["_ob_atoms"][1], r["_ob_atoms"][0]))
    return pairs


def quinone_mol_features(
    atom_rows: list[dict],
    pair_rows: list[dict],
    pair_scores: Sequence[float],
    mol_names: list[str],
) -> np.ndarray:
    """One mol-head row: MolDesc max + AtomPair/Atom Top3 (legacy ``MolData``)."""
    eligible = eligible_atom_rows(atom_rows)
    values: dict[str, float] = {}

    for key in MOL_DESC_KEYS:
        col = f"MolDesc__{key}"
        vals = [float(r.get(col, 0.0)) for r in eligible]
        values[col] = max(vals) if vals else 0.0

    ordered_pairs = sorted(
        zip(pair_scores, pair_rows),
        key=lambda x: -float(x[0]),
    )
    for k in range(1, 4):
        values[f"AtomPair__Top{k}"] = (
            float(ordered_pairs[k - 1][0]) if k - 1 < len(ordered_pairs) else 0.0
        )

    atom1_preds = sorted((float(r["Atom1_Pred"]) for r in pair_rows), reverse=True)
    for k in range(1, 4):
        values[f"Atom__Top{k}"] = atom1_preds[k - 1] if k - 1 < len(atom1_preds) else 0.0

    return np.asarray([[values.get(n, 0.0) for n in mol_names]], dtype=np.float64)
