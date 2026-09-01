"""Phase1 mol-head features (legacy ``get_top_pred`` + ``MolDesc``)."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .bond_lonepair import TARGET_FIELDS

_LEGACY_HEADS = (
    "StableOxygenation",
    "UnstableOxygenation",
    "Dehydrogenation",
    "Reduction",
    "Hydrolysis",
)

_MOL_DESC = [
    "MolDesc__atoms",
    "MolDesc__bonds",
    "MolDesc__TPSA",
    "MolDesc__logP",
    "MolDesc__MW",
    "MolDesc__MR",
    "MolDesc__HBD",
    "MolDesc__HBA1",
    "MolDesc__HBA2",
    "MolDesc__sbonds",
    "MolDesc__dbonds",
    "MolDesc__tbonds",
    "MolDesc__abonds",
    "MolDesc__heavy_atoms",
    "MolDesc__hydrogens",
    "MolDesc__NumRings",
]


def phase1_site_column_names(row: dict) -> list[str]:
    """404 numeric Bond_and_LonePair columns (exclude training-label placeholders)."""
    exclude = set(TARGET_FIELDS) | {f"Mol{t}" for t in TARGET_FIELDS}
    return [
        k
        for k, v in row.items()
        if isinstance(v, (int, float)) and not k.startswith("_") and k not in exclude
    ]


def phase1_mol_features(
    site_scores: np.ndarray,
    rows: list[dict],
    mol_names: list[str],
    *,
    topn: int = 5,
) -> np.ndarray:
    """One mol-head row: 16 ``MolDesc__*`` + top-5 site scores per class."""
    values: dict[str, float] = {}
    if rows:
        for col in _MOL_DESC:
            values[col] = float(rows[0].get(col, 0.0))

    for hi, head in enumerate(_LEGACY_HEADS):
        col = site_scores[:, hi]
        top = sorted((float(s) for s in col), reverse=True)
        if len(top) < topn:
            top = top + [0.0] * (topn - len(top))
        for k in range(topn):
            values[f"{head}_top{k}"] = top[k]

    return np.asarray([[float(values.get(n, 0.0)) for n in mol_names]], dtype=np.float64)
