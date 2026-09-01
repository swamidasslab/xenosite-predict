"""Reactivity mol-head features (legacy ``MolDesc`` / ``AtomTop5``)."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .atom import MOL_DESC

_MOL_OUT_INDEX = {
    "cyanide": 0,
    "dna": 1,
    "gsh": 2,
    "protein": 3,
}

_HEAD_TO_TARGET = {
    "cyanide": "MULTITARGET_Cyanide",
    "dna": "MULTITARGET_DNA",
    "gsh": "MULTITARGET_GSH",
    "protein": "MULTITARGET_Protein",
}

_TOP_LABEL = {
    "MULTITARGET_Cyanide": "Cyanide",
    "MULTITARGET_DNA": "DNA",
    "MULTITARGET_GSH": "GSH",
    "MULTITARGET_Protein": "Protein",
}


def reactivity_onnx_rows(atom_rows: list[dict]) -> list[dict]:
    """Duplicate feature keys with ``__New_Topological`` for ONNX column names."""
    out: list[dict] = []
    for row in atom_rows:
        aliased = dict(row)
        for key, val in row.items():
            if key.startswith("_") or "MULTITARGET" in key:
                continue
            aliased[f"{key}__New_Topological"] = val
        out.append(aliased)
    return out


def reactivity_mol_features(
    atom_rows: list[dict],
    scores_by_head: dict[str, Sequence[float]],
    mol_names: list[str],
) -> np.ndarray:
    """One mol-head row matching legacy ``MolDesc.run(..., 'Training__MolDesc__AtomTop5')``."""
    values: dict[str, float] = {}

    for key in (*MOL_DESC, "heavy_atoms", "NumRings"):
        col = f"MolDesc__{key}"
        vals = [float(r.get(col, 0.0)) for r in atom_rows]
        values[f"{col}__New_Topological"] = max(vals) if vals else 0.0

    for target, label in _TOP_LABEL.items():
        head = next(k for k, v in _HEAD_TO_TARGET.items() if v == target)
        scores = sorted((float(s) for s in scores_by_head.get(head, [])), reverse=True)
        for k in range(1, 6):
            values[f"{label}_TopSite{k}"] = scores[k - 1] if k - 1 < len(scores) else 0.0

    return np.asarray([[float(values.get(n, 0.0)) for n in mol_names]], dtype=np.float64)


def reactivity_mol_output_index(head: str) -> int:
    return _MOL_OUT_INDEX[head]
