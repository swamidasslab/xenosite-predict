"""Two-stage mol-head features: Top-N site scores plus aggregated site descriptors.

Used by epoxidation, quinone, and reactivity. Site ONNX runs first; mol ONNX
consumes ``Top{{k}}__AtomScore`` (or BondScore) plus MAX/MIN of site features.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np


def topn_site_features(
    site_scores: Sequence[float],
    site_rows: list[dict],
    mol_names: list[str],
    *,
    topn_prefix: str = "Top",
    score_suffix: str = "__AtomScore",
    mol_desc_suffix: str = "__MoleculeDescriptor",
    fill: float = 0.0,
) -> np.ndarray:
    """Build one mol-head row (shape ``(1, n_features)``) from site outputs."""
    scores = np.asarray(site_scores, dtype=np.float64)
    order = np.argsort(-scores)
    topn = 0
    for name in mol_names:
        if name.endswith(score_suffix) and name.startswith(topn_prefix):
            try:
                k = int(name.split("__")[0].replace(topn_prefix, ""))
                topn = max(topn, k)
            except ValueError:
                continue

    values: dict[str, float] = {}
    for k in range(1, topn + 1):
        key = f"{topn_prefix}{k}{score_suffix}"
        values[key] = float(scores[order[k - 1]]) if k - 1 < len(scores) else fill

    numeric_keys = []
    if site_rows:
        numeric_keys = [k for k, v in site_rows[0].items() if isinstance(v, (int, float)) and not k.startswith("_")]

    maxs: dict[str, float] = {}
    mins: dict[str, float] = {}
    for k in numeric_keys:
        col = [float(r.get(k, fill)) for r in site_rows]
        maxs[k] = max(col) if col else fill
        mins[k] = min(col) if col else fill

    for name in mol_names:
        if name.endswith(mol_desc_suffix):
            base = name.split(mol_desc_suffix)[0]
            values[name] = maxs.get(base, fill)
        elif name.endswith("__MAX"):
            values[name] = maxs.get(name[: -len("__MAX")], fill)
        elif name.endswith("__MIN"):
            values[name] = mins.get(name[: -len("__MIN")], fill)
        elif name in values:
            continue
        else:
            values[name] = fill

    return np.asarray([[values.get(n, fill) for n in mol_names]], dtype=np.float64)
