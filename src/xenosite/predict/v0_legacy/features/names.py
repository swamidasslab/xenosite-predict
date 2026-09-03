"""Load committed feature-column order. Inference never opens TSV or JSON."""

from __future__ import annotations

from typing import Optional

import numpy as np

from .name_tables import TABLES


def load_names(model: str, head: str) -> Optional[list[str]]:
    """Return ordered feature names for ``model/head``, or None if unknown."""
    names = TABLES.get((model, head))
    if names is None:
        return None
    return list(names)


def select_columns(row: dict[str, float], names: list[str], *, fill: float = 0.0) -> np.ndarray:
    vals = []
    for n in names:
        v = row.get(n, fill)
        try:
            vals.append(float(v))
        except (TypeError, ValueError):
            vals.append(fill)
    return np.asarray(vals, dtype=np.float64)


def matrix_from_rows(
    rows: list[dict[str, float]], names: Optional[list[str]], *, fill: float = 0.0
) -> tuple[np.ndarray, list[str]]:
    """Stack rows as ``(n_patterns, n_features)``. If names is None, use sorted keys."""
    if not rows:
        return np.zeros((0, 0), dtype=np.float64), []
    if names is None:
        keys: list[str] = []
        seen = set()
        for row in rows:
            for k in row:
                if k.startswith("_") or k in seen:
                    continue
                seen.add(k)
                keys.append(k)
        names = keys
    mat = np.stack([select_columns(r, names, fill=fill) for r in rows], axis=0)
    return mat, names
