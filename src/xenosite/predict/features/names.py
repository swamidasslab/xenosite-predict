"""Load committed feature-name JSON (column order). Inference never opens TSV."""

from __future__ import annotations

import json
from importlib import resources
from pathlib import Path
from typing import Optional

import numpy as np


def load_names(model: str, head: str) -> Optional[list[str]]:
    """Return ordered feature names for ``model/head``, or None if not committed yet."""
    pkg = "xenosite.predict.features"
    filename = f"{model}_{head}_names.json"
    try:
        data = resources.files(pkg).joinpath(filename).read_text(encoding="utf-8")
    except (FileNotFoundError, ModuleNotFoundError, AttributeError):
        path = Path(__file__).with_name(filename)
        if not path.is_file():
            return None
        data = path.read_text(encoding="utf-8")
    names = json.loads(data)
    if isinstance(names, dict):
        names = names.get("names") or names.get("columns")
    return list(names)


def select_columns(row: dict[str, float], names: list[str], *, fill: float = 0.0) -> np.ndarray:
    return np.asarray([float(row.get(n, fill)) for n in names], dtype=np.float64)


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
                if k not in seen:
                    seen.add(k)
                    keys.append(k)
        names = keys
    mat = np.stack([select_columns(r, names, fill=fill) for r in rows], axis=0)
    return mat, names
