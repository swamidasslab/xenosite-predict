"""Importable test helpers (conftest fixtures stay in conftest.py)."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = Path(__file__).resolve().parent / "fixtures" / "golden_smiles.json"


def onnx_weights_present(model: str | None = None) -> bool:
    root = ROOT / "weights" / "onnx"
    if model:
        return any((root / model).glob("*.onnx"))
    return any(root.rglob("*.onnx"))


def load_golden():
    if not GOLDEN.is_file():
        return []
    return json.loads(GOLDEN.read_text(encoding="utf-8"))


def compare_rdkit_ob_rows(rdkit_rows: list[dict], dump: dict, *, atol: float = 1e-4) -> list[str]:
    """Return names of overlapping columns that disagree (atol 1e-4).

    Used by ``@pytest.mark.live`` tests that dump OpenBabel features from the
    legacy test image. Do not loosen atol to hide descriptor drift.
    """
    import numpy as np

    cols = dump.get("columns") or []
    ob_rows = dump.get("rows") or []
    mismatches: list[str] = []
    for i, ob in enumerate(ob_rows):
        if i >= len(rdkit_rows):
            break
        ob_map = dict(zip(cols, ob))
        for c, val in ob_map.items():
            if c not in rdkit_rows[i]:
                continue
            try:
                if not np.isclose(float(val), float(rdkit_rows[i][c]), atol=atol):
                    mismatches.append(c)
            except (TypeError, ValueError):
                continue
    return mismatches
