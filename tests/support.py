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
