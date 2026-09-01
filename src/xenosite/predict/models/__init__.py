"""Compatibility package — runners live under ``v0_legacy/models/``."""

from __future__ import annotations

from pathlib import Path

__path__ = [
    str(Path(__file__).resolve().parent),
    str(Path(__file__).resolve().parent.parent / "v0_legacy" / "models"),
]

from ..v0_legacy.models import load_all

__all__ = ["load_all"]
