"""Compatibility package — runners live under ``v1/models/``."""

from __future__ import annotations

from pathlib import Path

__path__ = [
    str(Path(__file__).resolve().parent),
    str(Path(__file__).resolve().parent.parent / "v1" / "models"),
]

from ..v1.models import load_all as _load_v1
from ..v1.legacy import register_v0_models as _register_v0

_LOADED = False


def load_all() -> None:
    global _LOADED
    if _LOADED:
        return
    _load_v1()
    _register_v0()
    _LOADED = True


__all__ = ["load_all"]
