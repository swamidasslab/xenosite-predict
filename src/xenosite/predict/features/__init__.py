"""Compatibility package — modules live under ``v1/features/``."""

from __future__ import annotations

from pathlib import Path

__path__ = [
    str(Path(__file__).resolve().parent),
    str(Path(__file__).resolve().parent.parent / "v1" / "features"),
]

from ..v1.features import *  # noqa: F403
from ..v1.features import __all__ as __all__
