"""Compatibility package — modules live under ``v0_legacy/features/``."""

from __future__ import annotations

from pathlib import Path

__path__ = [
    str(Path(__file__).resolve().parent),
    str(Path(__file__).resolve().parent.parent / "v0_legacy" / "features"),
]

from ..v0_legacy.features import *  # noqa: F403
from ..v0_legacy.features import __all__ as __all__
