"""Deprecated alias of :mod:`xenosite.predict.v1`."""

from __future__ import annotations

import warnings
from pathlib import Path

warnings.warn(
    "xenosite.predict.v0_legacy has moved to xenosite.predict.v1.",
    DeprecationWarning,
    stacklevel=2,
)

__path__ = [str(Path(__file__).resolve().parent.parent / "v1")]
