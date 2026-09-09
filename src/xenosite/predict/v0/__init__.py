"""Deprecated legacy scoring overlay. Prefer :mod:`xenosite.predict.v1`.

Importing this package warns. Scoring version ``"0"`` still works via
``predict(..., models=[(name, "0")])`` without importing here.
"""

from __future__ import annotations

import warnings

warnings.warn(
    "xenosite.predict.v0 is deprecated; import xenosite.predict.v1. "
    "Scoring version '0' (legacy mapping) remains available as "
    "predict(..., models=[(name, '0')]).",
    DeprecationWarning,
    stacklevel=2,
)

from xenosite.predict.v1.legacy import LegacyRunner, register_v0_models

__all__ = ["LegacyRunner", "register_v0_models"]
