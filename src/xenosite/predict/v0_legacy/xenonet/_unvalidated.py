"""Warn that XenoNet is unvalidated until golden fixtures exist."""

from __future__ import annotations

import warnings

_UNVALIDATED = (
    "xenosite.predict.v0_legacy.xenonet is not validated until golden "
    "fixtures exist. Scores and graph structure may change."
)

_emitted = False


def warn_unvalidated() -> None:
    global _emitted
    if _emitted:
        return
    _emitted = True
    warnings.warn(_UNVALIDATED, UserWarning, stacklevel=3)
