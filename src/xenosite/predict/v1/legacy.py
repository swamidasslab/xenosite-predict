"""Scoring version ``"0"``: wrap v1 runners with legacy mapping defaults.

``predict(..., models=[(name, "0")])`` uses these wrappers. Importing this
module does not warn; :mod:`xenosite.predict.v0` does.
"""

from __future__ import annotations

from xenosite.predict.registry import ModelRunner, register_model, registered
from xenosite.predict.scoring import V0_PARAMETER

_LOADED = False


class LegacyRunner(ModelRunner):
    """v1 runner with legacy ``_parameter`` defaults and ``model_version`` ``"0"``."""

    def __init__(self, inner: ModelRunner):
        self._inner = inner
        self.name = inner.name
        self.version = "0"
        inner.version = "0"

    def _apply_legacy(self, molecule) -> None:
        for key, value in V0_PARAMETER.items():
            molecule._parameter.setdefault(key, value)

    def available(self, backend) -> bool:
        return self._inner.available(backend)

    def predict_molecule(self, molecule, backend) -> None:
        self._apply_legacy(molecule)
        self._inner.version = "0"
        self._inner.predict_molecule(molecule, backend)

    def from_onnx(self, molecule, backend) -> None:
        self._apply_legacy(molecule)
        self._inner.version = "0"
        return self._inner.from_onnx(molecule, backend)

    def from_legacy(self, molecule, native) -> None:
        self._apply_legacy(molecule)
        self._inner.version = "0"
        return self._inner.from_legacy(molecule, native)

    def __getattr__(self, name: str):
        return getattr(self._inner, name)


def register_v0_models() -> None:
    """Register scoring version ``"0"`` wrappers for every v1 model."""
    global _LOADED
    if _LOADED:
        return
    from xenosite.predict.v1.models import load_all as load_v1

    load_v1()
    seen: set[str] = set()
    for info in list(registered()):
        if info.version != "1" or info.name in seen:
            continue
        seen.add(info.name)
        factory = info.factory
        register_model(
            info.name,
            "0",
            factory=lambda f=factory: LegacyRunner(f()),
            default=False,
            blocked_reason=info.blocked_reason,
            heads=info.heads,
            two_stage=info.two_stage,
            pipeline=info.pipeline,
        )
    _LOADED = True
