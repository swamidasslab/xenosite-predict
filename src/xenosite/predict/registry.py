"""Built-in ``(name, version)`` registry with lazy factories.

``models=`` accepts a string (default version ``"0"``) or ``(name, version)``
pairs. Do not apply one version string across a list. Entry-point plugins are
deferred until a built-in ONNX model actually runs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional

from .errors import UnknownModel

Spec = tuple[str, str]
Factory = Callable[[], "ModelRunner"]


@dataclass
class ModelInfo:
    name: str
    version: str
    factory: Factory
    default: bool = True
    # Optional notes used by list_models / docs
    blocked_reason: Optional[str] = None
    heads: tuple[str, ...] = ()
    two_stage: bool = False
    pipeline: bool = False


class ModelRunner:
    """Lazy per-model object. Importing the package does not load ONNX."""

    name: str
    version: str

    def available(self, backend) -> bool:  # pragma: no cover - interface
        raise NotImplementedError

    def predict_molecule(self, molecule, backend) -> None:
        raise NotImplementedError


_REGISTRY: dict[Spec, ModelInfo] = {}


def register_model(
    name: str,
    version: str = "0",
    *,
    factory: Factory,
    default: bool = True,
    blocked_reason: Optional[str] = None,
    heads: tuple[str, ...] = (),
    two_stage: bool = False,
    pipeline: bool = False,
) -> None:
    """Internal registration. Not a public plugin API yet."""
    _REGISTRY[(name, version)] = ModelInfo(
        name=name,
        version=version,
        factory=factory,
        default=default,
        blocked_reason=blocked_reason,
        heads=heads,
        two_stage=two_stage,
        pipeline=pipeline,
    )


def normalize_models(
    models: Optional[str | Spec | Iterable[str | Spec]] = None,
    *,
    default: str = "epoxidation",
) -> list[Spec]:
    """Turn ``models=`` into a list of ``(name, version)`` pairs."""
    if models is None:
        models = default
    if isinstance(models, str):
        return [(models, default_version(models))]
    if isinstance(models, tuple) and len(models) == 2 and all(isinstance(x, str) for x in models):
        return [models]  # type: ignore[return-value]
    out: list[Spec] = []
    for item in models:  # type: ignore[union-attr]
        if isinstance(item, str):
            out.append((item, default_version(item)))
        elif isinstance(item, tuple) and len(item) == 2:
            out.append((str(item[0]), str(item[1])))
        else:
            raise UnknownModel(f"Invalid models entry: {item!r}")
    return out


def default_version(name: str) -> str:
    for (n, v), info in _REGISTRY.items():
        if n == name and info.default:
            return v
    # Unknown names still get "0" so the error is UnknownModel at lookup
    return "0"


def get_info(name: str, version: str) -> ModelInfo:
    try:
        return _REGISTRY[(name, version)]
    except KeyError as exc:
        known = ", ".join(f"{n}:{v}" for n, v in sorted(_REGISTRY)) or "(empty)"
        raise UnknownModel(
            f"Unknown model {name!r} version {version!r}. Known: {known}"
        ) from exc


def load_runner(name: str, version: str) -> ModelRunner:
    info = get_info(name, version)
    return info.factory()


def registered() -> list[ModelInfo]:
    return list(_REGISTRY.values())


def ensure_builtins() -> None:
    """Import model modules so they register. Safe to call more than once."""
    from . import models as _models  # noqa: F401

    _models.load_all()
