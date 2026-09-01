"""Backend protocol, picker, and env handling.

Picker (explicit env wins; first match):

1. ``XENOSITE_BACKEND`` is an ``http://`` / ``https://`` URL → HTTP backend
   (deployed xenosite-api). Optional ``XENOSITE_API_KEY`` as Bearer.
2. Else ``XENOSITE_MODELS_WEIGHTS`` → local ONNX directory.
3. Else auto-detect ``./weights/onnx/v0`` (or a flat ``./weights/onnx`` tree) → local ONNX.
4. Else user cache (``$XDG_CACHE_HOME/xenosite/onnx/v0``) if ``*.onnx`` exist.
5. Else, when ``XENOSITE_ONNX_URL`` is set, fetch that archive into the cache
   (an INFO line reports when weights are found or downloaded).
6. Else raise :class:`BackendNotConfigured`.

The archive URL is never compiled into this package; set ``XENOSITE_ONNX_URL``.
Tests must pass ``backend=`` / ``env={}`` and must not inherit a developer
shell. ``conftest.py`` clears ``XENOSITE_*`` unless a test opts in.
An isolated ``env={}`` does not auto-download.

Per-``(model, version)`` override: ``predict(..., backend=...)`` applies to all
models in the call; ``predict(..., backends={(name, version): backend})`` pins
individual models (so ONNX epoxidation can coexist with HTTP bioactivation).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping, Optional, Protocol, runtime_checkable

from ..errors import BackendNotConfigured
from ..weights import ENV_ONNX_URL, ENV_WEIGHTS, resolve_onnx_dir

ENV_BACKEND = "XENOSITE_BACKEND"
ENV_API_KEY = "XENOSITE_API_KEY"
ENV_LEGACY_URL = "XENOSITE_LEGACY_TEST_URL"

Spec = tuple[str, str]


@runtime_checkable
class PredictBackend(Protocol):
    """Swappable backend. Native results need not look like :class:`Molecule`."""

    name: str

    def available_models(self) -> list[Spec]:
        """``(name, version)`` pairs this process can actually run on this backend."""

    def predict_native(self, smiles: str, model: str, version: str) -> Any:
        """Run one model; return backend-native output for the adapter."""


def is_url(value: str) -> bool:
    return value.startswith("http://") or value.startswith("https://")


def resolve_backend(
    backend: Optional[str | PredictBackend] = None,
    *,
    env: Optional[Mapping[str, str]] = None,
    cwd: Optional[Path] = None,
    auto_download: Optional[bool] = None,
) -> PredictBackend:
    """Resolve a backend. ``env=None`` uses ``os.environ``; tests should pass a dict.

    ``auto_download`` defaults on only when ``env is None`` and
    ``XENOSITE_ONNX_URL`` is set. Isolated ``env`` mappings do not fetch.
    """
    if isinstance(backend, PredictBackend) and not isinstance(backend, str):
        return backend

    from .http import HttpBackend
    from .legacy import LegacyTestBackend
    from .onnx import OnnxBackend

    if isinstance(backend, str):
        if is_url(backend):
            return HttpBackend(backend)
        key = backend.lower()
        if key in {"onnx", "local"}:
            weights = resolve_onnx_dir(
                env=env, cwd=cwd, auto_download=auto_download, fallback=True
            )
            if weights is None:
                raise BackendNotConfigured(
                    "backend='onnx' needs local *.onnx files, "
                    f"{ENV_WEIGHTS}, or {ENV_ONNX_URL} (auto-downloaded on first use)."
                )
            return OnnxBackend(weights)
        if key in {"legacy", "legacy-test", "test-api"}:
            url = (env or os.environ).get(ENV_LEGACY_URL, "http://127.0.0.1:8099")
            return LegacyTestBackend(url)
        if key == "http":
            e = env if env is not None else os.environ
            url = e.get(ENV_BACKEND, "")
            if not is_url(url):
                raise BackendNotConfigured(
                    "backend='http' requires XENOSITE_BACKEND to be an http(s) URL"
                )
            return HttpBackend(url, api_key=e.get(ENV_API_KEY))
        raise BackendNotConfigured(f"Unknown backend {backend!r}")

    e = dict(os.environ if env is None else env)
    cwd = cwd or Path.cwd()

    url = e.get(ENV_BACKEND, "").strip()
    if is_url(url):
        return HttpBackend(url, api_key=e.get(ENV_API_KEY))

    weights = resolve_onnx_dir(env=env, cwd=cwd, auto_download=auto_download)
    if weights is not None:
        return OnnxBackend(weights)

    raise BackendNotConfigured(
        "No predictor backend configured. Set XENOSITE_BACKEND to an http(s) "
        "xenosite-api URL, XENOSITE_MODELS_WEIGHTS to an ONNX directory, put "
        "*.onnx files under ./weights/onnx/v0, or set XENOSITE_ONNX_URL "
        "(weights download on first predict())."
    )


def resolve_for_model(
    spec: Spec,
    *,
    backend: Optional[str | PredictBackend] = None,
    backends: Optional[Mapping[Spec, str | PredictBackend]] = None,
    env: Optional[Mapping[str, str]] = None,
    cwd: Optional[Path] = None,
) -> PredictBackend:
    """Per-model override, then the call-level backend, then env picker."""
    if backends and spec in backends:
        return resolve_backend(backends[spec], env=env, cwd=cwd)
    return resolve_backend(backend, env=env, cwd=cwd)
