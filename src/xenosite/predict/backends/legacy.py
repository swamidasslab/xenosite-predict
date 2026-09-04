"""Legacy Python-2 test-API backend (pytest live fixture).

Not production Flask. Endpoints:

- ``GET /health``
- ``POST /predict/<model>``  JSON ``{"smiles": ...}``
- ``POST /nn/<model>/<head>`` JSON matrix for random-vector parity
"""

from __future__ import annotations

from typing import Any

import httpx

from ..errors import UnknownModel

_MODEL_NAMES = (
    "epoxidation",
    "quinone",
    "reactivity",
    "ugt",
    "ndealk",
    "isozyme",
    "phase1",
    "bioactivation",
)
_MODELS = [(name, version) for name in _MODEL_NAMES for version in ("0", "1")]


class LegacyTestBackend:
    """POST to the derived ``legacy-test-api`` container."""

    name = "legacy"

    def __init__(self, origin: str, *, timeout: float = 120.0):
        self.origin = origin.rstrip("/")
        self.timeout = timeout

    def available_models(self) -> list[tuple[str, str]]:
        return list(_MODELS)

    def predict_native(self, smiles: str, model: str, version: str) -> Any:
        if (model, version) not in _MODELS:
            raise UnknownModel(f"legacy test API has no {model!r} {version!r}")
        url = f"{self.origin}/predict/{model}"
        with httpx.Client(timeout=self.timeout) as client:
            r = client.post(url, json={"smiles": smiles})
            r.raise_for_status()
            return r.json()

    def nn(self, model: str, head: str, matrix: list[list[float]]) -> Any:
        url = f"{self.origin}/nn/{model}/{head}"
        with httpx.Client(timeout=self.timeout) as client:
            r = client.post(url, json={"x": matrix})
            r.raise_for_status()
            return r.json()

    def dump_features(self, smiles: str, model: str) -> Any:
        url = f"{self.origin}/features/{model}"
        with httpx.Client(timeout=self.timeout) as client:
            r = client.post(url, json={"smiles": smiles})
            r.raise_for_status()
            return r.json()

    def health(self) -> bool:
        try:
            with httpx.Client(timeout=5.0) as client:
                r = client.get(f"{self.origin}/health")
                return r.status_code == 200
        except httpx.HTTPError:
            return False
