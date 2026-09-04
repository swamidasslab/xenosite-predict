"""HTTP backend against a deployed xenosite-api (not the legacy Flask app).

``XENOSITE_BACKEND`` is the API origin (no trailing path required).
``XENOSITE_API_KEY`` is sent as ``Authorization: Bearer …`` when set.

This backend never loads ONNX. Live parity tests must pin ONNX vs the
legacy test-API, not vs production HTTP.
"""

from __future__ import annotations

from typing import Any, Optional
from urllib.parse import quote

import httpx

from ..errors import UnknownModel
from ..types import Molecule

# xenosite-api routes (query: ?smiles=). Version is the scoring generation:
# ``"0"`` → ``/v0`` (legacy params), ``"1"`` → ``/v1`` (updated params).
_HTTP_MODELS = (
    "epoxidation",
    "quinone",
    "ugt",
    "ndealk",
    "isozyme",
    "phase1",
    "reactivity",
)
_V0_ROUTES: dict[tuple[str, str], str] = {
    (name, "0"): f"/v0/{name}" for name in _HTTP_MODELS
}
_V0_ROUTES[("bioactivation", "0")] = "/v0/bioactivation"
_V1_ROUTES: dict[tuple[str, str], str] = {
    (name, "1"): f"/v1/{name}" for name in _HTTP_MODELS
}
_ROUTES: dict[tuple[str, str], str] = {**_V0_ROUTES, **_V1_ROUTES}


class HttpBackend:
    """GET ``{origin}{route}?smiles=`` and return a Molecule-shaped dict."""

    name = "http"

    def __init__(self, origin: str, api_key: Optional[str] = None, *, timeout: float = 60.0):
        self.origin = origin.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    def _headers(self) -> dict[str, str]:
        h = {"accept": "application/json"}
        if self.api_key:
            h["authorization"] = f"Bearer {self.api_key}"
        return h

    def available_models(self) -> list[tuple[str, str]]:
        return list(_ROUTES)

    def predict_native(self, smiles: str, model: str, version: str) -> Any:
        route = _ROUTES.get((model, version))
        if route is None:
            raise UnknownModel(f"HTTP backend has no route for {model!r} {version!r}")
        url = f"{self.origin}{route}"
        with httpx.Client(timeout=self.timeout, headers=self._headers()) as client:
            r = client.get(url, params={"smiles": smiles})
            r.raise_for_status()
            return r.json()
