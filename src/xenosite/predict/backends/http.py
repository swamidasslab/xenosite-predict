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

# xenosite-api v0 routes (query: ?smiles=)
_V0_ROUTES: dict[tuple[str, str], str] = {
    ("epoxidation", "0"): "/v0/epoxidation",
    ("quinone", "0"): "/v0/quinone",
    ("ugt", "0"): "/v0/ugt",
    ("ndealk", "0"): "/v0/ndealk",
    ("isozyme", "0"): "/v0/isozyme",
    ("phase1", "0"): "/v0/phase1",
    ("bioactivation", "0"): "/v0/bioactivation",
    ("reactivity", "0"): "/v0/reactivity",
}


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
        return list(_V0_ROUTES)

    def predict_native(self, smiles: str, model: str, version: str) -> Any:
        route = _V0_ROUTES.get((model, version))
        if route is None:
            raise UnknownModel(f"HTTP backend has no route for {model!r} {version!r}")
        url = f"{self.origin}{route}"
        with httpx.Client(timeout=self.timeout, headers=self._headers()) as client:
            r = client.get(url, params={"smiles": smiles})
            r.raise_for_status()
            return r.json()
