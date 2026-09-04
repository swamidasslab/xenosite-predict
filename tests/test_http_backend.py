"""HTTP backend: msgpack, auth, concurrency, backoff, presentation contract."""

from __future__ import annotations

import asyncio
import math
from typing import Any

import httpx
import msgpack
import pytest

from xenosite.predict import list_models, predict, predict_many
from xenosite.predict.backends import ENV_API_KEY, resolve_backend
from xenosite.predict.backends.http import (
    HttpBackend,
    reset_http_state_for_tests,
    unpack_msgpack,
)
from xenosite.predict.errors import BackendRequestError, InvalidMolecule, ModelNotAvailable
from xenosite.predict.parallel import reset_pools_for_tests
from xenosite.predict.types import Molecule


ORIGIN = "https://example.test"


def _epoxidation_payload(smiles: str = "CCCCO", *, with_nan: bool = False) -> dict[str, Any]:
    mol = 0.1 if not with_nan else float("nan")
    bond = [0.01, 0.02, 0.03, 0.04]
    if with_nan:
        bond[1] = float("nan")
    return {
        "smiles": smiles,
        "name": {},
        "atoms": {
            "num": 5,
            "z": [6, 6, 6, 6, 8],
            "chrg": [0, 0, 0, 0, 0],
            "impHs": [3, 2, 2, 2, 1],
            "cipRank": [0, 2, 4, 3, 1],
            "reordered": [4, 3, 2, 1, 0],
        },
        "bonds": {
            "idx": [[0, 1], [1, 2], [2, 3], [3, 4]],
            "order": [1.0, 1.0, 1.0, 1.0],
        },
        "results": [
            {
                "model": "epoxidation",
                "model_version": "1",
                "mol": mol,
                "bond": bond,
            }
        ],
    }


def _ugt_payload() -> dict[str, Any]:
    return {
        "smiles": "CCCCO",
        "name": {},
        "atoms": {"num": 5, "z": [6, 6, 6, 6, 8], "reordered": [4, 3, 2, 1, 0]},
        "bonds": {"idx": [[0, 1], [1, 2], [2, 3], [3, 4]], "order": [1.0] * 4},
        "results": [
            {
                "model": "ugt",
                "model_version": "1",
                "atom": [0.0, 0.0, 0.0, 0.0, 0.99],
            }
        ],
    }


@pytest.fixture(autouse=True)
def _reset_http():
    reset_pools_for_tests()
    reset_http_state_for_tests()
    yield
    reset_pools_for_tests()
    reset_http_state_for_tests()


def _msgpack_response(payload: dict, status: int = 200) -> httpx.Response:
    return httpx.Response(
        status,
        content=msgpack.packb(payload),
        headers={"content-type": "application/x-msgpack"},
    )


def test_unpack_msgpack_preserves_nan():
    """NaN must round-trip via msgpack (JSON cannot); strict_map_key=False."""
    packed = msgpack.packb({"x": float("nan"), 1: "int-key"})
    out = unpack_msgpack(packed)
    assert math.isnan(out["x"])
    assert out[1] == "int-key"


def test_http_accepts_msgpack_and_canonical_detailed(monkeypatch):
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        assert request.headers["accept"] == "application/x-msgpack"
        assert request.url.params["smiles"] == "CCCCO"
        assert request.url.params["detailed"] == "true"
        assert "authorization" not in {k.lower(): v for k, v in request.headers.items()} or (
            request.headers.get("authorization") is None
        )
        return _msgpack_response(_epoxidation_payload())

    be = HttpBackend(ORIGIN, transport=httpx.MockTransport(handler), retry_max=1)
    mol = predict("OCCCC", model="epoxidation", backend=be, detailed=False)
    assert mol.smiles == "CCCCO"
    assert mol.atoms.z is None  # stripped
    assert mol.results[0].model == "epoxidation"
    assert seen and "/v1/epoxidation" in str(seen[0].url)


def test_bearer_from_env_on_url_backend():
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers.get("authorization", ""))
        return _msgpack_response(_epoxidation_payload("CCO"))

    env = {ENV_API_KEY: "secret-key"}
    be = resolve_backend(ORIGIN, env=env)
    assert isinstance(be, HttpBackend)
    assert be.api_key == "secret-key"
    be._transport = httpx.MockTransport(handler)
    predict("CCO", model="epoxidation", backend=be, env=env)
    assert seen == ["Bearer secret-key"]


def test_bearer_absent_when_no_key():
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return _msgpack_response(_epoxidation_payload("CCO"))

    be = HttpBackend(ORIGIN, transport=httpx.MockTransport(handler), retry_max=1)
    predict("CCO", model="epoxidation", backend=be)
    assert "authorization" not in seen[0].headers


def test_v0_vs_v1_routes():
    paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        paths.append(request.url.path)
        payload = _epoxidation_payload("CCO")
        payload["results"][0]["model_version"] = "0" if "/v0/" in request.url.path else "1"
        return _msgpack_response(payload)

    be = HttpBackend(ORIGIN, transport=httpx.MockTransport(handler), retry_max=1)
    predict("CCO", models=[("epoxidation", "0")], backend=be)
    predict("CCO", models=[("epoxidation", "1")], backend=be)
    assert paths[0].endswith("/v0/epoxidation")
    assert paths[1].endswith("/v1/epoxidation")


def test_bioactivation_unavailable():
    be = HttpBackend(ORIGIN, transport=httpx.MockTransport(lambda r: _msgpack_response({})), retry_max=1)
    with pytest.raises(ModelNotAvailable, match="bioactivation"):
        predict("CCO", models=[("bioactivation", "0")], backend=be)
    rows = list_models(backend=be)
    bio = [r for r in rows if r["name"] == "bioactivation"]
    assert bio and all(not r["available"] for r in bio)


def test_parameter_errors_on_http():
    be = HttpBackend(ORIGIN, transport=httpx.MockTransport(lambda r: _msgpack_response(_epoxidation_payload("CCO"))), retry_max=1)
    with pytest.raises(ModelNotAvailable, match="_parameter"):
        predict("CCO", model="epoxidation", backend=be, _parameter={"ndealk_site_mode": "legacy"})


def test_canonicalize_false_remaps_and_rdkit_warns():
    def handler(request: httpx.Request) -> httpx.Response:
        return _msgpack_response(_ugt_payload())

    be = HttpBackend(ORIGIN, transport=httpx.MockTransport(handler), retry_max=1)
    with pytest.warns(UserWarning, match="rdkit=True"):
        mol = predict(
            "OCCCC",
            model="ugt",
            backend=be,
            canonicalize=False,
            detailed=True,
            rdkit=True,
        )
    assert mol.smiles == "OCCCC"
    assert mol.results[0].atom[0] == pytest.approx(0.99)
    assert mol.atoms.z[0] == 8
    assert mol.rdkit is not None


def test_noncanonical_molecule_with_results_errors():
    be = HttpBackend(ORIGIN, transport=httpx.MockTransport(lambda r: _msgpack_response(_epoxidation_payload())), retry_max=1)
    bad = Molecule.model_validate(_epoxidation_payload("OCCCC"))
    # Force non-canonical smiles with existing results
    bad.smiles = "OCCCC"
    with pytest.raises(InvalidMolecule, match="non-canonical"):
        predict(bad, model="ugt", backend=be)


def test_backoff_retries_429(monkeypatch):
    attempts = {"n": 0}
    sleeps: list[float] = []

    async def fake_sleep(delay: float) -> None:
        sleeps.append(delay)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    monkeypatch.setattr("xenosite.predict.backends.http.random.uniform", lambda a, b: b)

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        if attempts["n"] < 3:
            return httpx.Response(429, text="slow down")
        return _msgpack_response(_epoxidation_payload("CCO"))

    be = HttpBackend(
        ORIGIN,
        transport=httpx.MockTransport(handler),
        retry_max=5,
        retry_base=0.5,
        retry_cap=10.0,
    )
    mol = predict("CCO", model="epoxidation", backend=be)
    assert mol.results
    assert attempts["n"] == 3
    assert len(sleeps) == 2
    assert sleeps[0] == pytest.approx(0.5)  # min(cap, 0.5 * 2**0)
    assert sleeps[1] == pytest.approx(1.0)


def test_concurrency_cap():
    in_flight = 0
    max_flight = 0
    lock = asyncio.Lock()

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal in_flight, max_flight
        async with lock:
            in_flight += 1
            max_flight = max(max_flight, in_flight)
        await asyncio.sleep(0.05)
        async with lock:
            in_flight -= 1
        return _msgpack_response(_epoxidation_payload("CCO"))

    # httpx.MockTransport expects sync handler; use a queue + sync sleep instead.
    import time

    sync_flight = {"n": 0, "max": 0}

    def sync_handler(request: httpx.Request) -> httpx.Response:
        sync_flight["n"] += 1
        sync_flight["max"] = max(sync_flight["max"], sync_flight["n"])
        time.sleep(0.05)
        sync_flight["n"] -= 1
        return _msgpack_response(_epoxidation_payload("CCO"))

    reset_http_state_for_tests()
    be = HttpBackend(
        ORIGIN,
        transport=httpx.MockTransport(sync_handler),
        max_concurrent=2,
        retry_max=1,
    )
    predict_many(["CCO"] * 6, model="epoxidation", backend=be)
    assert sync_flight["max"] <= 2


def test_http_error_wrapped():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    be = HttpBackend(ORIGIN, transport=httpx.MockTransport(handler), retry_max=1)
    with pytest.raises(BackendRequestError, match="500"):
        predict("CCO", model="epoxidation", backend=be)


def test_msgpack_nan_in_scores():
    def handler(request: httpx.Request) -> httpx.Response:
        return _msgpack_response(_epoxidation_payload("CCO", with_nan=True))

    be = HttpBackend(ORIGIN, transport=httpx.MockTransport(handler), retry_max=1)
    mol = predict("CCO", model="epoxidation", backend=be)
    assert math.isnan(mol.results[0].mol)
    assert math.isnan(mol.results[0].bond[1])


def test_list_models_http_isozyme():
    be = HttpBackend(ORIGIN)
    rows = { (r["name"], r["version"]): r for r in list_models(backend=be) }
    assert rows[("isozyme", "1")]["available"] is True
    assert rows[("epoxidation", "0")]["available"] is True
    assert rows[("bioactivation", "0")]["available"] is False
