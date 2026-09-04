"""HTTP backend against a deployed xenosite-api (not the legacy Flask app).

All I/O uses :class:`httpx.AsyncClient`. Responses are requested as
**msgpack** (``Accept: application/x-msgpack``) so IEEE-754 NaN/Inf in scores
round-trip (JSON cannot encode them). Unpack with ``strict_map_key=False``.

``XENOSITE_BACKEND`` is the API origin (no trailing path required).
``XENOSITE_API_KEY`` is sent as ``Authorization: Bearer …`` when set.
``XENOSITE_HTTP_MAX_CONCURRENT`` caps in-flight requests per origin (default 64).

This backend never loads ONNX. Live parity tests must pin ONNX vs the
legacy test-API, not vs production HTTP.
"""

from __future__ import annotations

import asyncio
import os
import random
import threading
from typing import Any, Mapping, Optional

import httpx
import msgpack

from ..errors import BackendRequestError, ModelNotAvailable, UnknownModel

# xenosite-api routes (query: ?smiles=&detailed=). Scoring generation:
# ``"0"`` → ``/v0``, ``"1"`` → ``/v1``. bioactivation is not exposed on HTTP.
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
_V1_ROUTES: dict[tuple[str, str], str] = {
    (name, "1"): f"/v1/{name}" for name in _HTTP_MODELS
}
_ROUTES: dict[tuple[str, str], str] = {**_V0_ROUTES, **_V1_ROUTES}

ENV_HTTP_MAX_CONCURRENT = "XENOSITE_HTTP_MAX_CONCURRENT"
ENV_HTTP_RETRY_BASE = "XENOSITE_HTTP_RETRY_BASE"
ENV_HTTP_RETRY_CAP = "XENOSITE_HTTP_RETRY_CAP"
ENV_HTTP_RETRY_MAX = "XENOSITE_HTTP_RETRY_MAX"

DEFAULT_MAX_CONCURRENT = 64
DEFAULT_RETRY_BASE = 0.25
DEFAULT_RETRY_CAP = 30.0
DEFAULT_RETRY_MAX = 5

_RETRY_STATUS = frozenset({408, 429, 503})
_MSGPACK_ACCEPT = "application/x-msgpack"

# Shared per-origin+loop AsyncClients (process-wide).
_lock = threading.Lock()
_clients: dict[tuple[str, int], httpx.AsyncClient] = {}
_semaphores: dict[str, threading.BoundedSemaphore] = {}
_sem_limits: dict[str, int] = {}


def _env_int(name: str, default: int, env: Optional[Mapping[str, str]] = None) -> int:
    e = os.environ if env is None else env
    raw = (e.get(name) or "").strip()
    return max(1, int(raw)) if raw else default


def _env_float(name: str, default: float, env: Optional[Mapping[str, str]] = None) -> float:
    e = os.environ if env is None else env
    raw = (e.get(name) or "").strip()
    return float(raw) if raw else default


def unpack_msgpack(data: bytes) -> Any:
    """Decode API msgpack payload.

    ``strict_map_key=False`` matches xenosite-api clients. Float NaN/Inf decode
    as IEEE-754 values (msgpack packs them; JSON cannot).
    """
    return msgpack.unpackb(data, raw=False, strict_map_key=False)


def _backoff_sleep(attempt: int, *, base: float, cap: float) -> float:
    """Full-jitter delay: ``uniform(0, min(cap, base * 2**attempt))``."""
    return random.uniform(0.0, min(cap, base * (2**attempt)))


def _scrub_error_message(exc: BaseException, api_key: Optional[str]) -> str:
    text = str(exc)
    if api_key and api_key in text:
        text = text.replace(api_key, "***")
    return text


async def _get_client(
    origin: str,
    headers: dict[str, str],
    timeout: float,
    transport: Optional[httpx.AsyncBaseTransport] = None,
) -> httpx.AsyncClient:
    # Per-instance transport (tests) bypasses the shared cache.
    if transport is not None:
        return httpx.AsyncClient(
            base_url=origin,
            headers=headers,
            timeout=timeout,
            transport=transport,
        )
    loop = asyncio.get_running_loop()
    key = (origin, id(loop))
    with _lock:
        client = _clients.get(key)
        if client is None or client.is_closed:
            client = httpx.AsyncClient(
                base_url=origin,
                headers=headers,
                timeout=timeout,
            )
            _clients[key] = client
        return client


def _get_semaphore(origin: str, limit: int) -> threading.BoundedSemaphore:
    """Process-wide cap; threading so sync bridges and asyncio.run loops share it."""
    with _lock:
        sem = _semaphores.get(origin)
        if sem is None or _sem_limits.get(origin) != limit:
            sem = threading.BoundedSemaphore(limit)
            _semaphores[origin] = sem
            _sem_limits[origin] = limit
        return sem


def reset_http_state_for_tests() -> None:
    """Close shared clients/semaphores (test helper)."""
    global _clients, _semaphores, _sem_limits
    with _lock:
        clients = list(_clients.values())
        _clients = {}
        _semaphores = {}
        _sem_limits = {}

    async def _close_all() -> None:
        for c in clients:
            if not c.is_closed:
                await c.aclose()

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        if clients:
            asyncio.run(_close_all())


def run_sync(coro):
    """Run ``coro`` from sync code (``predict`` bridge)."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    # Already in an async context — run in a fresh loop on a worker thread.
    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


class HttpBackend:
    """Async GET ``{origin}{route}?smiles=&detailed=true`` → msgpack Molecule dict."""

    name = "http"

    def __init__(
        self,
        origin: str,
        api_key: Optional[str] = None,
        *,
        timeout: float = 60.0,
        max_concurrent: Optional[int] = None,
        retry_base: Optional[float] = None,
        retry_cap: Optional[float] = None,
        retry_max: Optional[int] = None,
        env: Optional[Mapping[str, str]] = None,
        transport: Optional[httpx.AsyncBaseTransport] = None,
    ):
        self.origin = origin.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self._transport = transport
        e = env
        self.max_concurrent = (
            max_concurrent
            if max_concurrent is not None
            else _env_int(ENV_HTTP_MAX_CONCURRENT, DEFAULT_MAX_CONCURRENT, e)
        )
        self.retry_base = (
            retry_base
            if retry_base is not None
            else _env_float(ENV_HTTP_RETRY_BASE, DEFAULT_RETRY_BASE, e)
        )
        self.retry_cap = (
            retry_cap
            if retry_cap is not None
            else _env_float(ENV_HTTP_RETRY_CAP, DEFAULT_RETRY_CAP, e)
        )
        self.retry_max = (
            retry_max
            if retry_max is not None
            else _env_int(ENV_HTTP_RETRY_MAX, DEFAULT_RETRY_MAX, e)
        )

    def _headers(self) -> dict[str, str]:
        h = {"accept": _MSGPACK_ACCEPT}
        if self.api_key:
            h["authorization"] = f"Bearer {self.api_key}"
        return h

    def available_models(self) -> list[tuple[str, str]]:
        return list(_ROUTES)

    def predict_native(self, smiles: str, model: str, version: str) -> Any:
        """Sync bridge to :meth:`apredict_native`."""
        return run_sync(self.apredict_native(smiles, model, version))

    async def apredict_native(self, smiles: str, model: str, version: str) -> Any:
        route = _ROUTES.get((model, version))
        if route is None:
            if model == "bioactivation":
                raise ModelNotAvailable(
                    "bioactivation is not available on the HTTP backend "
                    "(no /v0 or /v1 route on xenosite-api)"
                )
            raise UnknownModel(f"HTTP backend has no route for {model!r} {version!r}")

        client = await _get_client(
            self.origin, self._headers(), self.timeout, transport=self._transport
        )
        ephemeral = self._transport is not None
        sem = _get_semaphore(self.origin, self.max_concurrent)
        params = {"smiles": smiles, "detailed": "true"}
        last_exc: BaseException | None = None

        try:
            # BoundedSemaphore is sync; acquire in a thread so we do not block the loop.
            await asyncio.to_thread(sem.acquire)
            try:
                for attempt in range(self.retry_max):
                    try:
                        r = await client.get(route, params=params)
                        if r.status_code in _RETRY_STATUS:
                            last_exc = httpx.HTTPStatusError(
                                f"HTTP {r.status_code}",
                                request=r.request,
                                response=r,
                            )
                            if attempt + 1 >= self.retry_max:
                                break
                            await asyncio.sleep(
                                _backoff_sleep(
                                    attempt, base=self.retry_base, cap=self.retry_cap
                                )
                            )
                            continue
                        if r.status_code >= 400:
                            raise BackendRequestError(
                                f"HTTP {r.status_code} for {model!r} {version!r}: "
                                f"{_scrub_error_message(Exception(r.text[:200]), self.api_key)}"
                            )
                        return unpack_msgpack(r.content)
                    except (
                        httpx.TimeoutException,
                        httpx.NetworkError,
                        httpx.RemoteProtocolError,
                    ) as exc:
                        last_exc = exc
                        if attempt + 1 >= self.retry_max:
                            break
                        await asyncio.sleep(
                            _backoff_sleep(
                                attempt, base=self.retry_base, cap=self.retry_cap
                            )
                        )
                        continue

                msg = _scrub_error_message(
                    last_exc or Exception("request failed"), self.api_key
                )
                raise BackendRequestError(
                    f"HTTP request failed for {model!r} {version!r} after "
                    f"{self.retry_max} attempts: {msg}"
                )
            finally:
                sem.release()
        finally:
            if ephemeral and not client.is_closed:
                await client.aclose()
