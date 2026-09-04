"""Parallel / async prediction helpers for high-throughput workloads.

Descriptor generation (OpenBabel / RDKit feature rows) dominates wall time and
is GIL-bound in-process. Process workers give real multi-core speedup; ONNX
Runtime sessions stay process-local (thread-safe ``Run`` is unused across
processes). Sync :func:`predict_many` and async :func:`apredict` /
:func:`apredict_many` all share the same job payload and worker entrypoint so
new models need no extra wiring beyond the existing :func:`predict` path.

HTTP backends use a thread pool (I/O-bound) instead of processes.
"""

from __future__ import annotations

import asyncio
import atexit
import multiprocessing as mp
import os
import threading
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Optional, Sequence, Union

from .api import predict
from .backends import PredictBackend, is_url, resolve_backend
from .backends.http import HttpBackend
from .backends.legacy import LegacyTestBackend
from .backends.onnx import ENV_ORT_INTRA, OnnxBackend
from .molecule import as_molecule
from .registry import Spec, ensure_builtins, normalize_models
from .types import Molecule

ModelsArg = Union[str, Spec, Iterable[str | Spec]]
BackendArg = Union[str, PredictBackend, None]
BackendMap = Optional[Mapping[Spec, str | PredictBackend]]
InputsArg = Sequence[str | Molecule]

ENV_WORKERS = "XENOSITE_WORKERS"

_process_pool: ProcessPoolExecutor | None = None
_process_pool_workers: int | None = None
_process_lock = threading.Lock()
_thread_pool: ThreadPoolExecutor | None = None
_thread_pool_workers: int | None = None
_thread_lock = threading.Lock()

# Per-process cache populated inside worker processes (and the parent when
# falling back to in-process execution).
_BACKEND_CACHE: dict[tuple[str, str], PredictBackend] = {}


@dataclass(frozen=True)
class _BackendSpec:
    """Picklable backend description reconstructed in each worker."""

    kind: str  # "onnx" | "http" | "legacy"
    arg: str  # weights dir, origin URL, …
    env: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class _PredictJob:
    smiles: str
    models: tuple[Spec, ...]
    backend: _BackendSpec
    metabolites: bool = False
    metabolites_min_score: float | None = None
    mapped_smiles: bool = False
    detailed: bool = False
    rdkit: bool = False
    parameter: tuple[tuple[str, Any], ...] = ()


def default_workers(env: Optional[Mapping[str, str]] = None) -> int:
    """Worker count: ``XENOSITE_WORKERS``, else CPU count (minimum 1)."""
    e = os.environ if env is None else env
    raw = (e.get(ENV_WORKERS) or "").strip()
    if raw:
        return max(1, int(raw))
    cpus = os.cpu_count() or 1
    return max(1, cpus)


def _ort_intra_op_for_workers(workers: int, env: Optional[Mapping[str, str]] = None) -> int:
    """Cap ORT threads per process so N workers do not oversubscribe cores."""
    e = os.environ if env is None else env
    raw = (e.get(ENV_ORT_INTRA) or "").strip()
    if raw:
        return max(1, int(raw))
    cpus = os.cpu_count() or 1
    return max(1, cpus // max(1, workers))


def _backend_spec(
    backend: BackendArg,
    *,
    env: Optional[Mapping[str, str]] = None,
) -> _BackendSpec:
    """Turn a user backend argument into a picklable spec."""
    env_items = tuple(sorted((env or {}).items()))
    if backend is None:
        be = resolve_backend(None, env=env)
        return _backend_spec(be, env=env)
    if isinstance(backend, OnnxBackend):
        return _BackendSpec("onnx", str(backend.root), env_items)
    if isinstance(backend, HttpBackend):
        return _BackendSpec("http", backend.origin, env_items)
    if isinstance(backend, LegacyTestBackend):
        return _BackendSpec("legacy", backend.origin, env_items)
    if isinstance(backend, str):
        if is_url(backend) or backend.lower() in {"http", "legacy", "legacy-test", "test-api"}:
            be = resolve_backend(backend, env=env)
            return _backend_spec(be, env=env)
        if backend.lower() in {"onnx", "local"}:
            be = resolve_backend(backend, env=env)
            return _backend_spec(be, env=env)
        # Treat as weights directory path
        return _BackendSpec("onnx", backend, env_items)
    raise TypeError(
        f"backend {type(backend).__name__!r} is not supported for parallel predict; "
        "pass 'onnx', a weights path, a URL, or an OnnxBackend/HttpBackend"
    )


def _resolve_cached(spec: _BackendSpec, *, workers: int = 1) -> PredictBackend:
    key = (spec.kind, spec.arg)
    cached = _BACKEND_CACHE.get(key)
    if cached is not None:
        return cached
    env = dict(spec.env)
    if spec.kind == "onnx":
        # Prefer fewer ORT threads when many processes share a machine.
        os.environ.setdefault(ENV_ORT_INTRA, str(_ort_intra_op_for_workers(workers, env)))
        be: PredictBackend = OnnxBackend(spec.arg)
    elif spec.kind == "http":
        be = HttpBackend(spec.arg, api_key=(env or {}).get("XENOSITE_API_KEY"))
    elif spec.kind == "legacy":
        be = LegacyTestBackend(spec.arg)
    else:
        raise ValueError(f"unknown backend kind {spec.kind!r}")
    _BACKEND_CACHE[key] = be
    return be


def _job_from_input(
    inp: str | Molecule,
    *,
    models: tuple[Spec, ...],
    backend: _BackendSpec,
    metabolites: bool,
    metabolites_min_score: float | None,
    mapped_smiles: bool,
    detailed: bool,
    rdkit: bool,
    parameter: Mapping[str, Any] | None,
) -> _PredictJob:
    _, molecule = as_molecule(inp)
    # Keep the original SMILES so workers can recover ``atoms.reordered``.
    smiles = inp if isinstance(inp, str) else molecule.smiles
    return _PredictJob(
        smiles=smiles,
        models=models,
        backend=backend,
        metabolites=metabolites,
        metabolites_min_score=metabolites_min_score,
        mapped_smiles=mapped_smiles,
        detailed=detailed,
        rdkit=rdkit,
        parameter=tuple(sorted((parameter or {}).items())),
    )


def _run_job(job: _PredictJob, *, workers: int = 1) -> Molecule:
    """Worker entrypoint: run sync :func:`predict` and return a :class:`Molecule`.

    Returning the object (not a dump) keeps ``Molecule.rdkit`` when ``rdkit=True``.
    """
    ensure_builtins()
    be = _resolve_cached(job.backend, workers=workers)
    return predict(
        job.smiles,
        models=list(job.models),
        backend=be,
        metabolites=job.metabolites,
        metabolites_min_score=job.metabolites_min_score,
        mapped_smiles=job.mapped_smiles,
        detailed=job.detailed,
        rdkit=job.rdkit,
        _parameter=dict(job.parameter) or None,
        env=dict(job.backend.env) or None,
    )


def _run_job_process(job: _PredictJob) -> Molecule:
    # workers hint is approximate inside the child; ORT intra-op already set via env.
    return _run_job(job, workers=default_workers())


def _shutdown_pools() -> None:
    global _process_pool, _process_pool_workers, _thread_pool, _thread_pool_workers
    with _process_lock:
        if _process_pool is not None:
            _process_pool.shutdown(wait=False, cancel_futures=True)
            _process_pool = None
            _process_pool_workers = None
    with _thread_lock:
        if _thread_pool is not None:
            _thread_pool.shutdown(wait=False, cancel_futures=True)
            _thread_pool = None
            _thread_pool_workers = None


atexit.register(_shutdown_pools)


def _get_process_pool(workers: int) -> ProcessPoolExecutor:
    global _process_pool, _process_pool_workers
    with _process_lock:
        if _process_pool is None or _process_pool_workers != workers:
            if _process_pool is not None:
                _process_pool.shutdown(wait=False, cancel_futures=True)
            # Spawn avoids inheriting parent ORT/OpenBabel state across forks.
            ctx = mp.get_context("spawn")
            _process_pool = ProcessPoolExecutor(max_workers=workers, mp_context=ctx)
            _process_pool_workers = workers
        return _process_pool


def _get_thread_pool(workers: int) -> ThreadPoolExecutor:
    global _thread_pool, _thread_pool_workers
    with _thread_lock:
        if _thread_pool is None or _thread_pool_workers != workers:
            if _thread_pool is not None:
                _thread_pool.shutdown(wait=False, cancel_futures=True)
            _thread_pool = ThreadPoolExecutor(max_workers=workers)
            _thread_pool_workers = workers
        return _thread_pool


def _use_processes(spec: _BackendSpec) -> bool:
    """Processes for CPU-bound ONNX; threads for HTTP I/O."""
    return spec.kind == "onnx"


def _prepare_jobs(
    inputs: InputsArg,
    model: Optional[str],
    models: Optional[ModelsArg],
    *,
    backend: BackendArg,
    backends: BackendMap,
    env: Optional[Mapping[str, str]],
    metabolites: bool,
    metabolites_min_score: float | None,
    mapped_smiles: bool,
    detailed: bool,
    rdkit: bool,
    _parameter: Optional[Mapping[str, Any]],
) -> tuple[list[_PredictJob], _BackendSpec]:
    if backends:
        raise ValueError(
            "predict_many/apredict_many do not support per-model backends=; "
            "pin a single backend= for the batch"
        )
    ensure_builtins()
    if models is None:
        models = model
    specs = tuple(normalize_models(models))
    bspec = _backend_spec(backend, env=env)
    jobs = [
        _job_from_input(
            inp,
            models=specs,
            backend=bspec,
            metabolites=metabolites,
            metabolites_min_score=metabolites_min_score,
            mapped_smiles=mapped_smiles,
            detailed=detailed,
            rdkit=rdkit,
            parameter=_parameter,
        )
        for inp in inputs
    ]
    return jobs, bspec


def _map_jobs(
    jobs: list[_PredictJob],
    *,
    bspec: _BackendSpec,
    workers: int,
    chunksize: int,
) -> list[Molecule]:
    if not jobs:
        return []
    if workers == 1 or len(jobs) == 1:
        return [_run_job(j, workers=1) for j in jobs]

    if _use_processes(bspec):
        # Advertise ORT thread budget to children via env before spawn.
        os.environ.setdefault(ENV_ORT_INTRA, str(_ort_intra_op_for_workers(workers)))
        pool = _get_process_pool(workers)
        return list(pool.map(_run_job_process, jobs, chunksize=max(1, chunksize)))
    pool_t = _get_thread_pool(workers)
    return list(pool_t.map(lambda j: _run_job(j, workers=workers), jobs))


def predict_many(
    inputs: InputsArg,
    model: Optional[str] = None,
    models: Optional[ModelsArg] = None,
    *,
    backend: BackendArg = None,
    backends: BackendMap = None,
    env: Optional[Mapping[str, str]] = None,
    metabolites: bool = False,
    metabolites_min_score: Optional[float] = None,
    mapped_smiles: bool = False,
    detailed: bool = False,
    rdkit: bool = False,
    workers: Optional[int] = None,
    chunksize: int = 1,
    _parameter: Optional[Mapping[str, Any]] = None,
) -> list[Molecule]:
    """Predict for many molecules in parallel (sync API).

    Uses a process pool for ONNX (descriptor generation is the bottleneck) and
    a thread pool for HTTP. ``workers=1`` forces sequential execution. Each
    input is parsed independently; pass SMILES strings for best pickling cost.
    """
    jobs, bspec = _prepare_jobs(
        inputs,
        model,
        models,
        backend=backend,
        backends=backends,
        env=env,
        metabolites=metabolites,
        metabolites_min_score=metabolites_min_score,
        mapped_smiles=mapped_smiles,
        detailed=detailed,
        rdkit=rdkit,
        _parameter=_parameter,
    )
    n = default_workers(env) if workers is None else max(1, int(workers))
    return _map_jobs(jobs, bspec=bspec, workers=n, chunksize=chunksize)


async def apredict(
    inp: str | Molecule,
    model: Optional[str] = None,
    models: Optional[ModelsArg] = None,
    *,
    backend: BackendArg = None,
    backends: BackendMap = None,
    env: Optional[Mapping[str, str]] = None,
    metabolites: bool = False,
    metabolites_min_score: Optional[float] = None,
    mapped_smiles: bool = False,
    detailed: bool = False,
    rdkit: bool = False,
    workers: Optional[int] = None,
    _parameter: Optional[Mapping[str, Any]] = None,
) -> Molecule:
    """Async single-molecule predict (offloads to the shared worker pool).

    Concurrent ``asyncio.gather`` of several :func:`apredict` calls shares the
    process pool and overlaps descriptor generation across molecules.
    """
    if backends:
        # Per-model overrides stay on the event-loop thread via to_thread so we
        # still reuse the sync predict path without pickling backend objects.
        return await asyncio.to_thread(
            predict,
            inp,
            model,
            models,
            backend=backend,
            backends=backends,
            env=env,
            metabolites=metabolites,
            metabolites_min_score=metabolites_min_score,
            mapped_smiles=mapped_smiles,
            detailed=detailed,
            rdkit=rdkit,
            _parameter=_parameter,
        )
    jobs, bspec = _prepare_jobs(
        [inp],
        model,
        models,
        backend=backend,
        backends=None,
        env=env,
        metabolites=metabolites,
        metabolites_min_score=metabolites_min_score,
        mapped_smiles=mapped_smiles,
        detailed=detailed,
        rdkit=rdkit,
        _parameter=_parameter,
    )
    n = default_workers(env) if workers is None else max(1, int(workers))
    loop = asyncio.get_running_loop()
    job = jobs[0]
    if n == 1:
        return await asyncio.to_thread(_run_job, job, workers=1)
    if _use_processes(bspec):
        os.environ.setdefault(ENV_ORT_INTRA, str(_ort_intra_op_for_workers(n)))
        return await loop.run_in_executor(_get_process_pool(n), _run_job_process, job)
    return await loop.run_in_executor(
        _get_thread_pool(n), lambda: _run_job(job, workers=n)
    )


async def apredict_many(
    inputs: InputsArg,
    model: Optional[str] = None,
    models: Optional[ModelsArg] = None,
    *,
    backend: BackendArg = None,
    backends: BackendMap = None,
    env: Optional[Mapping[str, str]] = None,
    metabolites: bool = False,
    metabolites_min_score: Optional[float] = None,
    mapped_smiles: bool = False,
    detailed: bool = False,
    rdkit: bool = False,
    workers: Optional[int] = None,
    chunksize: int = 1,
    _parameter: Optional[Mapping[str, Any]] = None,
) -> list[Molecule]:
    """Async many-molecule predict; same worker strategy as :func:`predict_many`."""
    return await asyncio.to_thread(
        predict_many,
        inputs,
        model,
        models,
        backend=backend,
        backends=backends,
        env=env,
        metabolites=metabolites,
        metabolites_min_score=metabolites_min_score,
        mapped_smiles=mapped_smiles,
        detailed=detailed,
        rdkit=rdkit,
        workers=workers,
        chunksize=chunksize,
        _parameter=_parameter,
    )


def reset_pools_for_tests() -> None:
    """Shut down shared pools (test helper)."""
    _shutdown_pools()
    _BACKEND_CACHE.clear()
