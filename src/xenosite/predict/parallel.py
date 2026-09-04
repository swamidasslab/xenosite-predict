"""Parallel / async prediction helpers for high-throughput workloads.

ONNX uses a process pool (descriptor generation is CPU-bound). HTTP uses
``asyncio.gather`` over :meth:`HttpBackend.apredict_native` under the per-origin
concurrency semaphore — not a thread pool of sync clients.
"""

from __future__ import annotations

import asyncio
import atexit
import multiprocessing as mp
import os
import threading
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Optional, Sequence, Union

from .api import predict
from .backends import ENV_API_KEY, PredictBackend, is_url, resolve_backend
from .backends.http import HttpBackend
from .backends.legacy import LegacyTestBackend
from .backends.onnx import ENV_ORT_INTRA, OnnxBackend
from .molecule import as_molecule
from .registry import Spec, ensure_builtins, load_runner, normalize_models
from .types import Molecule

ModelsArg = Union[str, Spec, Iterable[str | Spec]]
BackendArg = Union[str, PredictBackend, None]
BackendMap = Optional[Mapping[Spec, str | PredictBackend]]
InputsArg = Sequence[str | Molecule]

ENV_WORKERS = "XENOSITE_WORKERS"

_process_pool: ProcessPoolExecutor | None = None
_process_pool_workers: int | None = None
_process_lock = threading.Lock()

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
    canonicalize: bool = True
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
    env_items = tuple(sorted((env or {}).items()))
    if backend is None:
        be = resolve_backend(None, env=env)
        return _backend_spec(be, env=env)
    if isinstance(backend, OnnxBackend):
        return _BackendSpec("onnx", str(backend.root), env_items)
    if isinstance(backend, HttpBackend):
        # Preserve api_key in env for worker rebuild.
        items = dict(env_items)
        if backend.api_key and ENV_API_KEY not in items:
            items[ENV_API_KEY] = backend.api_key
        return _BackendSpec("http", backend.origin, tuple(sorted(items.items())))
    if isinstance(backend, LegacyTestBackend):
        return _BackendSpec("legacy", backend.origin, env_items)
    if isinstance(backend, str):
        if is_url(backend) or backend.lower() in {"http", "legacy", "legacy-test", "test-api"}:
            be = resolve_backend(backend, env=env)
            return _backend_spec(be, env=env)
        if backend.lower() in {"onnx", "local"}:
            be = resolve_backend(backend, env=env)
            return _backend_spec(be, env=env)
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
        os.environ.setdefault(ENV_ORT_INTRA, str(_ort_intra_op_for_workers(workers, env)))
        be: PredictBackend = OnnxBackend(spec.arg)
    elif spec.kind == "http":
        be = HttpBackend(spec.arg, api_key=env.get(ENV_API_KEY), env=env)
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
    canonicalize: bool,
    rdkit: bool,
    parameter: Mapping[str, Any] | None,
) -> _PredictJob:
    _, molecule = as_molecule(inp)
    smiles = inp if isinstance(inp, str) else molecule.smiles
    return _PredictJob(
        smiles=smiles,
        models=models,
        backend=backend,
        metabolites=metabolites,
        metabolites_min_score=metabolites_min_score,
        mapped_smiles=mapped_smiles,
        detailed=detailed,
        canonicalize=canonicalize,
        rdkit=rdkit,
        parameter=tuple(sorted((parameter or {}).items())),
    )


def _run_job(job: _PredictJob, *, workers: int = 1) -> Molecule:
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
        canonicalize=job.canonicalize,
        rdkit=job.rdkit,
        _parameter=dict(job.parameter) or None,
        env=dict(job.backend.env) or None,
    )


def _run_job_process(job: _PredictJob) -> Molecule:
    return _run_job(job, workers=default_workers())


def _shutdown_pools() -> None:
    global _process_pool, _process_pool_workers
    with _process_lock:
        if _process_pool is not None:
            _process_pool.shutdown(wait=False, cancel_futures=True)
            _process_pool = None
            _process_pool_workers = None


atexit.register(_shutdown_pools)


def _get_process_pool(workers: int) -> ProcessPoolExecutor:
    global _process_pool, _process_pool_workers
    with _process_lock:
        if _process_pool is None or _process_pool_workers != workers:
            if _process_pool is not None:
                _process_pool.shutdown(wait=False, cancel_futures=True)
            ctx = mp.get_context("spawn")
            _process_pool = ProcessPoolExecutor(max_workers=workers, mp_context=ctx)
            _process_pool_workers = workers
        return _process_pool


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
    canonicalize: bool,
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
            canonicalize=canonicalize,
            rdkit=rdkit,
            parameter=_parameter,
        )
        for inp in inputs
    ]
    return jobs, bspec


def _map_jobs_onnx(
    jobs: list[_PredictJob],
    *,
    workers: int,
    chunksize: int,
) -> list[Molecule]:
    if not jobs:
        return []
    if workers == 1 or len(jobs) == 1:
        return [_run_job(j, workers=1) for j in jobs]
    os.environ.setdefault(ENV_ORT_INTRA, str(_ort_intra_op_for_workers(workers)))
    pool = _get_process_pool(workers)
    return list(pool.map(_run_job_process, jobs, chunksize=max(1, chunksize)))


async def _run_job_http_async(job: _PredictJob) -> Molecule:
    """One HTTP molecule on the current event loop (awaits ``apredict_native``)."""
    from ._private import add_metabolites
    from .api import _attach_rdkit
    from .molecule import apply_presentation, prepare_backend_molecule
    from .scoring import apply_scoring_parameters

    ensure_builtins()
    be = _resolve_cached(job.backend)
    if not isinstance(be, HttpBackend):
        return _run_job(job, workers=1)

    want_rdkit_during = job.rdkit and job.canonicalize
    rdmol, molecule, input_smiles = prepare_backend_molecule(
        job.smiles, rdkit=want_rdkit_during
    )
    user_parameter = dict(job.parameter) or None
    for name, version in job.models:
        runner = load_runner(name, version)
        molecule._parameter = apply_scoring_parameters(version, user_parameter)
        payload = await be.apredict_native(molecule.smiles, name, version)
        runner._merge_molecule_payload(molecule, payload)

    if job.metabolites:
        add_metabolites(
            molecule,
            min_score=job.metabolites_min_score,
            mapped_smiles=job.mapped_smiles,
            rdmol=rdmol,
            rdkit=False,
        )
    apply_presentation(
        molecule,
        canonicalize=job.canonicalize,
        detailed=job.detailed,
        input_smiles=input_smiles,
    )
    _attach_rdkit(
        molecule,
        rdkit=job.rdkit,
        used_http=True,
        canonicalize=job.canonicalize,
        input_smiles=input_smiles,
        rdmol=rdmol,
    )
    return molecule


async def _amap_jobs_http(jobs: list[_PredictJob]) -> list[Molecule]:
    """Run HTTP jobs concurrently on one event loop under the per-origin semaphore."""
    if not jobs:
        return []
    return list(await asyncio.gather(*[_run_job_http_async(j) for j in jobs]))


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
    canonicalize: bool = True,
    rdkit: bool = False,
    workers: Optional[int] = None,
    chunksize: int = 1,
    _parameter: Optional[Mapping[str, Any]] = None,
) -> list[Molecule]:
    """Predict for many molecules in parallel (sync API).

    ONNX uses a process pool; HTTP gathers async requests under the per-origin
    concurrency cap. ``workers=1`` forces sequential ONNX execution.
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
        canonicalize=canonicalize,
        rdkit=rdkit,
        _parameter=_parameter,
    )
    if bspec.kind == "http":
        from .backends.http import run_sync

        # Keep the caller-supplied HttpBackend (transport, keys, limits).
        if isinstance(backend, HttpBackend):
            _BACKEND_CACHE[("http", backend.origin)] = backend
        elif backend is None or isinstance(backend, str):
            _BACKEND_CACHE[("http", bspec.arg)] = _resolve_cached(bspec)
        return run_sync(_amap_jobs_http(jobs))
    n = default_workers(env) if workers is None else max(1, int(workers))
    return _map_jobs_onnx(jobs, workers=n, chunksize=chunksize)


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
    canonicalize: bool = True,
    rdkit: bool = False,
    workers: Optional[int] = None,
    _parameter: Optional[Mapping[str, Any]] = None,
) -> Molecule:
    """Async single-molecule predict."""
    if backends:
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
            canonicalize=canonicalize,
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
        canonicalize=canonicalize,
        rdkit=rdkit,
        _parameter=_parameter,
    )
    job = jobs[0]
    if bspec.kind == "http":
        if isinstance(backend, HttpBackend):
            _BACKEND_CACHE[("http", backend.origin)] = backend
        return await _run_job_http_async(job)
    n = default_workers(env) if workers is None else max(1, int(workers))
    loop = asyncio.get_running_loop()
    if n == 1:
        return await asyncio.to_thread(_run_job, job, workers=1)
    os.environ.setdefault(ENV_ORT_INTRA, str(_ort_intra_op_for_workers(n)))
    return await loop.run_in_executor(_get_process_pool(n), _run_job_process, job)


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
    canonicalize: bool = True,
    rdkit: bool = False,
    workers: Optional[int] = None,
    chunksize: int = 1,
    _parameter: Optional[Mapping[str, Any]] = None,
) -> list[Molecule]:
    """Async many-molecule predict."""
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
        canonicalize=canonicalize,
        rdkit=rdkit,
        _parameter=_parameter,
    )
    if bspec.kind == "http":
        return await _amap_jobs_http(jobs)
    return await asyncio.to_thread(
        _map_jobs_onnx,
        jobs,
        workers=default_workers(env) if workers is None else max(1, int(workers)),
        chunksize=chunksize,
    )


def reset_pools_for_tests() -> None:
    """Shut down shared pools (test helper)."""
    _shutdown_pools()
    _BACKEND_CACHE.clear()
    from .backends.http import reset_http_state_for_tests

    reset_http_state_for_tests()
