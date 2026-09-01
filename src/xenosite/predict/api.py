"""``predict`` / ``list_models`` — backend-agnostic user API."""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Optional, Union

from .backends import PredictBackend, resolve_backend, resolve_for_model
from .errors import BackendNotConfigured
from .molecule import as_molecule
from .registry import Spec, ensure_builtins, load_runner, normalize_models, registered
from ._private import add_metabolites
from .types import Molecule

ModelsArg = Union[str, Spec, Iterable[str | Spec]]
BackendArg = Union[str, PredictBackend, None]
BackendMap = Optional[Mapping[Spec, str | PredictBackend]]


def predict(
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
    _parameter: Optional[Mapping[str, Any]] = None,
) -> Molecule:
    """Run one or more models and return a :class:`Molecule` with appended results.

    Parameters
    ----------
    inp:
        SMILES string or an existing :class:`Molecule` (results are appended).
    model:
        Single model name (default version). Ignored if ``models`` is set.
    models:
        Names and/or ``(name, version)`` pairs. Parse/canonicalize once.
    backend:
        Pin every model in this call: ``"onnx"``, ``"http"``, ``"legacy"``,
        a URL, or a :class:`PredictBackend`. ``None`` uses the env picker.
    backends:
        Per-``(name, version)`` override (wins over ``backend``).
    env:
        Environment mapping for the picker. ``None`` uses ``os.environ``.
        Tests should pass ``env={}`` or rely on the autouse clearer.
    metabolites:
        When ``True``, attach every metabolite structure the forest ruleset
        generates for the substrate, annotated with predictor site scores and
        sorted by score (descending). See ``xenosite.predict.forest``.
    metabolites_min_score:
        When set, drop metabolites whose site score is below this threshold.
        Default ``None`` includes all forest products.
    mapped_smiles:
        When ``True`` (with ``metabolites=True``), add ``mapped_smiles`` to each
        forest metabolite — canonical SMILES with ``:N`` atom-map numbers tracing
        heavy atoms back to the parent (1-based; new atoms unmapped). ``map_idx``
        is always populated when forest metabolites are attached.
    _parameter:
        Internal per-call options (not part of the public HTTP API). Runners
        read ``molecule._parameter``; e.g. ``ndealk_site_mode`` is ``legacy``
        for golden parity tests and ``principled`` (default) for production;
        ``quinone_omp_mode`` is ``legacy`` (deterministic sorted BFS) for golden
        tests and ``principled`` (any qualifying tied shortest path; binary) for
        production; ``mean`` averages all shortest-path indicators (fractional);
        ``symmetry_group_mode`` is ``openbabel`` for golden parity and ``rdkit``
        (default) for production bond-class deduplication and score pooling
        (mean of active scores per class); ``bond_nrings_mode`` is ``legacy``
        (DFS back-edge atom counts) for golden/ob dumps and ``principled``
        (RDKit ``RingInfo.NumAtomRings`` per BondTD endpoint) for production.

    Notes
    -----
    One molecule at a time (no batch API). Import does not open ONNX, HTTP, or OpenBabel.
    First use downloads ONNX weights when XENOSITE_ONNX_URL is set.
    """
    ensure_builtins()
    if models is None:
        models = model
    specs = normalize_models(models)
    _, molecule = as_molecule(inp)
    if _parameter:
        molecule._parameter = dict(_parameter)

    for spec in specs:
        be = resolve_for_model(spec, backend=backend, backends=backends, env=env)
        runner = load_runner(*spec)
        runner.predict_molecule(molecule, be)
    if metabolites:
        add_metabolites(
            molecule,
            min_score=metabolites_min_score,
            mapped_smiles=mapped_smiles,
        )
    return molecule


def list_models(
    *,
    backend: BackendArg = None,
    env: Optional[Mapping[str, str]] = None,
) -> list[dict]:
    """What this process can actually run (backend-aware), not a fictional union.

    Each item is ``{"name", "version", "available", "backend", "reason"}``.
    """
    ensure_builtins()
    try:
        be = resolve_backend(backend, env=env)
        available = set(be.available_models())
        bname = be.name
    except BackendNotConfigured as exc:
        available = set()
        bname = None
        default_reason = str(exc)
    else:
        default_reason = ""

    out = []
    for info in registered():
        spec = (info.name, info.version)
        ok = spec in available and not info.blocked_reason
        reason = info.blocked_reason or ("" if ok else (default_reason or "not on this backend"))
        if ok and bname == "onnx" and info.name not in ("bioactivation",):
            from .features import _ob

            if not _ob.installed():
                ok = False
                reason = "OpenBabel is required for ONNX descriptors (uv add openbabel)"
        out.append(
            {
                "name": info.name,
                "version": info.version,
                "available": bool(ok),
                "backend": bname,
                "reason": reason,
                "heads": list(info.heads),
                "two_stage": info.two_stage,
                "pipeline": info.pipeline,
            }
        )
    return out
