"""``predict`` / ``list_models`` — backend-agnostic user API."""

from __future__ import annotations

import warnings
from typing import Any, Iterable, Mapping, Optional, Union

from .backends import PredictBackend, resolve_backend, resolve_for_model
from .backends.http import HttpBackend
from .errors import BackendNotConfigured, ModelNotAvailable
from .molecule import apply_presentation, prepare_backend_molecule
from .registry import Spec, ensure_builtins, load_runner, normalize_models, registered
from ._private import add_metabolites
from .scoring import apply_scoring_parameters
from .types import Molecule

ModelsArg = Union[str, Spec, Iterable[str | Spec]]
BackendArg = Union[str, PredictBackend, None]
BackendMap = Optional[Mapping[Spec, str | PredictBackend]]

_RDKIT_HTTP_WARNING = (
    "rdkit=True with the HTTP backend forces a local RDKit reparse after the "
    "response, which slows requests; prefer rdkit=False unless you need the mol."
)


def _guard_remote_parameter(
    specs: list[Spec],
    *,
    backend: BackendArg,
    backends: BackendMap,
    env: Optional[Mapping[str, str]],
    parameter: Optional[Mapping[str, Any]],
) -> None:
    if not parameter:
        return
    for spec in specs:
        be = resolve_for_model(spec, backend=backend, backends=backends, env=env)
        name = getattr(be, "name", "")
        if name in {"http", "legacy"}:
            raise ModelNotAvailable(
                "_parameter overlays are only supported on the local ONNX backend; "
                f"got backend {name!r} for {spec[0]!r}"
            )


def _attach_rdkit(
    molecule: Molecule,
    *,
    rdkit: bool,
    used_http: bool,
    canonicalize: bool,
    input_smiles: str,
    rdmol,
) -> None:
    if not rdkit:
        molecule.rdkit = None
        return
    if used_http:
        warnings.warn(_RDKIT_HTTP_WARNING, UserWarning, stacklevel=3)
        from .molecule import parse_smiles

        target = molecule.smiles if canonicalize else input_smiles
        mol, _ = parse_smiles(target if canonicalize else input_smiles, rdkit=True)
        # When not canonicalize, parse_smiles still yields canonical mol — reparse input.
        if not canonicalize:
            from rdkit import Chem

            mol = Chem.MolFromSmiles(input_smiles)
        molecule.rdkit = mol
        return
    if canonicalize and rdmol is not None:
        molecule.rdkit = rdmol
        return
    if not canonicalize:
        from rdkit import Chem

        molecule.rdkit = Chem.MolFromSmiles(input_smiles)
        return
    if molecule.rdkit is None and rdmol is not None:
        molecule.rdkit = rdmol


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
    detailed: bool = False,
    canonicalize: bool = True,
    rdkit: bool = False,
    _parameter: Optional[Mapping[str, Any]] = None,
) -> Molecule:
    """Run one or more models and return a :class:`Molecule` with appended results.

    Backends always receive a **canonical** molecule with **detailed** topology.
    ``canonicalize`` and ``detailed`` only affect the returned presentation.

    Parameters
    ----------
    inp:
        SMILES string or an existing :class:`Molecule` (results are appended).
    model:
        Single model name (default version). Ignored if ``models`` is set.
    models:
        Names and/or ``(name, version)`` pairs. Parse/canonicalize once.
        Version ``"1"`` (default) uses updated scoring parameters; ``"0"``
        uses legacy parameters that match golden / ``/v0`` HTTP.
    backend:
        Pin every model in this call: ``"onnx"``, ``"http"``, ``"legacy"``,
        a URL, or a :class:`PredictBackend`. ``None`` uses the env picker.
    backends:
        Per-``(name, version)`` override (wins over ``backend``).
    env:
        Environment mapping for the picker. ``None`` uses ``os.environ``.
    metabolites:
        Attach forest-inferred metabolites after scoring.
    metabolites_min_score:
        Drop forest metabolites below this site score.
    mapped_smiles:
        When ``True`` (with ``metabolites=True``), add ``mapped_smiles`` maps.
    detailed:
        When ``True``, keep atom/bond detail fields on the returned molecule.
        Backends always see detailed topology regardless.
    canonicalize:
        When ``True`` (default), return canonical SMILES atom order. When
        ``False``, remap scores/topology to the **input** atom order.
    rdkit:
        Keep an RDKit mol on the result matching the returned atom order.
        With HTTP this triggers a local reparse and a :class:`UserWarning`.
    _parameter:
        Scoring-parameter overlay (ONNX only; errors on HTTP/legacy).
    """
    ensure_builtins()
    if models is None:
        models = model
    specs = normalize_models(models)
    _guard_remote_parameter(
        specs, backend=backend, backends=backends, env=env, parameter=_parameter
    )

    # Keep RDKit from the backend parse when possible (ONNX path).
    want_rdkit_during = rdkit and canonicalize
    rdmol, molecule, input_smiles = prepare_backend_molecule(
        inp, rdkit=want_rdkit_during
    )
    user_parameter = dict(_parameter) if _parameter is not None else None
    used_http = False

    for spec in specs:
        be = resolve_for_model(spec, backend=backend, backends=backends, env=env)
        if isinstance(be, HttpBackend) or getattr(be, "name", "") == "http":
            used_http = True
        runner = load_runner(*spec)
        molecule._parameter = apply_scoring_parameters(spec[1], user_parameter)
        runner.predict_molecule(molecule, be)

    if metabolites:
        add_metabolites(
            molecule,
            min_score=metabolites_min_score,
            mapped_smiles=mapped_smiles,
            rdmol=rdmol,
            rdkit=rdkit and not used_http,
        )

    apply_presentation(
        molecule,
        canonicalize=canonicalize,
        detailed=detailed,
        input_smiles=input_smiles,
    )
    _attach_rdkit(
        molecule,
        rdkit=rdkit,
        used_http=used_http,
        canonicalize=canonicalize,
        input_smiles=input_smiles,
        rdmol=rdmol,
    )
    return molecule


def list_models(
    *,
    backend: BackendArg = None,
    env: Optional[Mapping[str, str]] = None,
) -> list[dict]:
    """What this process can actually run (backend-aware), not a fictional union.

    Each item is ``{"name", "version", "available", "backend", "reason"}``.
    Versions ``"0"`` and ``"1"`` are listed separately (legacy vs updated scoring).
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
            from xenosite.predict.v1.features._ob import installed as openbabel_installed

            if not openbabel_installed():
                ok = False
                reason = "OpenBabel is required for ONNX descriptors (uv add openbabel)"
        if ok and bname == "onnx" and info.name == "bioactivation":
            ok = False
            reason = "bioactivation is a pipeline; ONNX heads are not a full predict path"
        if ok and bname == "http" and info.name == "bioactivation":
            ok = False
            reason = "bioactivation is not available on the HTTP backend"
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
