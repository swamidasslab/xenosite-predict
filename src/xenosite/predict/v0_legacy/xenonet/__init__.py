"""Internal XenoNet weighted-network engine. Not part of the public API.

Call :func:`build_network` from tests. Do not register this as a ``predict()``
model. Bioactivation (PBS/MBS) is out of scope here. Importing this package
warns until golden fixtures exist.

Quirks vs Python-2 XenoNet 1.0 are listed in :mod:`quirks`. Scoring ``"0"``
uses the unpooled Class-row walk. Scoring ``"1"`` uses pooled atom/bond
lookup on the same ONNX rows.
"""

from __future__ import annotations

from typing import Mapping, Optional, Sequence

from ._unvalidated import warn_unvalidated

warn_unvalidated()

from rdkit import Chem

from xenosite.predict.backends import PredictBackend, resolve_backend
from xenosite.predict.backends.onnx import OnnxBackend
from .graph import XenoGraph
from .score import Phase1SiteTable, phase1_site_table
from .search import evaluate_paths


def _onnx_backend(
    backend: Optional[str | PredictBackend] = None,
    env: Optional[Mapping[str, str]] = None,
) -> OnnxBackend:
    if isinstance(backend, OnnxBackend):
        return backend
    be = resolve_backend(backend, env=env)
    if not isinstance(be, OnnxBackend):
        raise TypeError("xenonet build_network requires the ONNX backend")
    return be


def build_network(
    smiles: str,
    *,
    depth_limit: int = 1,
    beam_width: int = 1000,
    targets: Sequence[str] = (),
    scoring: str = "0",
    backend: Optional[str | PredictBackend] = None,
    env: Optional[Mapping[str, str]] = None,
    trim_threshold: float = 0.0,
    likelihoods: bool = True,
) -> XenoGraph:
    """Beam-search a Phase I weighted network. Tests-only; no public contract.

    Parameters
    ----------
    smiles:
        Start molecule.
    depth_limit:
        Maximum number of reaction steps.
    beam_width:
        Max children kept per parent (large values avoid heap-tie drops).
    targets:
        If set, only paths that terminate at one of these SMILES are kept.
    scoring:
        ``"0"`` unpooled Class rows; ``"1"`` pooled atom/bond lookup.
    """
    if scoring not in ("0", "1"):
        raise ValueError(f"scoring must be '0' or '1', got {scoring!r}")
    be = _onnx_backend(backend, env=env)
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        from xenosite.predict.errors import InvalidMolecule

        raise InvalidMolecule(f"Not a valid SMILES string: {smiles}")
    target_can = list(targets)
    cache: dict[str, Phase1SiteTable] = {}
    graph = evaluate_paths(
        mol,
        backend=be,
        depth_limit=depth_limit,
        beam_width=beam_width,
        targets=target_can,
        scoring=scoring,
        cache=cache,
    )
    if graph.is_empty() and not graph.nodes:
        graph.nodes[graph.root] = graph.nodes.get(graph.root) or graph._ensure_node(graph.root)
        return graph
    if not graph.is_empty():
        graph = graph.trim_weights(threshold=trim_threshold)
        if likelihoods and not graph.is_empty():
            n_nodes = max(2, len(graph.nodes))
            graph.compute_metabolite_likelihoods(
                mode="max", max_iterations=n_nodes * 10
            )
    return graph


def site_table_for(
    smiles: str,
    *,
    backend: Optional[str | PredictBackend] = None,
    env: Optional[Mapping[str, str]] = None,
) -> Phase1SiteTable:
    """Phase1 Class rows for one molecule (unit tests / gather)."""
    be = _onnx_backend(backend, env=env)
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        from xenosite.predict.errors import InvalidMolecule

        raise InvalidMolecule(f"Not a valid SMILES string: {smiles}")
    return phase1_site_table(mol, be)
