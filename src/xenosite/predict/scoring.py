"""Scoring-version defaults for ``predict`` (HTTP ``/v0`` vs ``/v1``).

Version ``"0"`` matches legacy-test-api / golden fixtures (historical site keys,
one BFS quinone path, OpenBabel symmetry, DFS ring counts). Version ``"1"`` is
the updated mapping (RDKit symmetry pooling, principled site/OMP/NRings). Same
ONNX weights; only how row scores map onto atoms and bonds differs.

Callers pick a version with ``models=[("epoxidation", "0")]`` or omit it to get
``"1"``. ``_parameter`` overlays these defaults for tests and ablations.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

from .errors import UnknownModel

SCORING_VERSIONS = ("0", "1")
DEFAULT_SCORING_VERSION = "1"

# Golden / OpenBabel parity — HTTP ``/v0`` and ``models=[(name, "0")]``.
V0_PARAMETER: dict[str, str] = {
    "ndealk_site_mode": "legacy",
    "quinone_omp_mode": "legacy",
    "symmetry_group_mode": "openbabel",
    "bond_nrings_mode": "legacy",
}

# Updated production mapping — HTTP ``/v1`` and the default ``models=["name"]``.
V1_PARAMETER: dict[str, str] = {
    "ndealk_site_mode": "principled",
    "quinone_omp_mode": "principled",
    "symmetry_group_mode": "rdkit",
    "bond_nrings_mode": "principled",
}


def parameters_for_version(version: str) -> dict[str, str]:
    """Return the default ``_parameter`` bundle for a scoring version."""
    if version == "0":
        return dict(V0_PARAMETER)
    if version == "1":
        return dict(V1_PARAMETER)
    raise UnknownModel(f"Unknown scoring version {version!r}; use '0' or '1'")


def apply_scoring_parameters(
    version: str,
    user: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Build ``molecule._parameter`` for one model run.

    Version ``"0"`` injects the legacy bundle when ``user`` is omitted.
    Version ``"1"`` leaves the mapping empty (runners already default to the
    updated flags) unless ``user`` overlays keys.
    """
    defaults = parameters_for_version(version)
    if user is not None:
        return {**defaults, **dict(user)}
    if version == "0":
        return defaults
    return {}
