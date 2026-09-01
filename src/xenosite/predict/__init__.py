"""Public user API for ``xenosite.predict``.

Call :func:`predict` with a SMILES string or an existing :class:`Molecule`.
Results append; parse/canonicalize happens once when several models run.

See the package README for backends, environment variables, and versions.
"""

from .api import list_models, predict
from .registry import register_model
from .errors import (
    BackendNotConfigured,
    InvalidMolecule,
    ModelNotAvailable,
    OpenBabelNotAvailable,
    UnknownModel,
    WeightsDownloadError,
    WeightsNotFound,
)
from .weights import download_weights, ensure_weights
from .types import (
    AtomBondResult,
    AtomResult,
    Atoms,
    BondResult,
    Bonds,
    Metabolite,
    ModelResult,
    MolAtomPairResult,
    MolAtomResult,
    MolBondResult,
    Molecule,
    Result,
)

__all__ = [
    "predict",
    "list_models",
    "register_model",
    "Molecule",
    "Atoms",
    "Bonds",
    "Result",
    "MolBondResult",
    "MolAtomResult",
    "MolAtomPairResult",
    "AtomResult",
    "BondResult",
    "AtomBondResult",
    "Metabolite",
    "ModelResult",
    "InvalidMolecule",
    "UnknownModel",
    "BackendNotConfigured",
    "WeightsNotFound",
    "WeightsDownloadError",
    "ModelNotAvailable",
    "OpenBabelNotAvailable",
    "download_weights",
    "ensure_weights",
]
