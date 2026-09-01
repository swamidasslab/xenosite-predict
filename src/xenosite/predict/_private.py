"""Private integration API for in-repo callers (e.g. ``xenosite-api``).

Import as ``xenosite.predict._private``. Not part of the stable public API —
signature and exports may change without a major version bump.

Forest metabolite attachment is implemented once in :func:`~xenosite.predict.forest.attach_metabolites`.
This module re-exports that function (as ``add_metabolites``) plus discovery helpers.

Typical **xenosite-api** usage after legacy HTTP adapters populate scores::

    from xenosite.predict.types import Molecule as PredictMolecule
    from xenosite.predict._private import add_metabolites, metabolite_supported

    mol = PredictMolecule.model_validate(api_molecule.model_dump())
    if metabolites_requested:
        add_metabolites(mol, mapped_smiles=mapped_smiles)
    # copy ``result.metabolite`` (and optional map fields) back onto the API model
"""

from __future__ import annotations

from .forest import (
    attach_metabolites,
    metabolite_supported,
    supported_metabolite_models,
)

# Alias for service layers; same function object as attach_metabolites.
add_metabolites = attach_metabolites

__all__ = [
    "add_metabolites",
    "attach_metabolites",
    "metabolite_supported",
    "supported_metabolite_models",
]
