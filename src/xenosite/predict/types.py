"""User-API molecule types, ported from ``xenosite-api`` ``types.py``.

Indices are **0-based RDKit** atom and bond indices. Scores are floats.
This package does not perform name lookup; ``name`` is optional metadata.
"""

from __future__ import annotations

from typing import Any, Optional, Union

from pydantic import BaseModel as _BaseModel
from pydantic import ConfigDict, Field, NonNegativeInt, PositiveInt, PrivateAttr

Number = float


class BaseModel(_BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)


class Bonds(BaseModel):
    """Bonds in RDKit order (``GetBonds()``). ``idx`` is ``(begin, end)`` atom pairs."""

    idx: list[tuple[NonNegativeInt, NonNegativeInt]]
    order: Optional[list[Number]] = None


class Atoms(BaseModel):
    """Heavy-atom topology in RDKit canonical-SMILES atom order."""

    num: PositiveInt
    reordered: Optional[list[NonNegativeInt]] = None
    z: Optional[list[NonNegativeInt]] = None
    impHs: Optional[list[NonNegativeInt]] = None
    cipRank: Optional[list[NonNegativeInt]] = None
    chrg: Optional[list[int]] = None


class Metabolite(BaseModel):
    """A inferred or bioactivation metabolite (pathway + site atoms + score).

    ``atom`` lists **0-based RDKit** indices for the site of metabolism on the
    parent, matching ``Molecule.atoms`` / ``Molecule.bonds.idx``.

    ``map_idx`` lists **1-based** parent atom numbers for each heavy atom in
    ``smiles`` (canonical order). ``0`` marks newly introduced atoms.

    When requested, ``mapped_smiles`` is the same structure with atom-map
    numbers embedded (e.g. ``[CH2:1]``) tracing atoms back to the parent.
    """

    smiles: str
    atom: Optional[list[NonNegativeInt]] = None
    map_idx: Optional[list[NonNegativeInt]] = None
    mapped_smiles: Optional[str] = None
    pathway: Optional[str] = None
    score: Number = None  # type: ignore[assignment]


class Result(BaseModel):
    """One model (or model head) attached to a :class:`Molecule`."""

    model: str
    version: str
    depiction: Optional[str] = None
    metabolite: Optional[list[Metabolite]] = None


class MolBondResult(Result):
    mol: Number
    bond: list[Number]


class AtomResult(Result):
    atom: list[Number]


class BondResult(Result):
    bond: list[Number]


class MolAtomResult(Result):
    mol: Number
    atom: list[Number]


class MolAtomPairResult(MolAtomResult):
    pair: list[Number]
    pair_idx: list[tuple[int, int]]


class AtomBondResult(Result):
    bond: list[Number]
    atom: list[Number]


ModelResult = Union[
    MolAtomPairResult,
    MolAtomResult,
    MolBondResult,
    AtomBondResult,
    AtomResult,
    BondResult,
    Result,
]

Results = list[ModelResult]


class Molecule(BaseModel):
    """Primary return type: canonical SMILES, topology, and appended model results."""

    smiles: str = Field(description="Non-isomeric canonical SMILES.")
    results: Results = Field(default_factory=list)
    atoms: Atoms
    bonds: Bonds
    name: Optional[dict[str, Union[int, str]]] = Field(default_factory=dict)
    _parameter: dict[str, Any] = PrivateAttr(default_factory=dict)
    model_config = ConfigDict(json_schema_extra={})
