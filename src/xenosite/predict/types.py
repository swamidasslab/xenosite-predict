"""User-API molecule types, ported from ``xenosite-api`` ``types.py``.

Indices are **0-based RDKit** atom and bond indices in canonical SMILES
order. Scores are floats aligned with ``atoms`` / ``bonds.idx``.
This package does not perform name lookup; ``name`` is optional metadata.
"""

from __future__ import annotations

from typing import Any, Literal, Optional, Union

from pydantic import BaseModel as _BaseModel
from pydantic import ConfigDict, Field, NonNegativeInt, PositiveInt, PrivateAttr

Number = float
ModelVersion = Literal["0", "1"]


class BaseModel(_BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, protected_namespaces=())


class Bonds(BaseModel):
    """Bonds in canonical-SMILES atom order, listed to match score arrays."""

    idx: list[tuple[NonNegativeInt, NonNegativeInt]] = Field(
        description=(
            "Bond endpoints as pairs of 0-based atom indices. Score arrays such "
            "as `bond` line up with this list."
        )
    )
    order: Optional[list[Number]] = Field(
        default=None,
        description="Bond orders (single, double, …) for each bond in `idx` when `detailed=True`.",
    )


class Atoms(BaseModel):
    """Heavy-atom topology in RDKit canonical-SMILES atom order."""

    num: PositiveInt = Field(description="Number of heavy atoms.")
    reordered: Optional[list[NonNegativeInt]] = Field(
        default=None,
        description=(
            "When `detailed=True`, original (input) atom indices in canonical "
            "SMILES order. Omitted otherwise."
        ),
    )
    z: Optional[list[NonNegativeInt]] = Field(
        default=None,
        description="Atomic numbers for each heavy atom when `detailed=True`.",
    )
    impHs: Optional[list[NonNegativeInt]] = Field(
        default=None,
        description="Implicit hydrogen counts for each heavy atom when `detailed=True`.",
    )
    cipRank: Optional[list[NonNegativeInt]] = Field(
        default=None,
        description="CIP stereochemistry ranks for each heavy atom when `detailed=True`.",
    )
    chrg: Optional[list[int]] = Field(
        default=None,
        description="Formal charges for each heavy atom when `detailed=True`.",
    )


class Metabolite(BaseModel):
    """A inferred or bioactivation metabolite (pathway + site atoms + score).

    ``atom`` lists **0-based RDKit** indices for the site of metabolism on the
    parent, matching ``Molecule.atoms`` / ``Molecule.bonds.idx``.

    ``map_idx`` lists **1-based** parent atom numbers for each heavy atom in
    ``smiles`` (canonical order). ``0`` marks newly introduced atoms.

    When requested, ``mapped_smiles`` is the same structure with atom-map
    numbers embedded (e.g. ``[CH2:1]``) tracing atoms back to the parent.

    Conjugation adducts use a dummy ``*`` with a CXSMILES ``atomLabel``
    (``GlcA``, ``GSH``, ``Protein``, ``DNA``, ``CN``) so depictions can name
    the conjugate.
    """

    smiles: str
    atom: Optional[list[NonNegativeInt]] = None
    map_idx: Optional[list[NonNegativeInt]] = None
    mapped_smiles: Optional[str] = None
    pathway: Optional[str] = None
    score: Number = None  # type: ignore[assignment]
    rdkit: Optional[Any] = Field(
        default=None,
        exclude=True,
        repr=False,
        description="In-process RDKit mol when ``rdkit=True``. Omitted from JSON.",
    )


class Result(BaseModel):
    """One model (or model head) attached to a :class:`Molecule`.

    ``model_version`` is the scoring generation: ``"0"`` (legacy parameters) or
    ``"1"`` (updated parameters, the ``predict()`` default). The registry still
    keys runners by ``(name, version)``; that value is copied here.
    """

    model: str
    model_version: ModelVersion
    depiction: Optional[str] = None
    metabolite: Optional[list[Metabolite]] = None


class MolBondResult(Result):
    mol: Number
    bond: list[Number] = Field(description="Per-bond scores, aligned with `bonds.idx`.")


class AtomResult(Result):
    atom: list[Number] = Field(
        description="Per-atom scores in canonical SMILES atom order (same as `Molecule.atoms`)."
    )


class BondResult(Result):
    bond: list[Number] = Field(description="Per-bond scores, aligned with `bonds.idx`.")


class MolAtomResult(Result):
    mol: Number
    atom: list[Number] = Field(
        description="Per-atom scores in canonical SMILES atom order (same as `Molecule.atoms`)."
    )


class MolAtomPairResult(MolAtomResult):
    pair: list[Number]
    pair_idx: list[tuple[int, int]]


class AtomBondResult(Result):
    bond: list[Number] = Field(description="Per-bond scores, aligned with `bonds.idx`.")
    atom: list[Number] = Field(
        description="Per-atom scores in canonical SMILES atom order (same as `Molecule.atoms`)."
    )


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
    """Primary return type: SMILES, topology, and appended model results.

    Backends always see canonical non-isomeric SMILES atom order. After
    presentation (``canonicalize=False``), indices may match the input order.
    """

    smiles: str = Field(description="Non-isomeric SMILES (canonical unless remapped).")
    results: Results = Field(default_factory=list)
    atoms: Atoms
    bonds: Bonds
    name: Optional[dict[str, Union[int, str]]] = Field(default_factory=dict)
    rdkit: Optional[Any] = Field(
        default=None,
        exclude=True,
        repr=False,
        description="In-process RDKit mol when ``rdkit=True``. Omitted from JSON.",
    )
    _parameter: dict[str, Any] = PrivateAttr(default_factory=dict)
    _input_smiles: Optional[str] = PrivateAttr(default=None)
    model_config = ConfigDict(json_schema_extra={}, protected_namespaces=())
