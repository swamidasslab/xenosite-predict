"""SMILES parse / canonicalize, topology, and presentation helpers.

Canonical SMILES is **non-isomeric** (``isomericSmiles=False``), matching the
XenoSite API. Backends always see canonical atom order with detailed topology.
User-facing ``canonicalize`` / ``detailed`` are applied after prediction.
"""

from __future__ import annotations

import ast
from typing import Any, Optional, Sequence, Union

from rdkit import Chem
from rdkit.Chem import rdchem

from .errors import InvalidMolecule
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

RdkMol = rdchem.Mol

MIN_ATOMS = 2


def _parse_rdkit(smiles: str) -> RdkMol:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise InvalidMolecule(f"Not a valid SMILES string: {smiles}")
    if mol.GetNumAtoms() < MIN_ATOMS:
        raise InvalidMolecule(f"Molecule must have at least {MIN_ATOMS} atoms: {smiles}")
    return mol


def canonicalize_smiles(smiles: str) -> str:
    """Return non-isomeric canonical SMILES, or raise :class:`InvalidMolecule`."""
    return Chem.MolToSmiles(_parse_rdkit(smiles), isomericSmiles=False)


def is_canonical_smiles(smiles: str) -> bool:
    """True when ``smiles`` already equals its non-isomeric canonical form."""
    return smiles == canonicalize_smiles(smiles)


def _smiles_atom_output_order(mol: RdkMol) -> list[int]:
    """Input atom indices in the order they appear in the last ``MolToSmiles``."""
    order = [int(i) for i in ast.literal_eval(mol.GetProp("_smilesAtomOutputOrder"))]
    n = mol.GetNumAtoms()
    return [i for i in order if 0 <= i < n][:n]


def parse_smiles(
    smiles: str, *, detailed: bool = False, rdkit: bool = False
) -> tuple[RdkMol, Molecule]:
    """Parse SMILES into an RDKit mol (canonical atom order) and a :class:`Molecule`.

    Re-parses the canonical SMILES so atom indices — and therefore score arrays
    — match that string. Bond ``idx`` follows ``mol.GetBonds()``. When
    ``rdkit=True``, the reparsed mol is stored on ``Molecule.rdkit``.
    """
    src = _parse_rdkit(smiles)
    canonical = Chem.MolToSmiles(src, isomericSmiles=False)
    reordered = _smiles_atom_output_order(src) if detailed else None
    mol = Chem.MolFromSmiles(canonical)
    if mol is None:
        raise InvalidMolecule(f"Canonical SMILES could not be re-parsed: {canonical}")
    molecule = molecule_from_rdkit(
        mol, smiles=canonical, detailed=detailed, reordered=reordered, rdkit=rdkit
    )
    molecule._input_smiles = smiles
    return mol, molecule


def molecule_from_rdkit(
    mol: RdkMol,
    *,
    smiles: Optional[str] = None,
    detailed: bool = False,
    name: Optional[dict[str, Union[int, str]]] = None,
    reordered: Optional[Sequence[int]] = None,
    rdkit: bool = False,
) -> Molecule:
    """Build topology-only :class:`Molecule` from an RDKit mol.

    ``mol`` must already be in canonical SMILES atom order (as after
    :func:`parse_smiles`). ``reordered`` is the input-atom mapping captured
    before that reparse.
    """
    if smiles is None:
        smiles = Chem.MolToSmiles(mol, isomericSmiles=False)
        if reordered is None and detailed:
            reordered = _smiles_atom_output_order(mol)
    n = mol.GetNumAtoms()
    if n < MIN_ATOMS:
        raise InvalidMolecule(f"Molecule must have at least {MIN_ATOMS} atoms: {smiles}")

    atoms: dict = {"num": n}
    bonds_list = list(mol.GetBonds())
    bonds: dict = {
        "idx": [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in bonds_list],
    }
    if detailed:
        bonds["order"] = [b.GetBondTypeAsDouble() for b in bonds_list]
        _atoms = list(mol.GetAtoms())
        atoms["z"] = [a.GetAtomicNum() for a in _atoms]
        atoms["chrg"] = [a.GetFormalCharge() for a in _atoms]
        atoms["impHs"] = [a.GetNumImplicitHs() for a in _atoms]
        atoms["cipRank"] = list(
            Chem.CanonicalRankAtoms(mol, breakTies=False, includeIsotopes=False)
        )
        if reordered is not None:
            atoms["reordered"] = [int(i) for i in reordered]

    return Molecule(
        smiles=smiles,
        atoms=Atoms(**atoms),
        bonds=Bonds(**bonds),
        name=name or {},
        results=[],
        rdkit=mol if rdkit else None,
    )


def _ensure_details(molecule: Molecule, *, rdkit: bool = False) -> None:
    """Fill detailed topology on an already-canonical :class:`Molecule`."""
    if molecule.atoms.z is not None and molecule.atoms.reordered is not None:
        if rdkit and molecule.rdkit is None:
            mol, _ = parse_smiles(molecule.smiles, rdkit=True)
            molecule.rdkit = mol
        return
    mol, filled = parse_smiles(molecule.smiles, detailed=True, rdkit=rdkit)
    molecule.atoms = filled.atoms
    molecule.bonds.order = filled.bonds.order
    if rdkit and molecule.rdkit is None:
        molecule.rdkit = mol


def as_molecule(
    inp: Union[str, Molecule], *, detailed: bool = False, rdkit: bool = False
) -> tuple[Optional[RdkMol], Molecule]:
    """Accept SMILES or an existing :class:`Molecule`. Parse only when needed."""
    if isinstance(inp, Molecule):
        if detailed:
            _ensure_details(inp, rdkit=rdkit)
        elif rdkit and inp.rdkit is None:
            mol, _ = parse_smiles(inp.smiles, rdkit=True)
            inp.rdkit = mol
        return inp.rdkit, inp
    return parse_smiles(inp, detailed=detailed, rdkit=rdkit)


def prepare_backend_molecule(
    inp: Union[str, Molecule], *, rdkit: bool = False
) -> tuple[Optional[RdkMol], Molecule, str]:
    """Build the canonical + detailed molecule every backend call must see.

    Returns ``(rdmol, molecule, input_smiles)``. ``input_smiles`` is the original
    user string (for ``canonicalize=False`` presentation).

    Raises :class:`InvalidMolecule` when appending to a non-canonical molecule
    that already has results.
    """
    if isinstance(inp, str):
        rdmol, molecule = parse_smiles(inp, detailed=True, rdkit=rdkit)
        return rdmol, molecule, inp

    input_smiles = inp._input_smiles or inp.smiles
    if not is_canonical_smiles(inp.smiles):
        if inp.results:
            raise InvalidMolecule(
                "Cannot append predictions to a non-canonical Molecule that "
                "already has results; pass a SMILES string or a canonical Molecule."
            )
        rdmol, molecule = parse_smiles(inp.smiles, detailed=True, rdkit=rdkit)
        molecule._input_smiles = input_smiles
        return rdmol, molecule, input_smiles

    _ensure_details(inp, rdkit=rdkit)
    if inp._input_smiles is None:
        inp._input_smiles = input_smiles
    return inp.rdkit, inp, input_smiles


def _permute_atom_vector(values: Sequence[Any], reordered: Sequence[int]) -> list[Any]:
    """Map canonical-order vector → input-order using ``reordered[i]=input_idx``."""
    out: list[Any] = [None] * len(reordered)
    for can_i, inp_i in enumerate(reordered):
        out[int(inp_i)] = values[can_i]
    return out


def _map_atom_index(i: int, reordered: Sequence[int]) -> int:
    return int(reordered[i])


def _reorder_metabolite(met: Metabolite, reordered: Sequence[int]) -> Metabolite:
    data = met.model_dump()
    if met.atom is not None:
        data["atom"] = [_map_atom_index(i, reordered) for i in met.atom]
    if met.map_idx is not None:
        # 1-based parent indices in canonical parent order; 0 = new atom.
        mapped: list[int] = []
        for v in met.map_idx:
            if not v:
                mapped.append(0)
            else:
                mapped.append(_map_atom_index(int(v) - 1, reordered) + 1)
        data["map_idx"] = mapped
    data["rdkit"] = met.rdkit
    return Metabolite.model_validate(data)


def _reorder_result(result: ModelResult, reordered: Sequence[int]) -> ModelResult:
    data = result.model_dump()
    if result.metabolite is not None:
        data["metabolite"] = [
            _reorder_metabolite(m, reordered).model_dump() for m in result.metabolite
        ]
    if isinstance(result, (AtomResult, MolAtomResult, MolAtomPairResult, AtomBondResult)):
        if getattr(result, "atom", None) is not None:
            data["atom"] = _permute_atom_vector(result.atom, reordered)
    if isinstance(result, MolAtomPairResult):
        data["pair_idx"] = [
            (_map_atom_index(a, reordered), _map_atom_index(b, reordered))
            for a, b in result.pair_idx
        ]
        # pair scores stay aligned with pair_idx entries (same chemical pairs).
    if isinstance(result, (MolBondResult, BondResult, AtomBondResult)):
        # Bond score arrays stay aligned with remapped bonds.idx (same order).
        pass
    # Re-validate as the same result variant.
    for cls in (
        MolAtomPairResult,
        MolAtomResult,
        MolBondResult,
        AtomBondResult,
        AtomResult,
        BondResult,
        Result,
    ):
        if isinstance(result, cls):
            return cls.model_validate(data)
    return Result.model_validate(data)


def in_input_order(molecule: Molecule, *, input_smiles: Optional[str] = None) -> Molecule:
    """Permute a canonical-order :class:`Molecule` into input atom order (in place).

    Requires ``atoms.reordered`` where ``reordered[i]`` is the input atom index at
    canonical position ``i``. Sets ``smiles`` to the original input SMILES.
    """
    reordered = molecule.atoms.reordered
    if not reordered:
        raise InvalidMolecule(
            "Cannot remap to input order without atoms.reordered "
            "(backend molecules must be detailed=True)."
        )
    if len(reordered) != molecule.atoms.num:
        raise InvalidMolecule(
            f"atoms.reordered length {len(reordered)} != atoms.num {molecule.atoms.num}"
        )

    smi = input_smiles or molecule._input_smiles
    if smi is None:
        raise InvalidMolecule("Cannot remap to input order without the original input SMILES.")

    # Atom property vectors (canonical → input).
    for field in ("z", "chrg", "impHs", "cipRank"):
        vals = getattr(molecule.atoms, field)
        if vals is not None:
            setattr(molecule.atoms, field, _permute_atom_vector(vals, reordered))

    # Bonds: endpoints mapped; keep score alignment by rewriting idx in place.
    new_idx = [
        (_map_atom_index(a, reordered), _map_atom_index(b, reordered))
        for a, b in molecule.bonds.idx
    ]
    molecule.bonds.idx = new_idx
    # order stays parallel to idx

    molecule.results = [_reorder_result(r, reordered) for r in molecule.results]

    # After remap, identity mapping in input order.
    molecule.atoms.reordered = list(range(molecule.atoms.num))
    molecule.smiles = smi
    molecule.rdkit = None
    return molecule


def strip_details(molecule: Molecule) -> Molecule:
    """Clear detailed topology fields (in place)."""
    molecule.atoms.reordered = None
    molecule.atoms.z = None
    molecule.atoms.chrg = None
    molecule.atoms.impHs = None
    molecule.atoms.cipRank = None
    molecule.bonds.order = None
    return molecule


def apply_presentation(
    molecule: Molecule,
    *,
    canonicalize: bool = True,
    detailed: bool = False,
    input_smiles: Optional[str] = None,
) -> Molecule:
    """Apply user-facing ``canonicalize`` / ``detailed`` after backend prediction."""
    if not canonicalize:
        in_input_order(molecule, input_smiles=input_smiles)
    if not detailed:
        strip_details(molecule)
    elif canonicalize and molecule.atoms.reordered is None:
        # detailed requested but mapping missing — should not happen for backend path
        pass
    return molecule
