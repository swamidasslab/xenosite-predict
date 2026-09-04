"""SMILES parse / canonicalize and topology (shared by every model).

Canonical SMILES is **non-isomeric** (``isomericSmiles=False``), matching the
current XenoSite API gather path. Atom and bond indices are 0-based RDKit
indices in **canonical SMILES atom order** (the input is re-parsed from that
string). Name lookup is intentionally omitted.

When ``detailed=True``, topology includes atomic numbers, charges, implicit
hydrogens, CIP ranks, bond orders, and ``atoms.reordered``: the original
(input) atom indices in canonical SMILES order.
"""

from __future__ import annotations

import ast
from typing import Optional, Sequence, Union

from rdkit import Chem
from rdkit.Chem import rdchem

from .errors import InvalidMolecule
from .types import Atoms, Bonds, Molecule

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


def _smiles_atom_output_order(mol: RdkMol) -> list[int]:
    """Input atom indices in the order they appear in the last ``MolToSmiles``."""
    order = [int(i) for i in ast.literal_eval(mol.GetProp("_smilesAtomOutputOrder"))]
    n = mol.GetNumAtoms()
    return [i for i in order if 0 <= i < n][:n]


def parse_smiles(smiles: str, *, detailed: bool = False) -> tuple[RdkMol, Molecule]:
    """Parse SMILES into an RDKit mol (canonical atom order) and a :class:`Molecule`.

    Re-parses the canonical SMILES so atom indices — and therefore score arrays
    — match that string. Bond ``idx`` follows ``mol.GetBonds()``.
    """
    src = _parse_rdkit(smiles)
    canonical = Chem.MolToSmiles(src, isomericSmiles=False)
    reordered = _smiles_atom_output_order(src) if detailed else None
    mol = Chem.MolFromSmiles(canonical)
    if mol is None:
        raise InvalidMolecule(f"Canonical SMILES could not be re-parsed: {canonical}")
    return mol, molecule_from_rdkit(
        mol, smiles=canonical, detailed=detailed, reordered=reordered
    )


def molecule_from_rdkit(
    mol: RdkMol,
    *,
    smiles: Optional[str] = None,
    detailed: bool = False,
    name: Optional[dict[str, Union[int, str]]] = None,
    reordered: Optional[Sequence[int]] = None,
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
    )


def _ensure_details(molecule: Molecule) -> None:
    """Fill detailed topology on an already-canonical :class:`Molecule`."""
    if molecule.atoms.z is not None:
        return
    _, filled = parse_smiles(molecule.smiles, detailed=True)
    molecule.atoms = filled.atoms
    molecule.bonds.order = filled.bonds.order


def as_molecule(
    inp: Union[str, Molecule], *, detailed: bool = False
) -> tuple[Optional[RdkMol], Molecule]:
    """Accept SMILES or an existing :class:`Molecule`. Parse only when needed."""
    if isinstance(inp, Molecule):
        if detailed:
            _ensure_details(inp)
        return None, inp
    return parse_smiles(inp, detailed=detailed)
