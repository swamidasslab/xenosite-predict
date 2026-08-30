"""SMILES parse / canonicalize and topology (shared by every model).

Canonical SMILES is **non-isomeric** (``isomericSmiles=False``), matching the
current XenoSite API gather path. Atom and bond indices are 0-based RDKit.
Name lookup is intentionally omitted.
"""

from __future__ import annotations

from typing import Optional, Union

from rdkit import Chem
from rdkit.Chem import rdchem

from .errors import InvalidMolecule
from .types import Atoms, Bonds, Molecule

RdkMol = rdchem.Mol

MIN_ATOMS = 2


def canonicalize_smiles(smiles: str) -> str:
    """Return non-isomeric canonical SMILES, or raise :class:`InvalidMolecule`."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise InvalidMolecule(f"Not a valid SMILES string: {smiles}")
    if mol.GetNumAtoms() < MIN_ATOMS:
        raise InvalidMolecule(f"Molecule must have at least {MIN_ATOMS} atoms: {smiles}")
    return Chem.MolToSmiles(mol, isomericSmiles=False)


def parse_smiles(smiles: str, *, detailed: bool = False) -> tuple[RdkMol, Molecule]:
    """Parse SMILES into an RDKit mol (canonical atom order) and a :class:`Molecule`.

    Re-parses the canonical SMILES so atom indices match that string. Bond
    ``idx`` follows ``mol.GetBonds()``.
    """
    canonical = canonicalize_smiles(smiles)
    mol = Chem.MolFromSmiles(canonical)
    if mol is None:
        raise InvalidMolecule(f"Canonical SMILES could not be re-parsed: {canonical}")
    return mol, molecule_from_rdkit(mol, smiles=canonical, detailed=detailed)


def molecule_from_rdkit(
    mol: RdkMol,
    *,
    smiles: Optional[str] = None,
    detailed: bool = False,
    name: Optional[dict[str, Union[int, str]]] = None,
) -> Molecule:
    """Build topology-only :class:`Molecule` from an RDKit mol."""
    if smiles is None:
        smiles = Chem.MolToSmiles(mol, isomericSmiles=False)
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
        order = mol.GetPropsAsDict(True, True).get("_smilesAtomOutputOrder")
        if order is not None:
            atoms["reordered"] = list(order)

    return Molecule(
        smiles=smiles,
        atoms=Atoms(**atoms),
        bonds=Bonds(**bonds),
        name=name or {},
        results=[],
    )


def as_molecule(inp: Union[str, Molecule]) -> tuple[Optional[RdkMol], Molecule]:
    """Accept SMILES or an existing :class:`Molecule`. Parse only when needed."""
    if isinstance(inp, Molecule):
        return None, inp
    return parse_smiles(inp)
