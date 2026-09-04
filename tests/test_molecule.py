"""Molecule parse / canonicalize tests (no name lookup, no Docker)."""

import pytest

from xenosite.predict import InvalidMolecule, Molecule, predict
from xenosite.predict.molecule import canonicalize_smiles, parse_smiles


def test_canonical_non_isomeric():
    assert canonicalize_smiles("OCCCC") == "CCCCO"


def test_stereo_stripped():
    # Testosterone propionate-like stereo should collapse when isomericSmiles=False
    smi = "C[C@H](O)C"
    out = canonicalize_smiles(smi)
    assert "@" not in out


def test_invalid_smiles():
    with pytest.raises(InvalidMolecule):
        parse_smiles("INVALID")


def test_too_small():
    with pytest.raises(InvalidMolecule):
        parse_smiles("C")


def test_topology_aspirin():
    mol, molecule = parse_smiles("O=C(C)Oc1ccccc1C(=O)O", detailed=True)
    assert molecule.smiles
    assert molecule.atoms.num >= 2
    assert len(molecule.bonds.idx) == mol.GetNumBonds()
    assert molecule.atoms.z is not None
    assert all(i >= 0 for pair in molecule.bonds.idx for i in pair)


def test_detailed_omitted_by_default():
    _, molecule = parse_smiles("OCCCC")
    assert molecule.smiles == "CCCCO"
    assert molecule.atoms.z is None
    assert molecule.atoms.reordered is None
    assert molecule.bonds.order is None


def test_detailed_atom_bond_fields():
    _, molecule = parse_smiles("OCCCC", detailed=True)
    assert molecule.smiles == "CCCCO"
    assert molecule.atoms.num == 5
    assert molecule.atoms.z == [6, 6, 6, 6, 8]
    assert molecule.atoms.chrg == [0, 0, 0, 0, 0]
    assert molecule.atoms.impHs == [3, 2, 2, 2, 1]
    assert molecule.atoms.cipRank == [0, 2, 4, 3, 1]
    assert molecule.atoms.reordered == [4, 3, 2, 1, 0]
    assert molecule.bonds.idx == [(0, 1), (1, 2), (2, 3), (3, 4)]
    assert molecule.bonds.order == [1.0, 1.0, 1.0, 1.0]


def test_canonical_atom_order():
    """Scores and topology use canonical SMILES atom order, not the input order."""
    mol, molecule = parse_smiles("OCCCC", detailed=True)
    assert [a.GetAtomicNum() for a in mol.GetAtoms()] == [6, 6, 6, 6, 8]
    assert molecule.atoms.z[-1] == 8
    assert molecule.atoms.reordered[0] == 4


def test_already_canonical_reordered_is_identity():
    _, molecule = parse_smiles("CCCCO", detailed=True)
    assert molecule.atoms.reordered == [0, 1, 2, 3, 4]


def test_as_molecule_detailed_fills_existing():
    from xenosite.predict.molecule import as_molecule

    _, molecule = parse_smiles("OCCCC")
    assert molecule.atoms.z is None
    rdkit_mol, filled = as_molecule(molecule, detailed=True)
    assert rdkit_mol is None
    assert filled is molecule
    assert molecule.atoms.z == [6, 6, 6, 6, 8]
    # Existing Molecule is already canonical, so reordered is identity.
    assert molecule.atoms.reordered == [0, 1, 2, 3, 4]


def test_rdkit_default_omitted():
    mol, molecule = parse_smiles("CCCCO")
    assert molecule.rdkit is None
    assert mol.GetNumAtoms() == 5
    assert "rdkit" not in molecule.model_dump()


def test_rdkit_flag_keeps_parse_mol():
    from rdkit import Chem

    mol, molecule = parse_smiles("OCCCC", rdkit=True)
    assert molecule.rdkit is mol
    assert Chem.MolToSmiles(molecule.rdkit, isomericSmiles=False) == "CCCCO"
    assert "rdkit" not in molecule.model_dump()


def test_as_molecule_rdkit_fills_existing():
    from rdkit import Chem

    from xenosite.predict.molecule import as_molecule

    _, molecule = parse_smiles("OCCCC")
    assert molecule.rdkit is None
    rdkit_mol, filled = as_molecule(molecule, rdkit=True)
    assert filled is molecule
    assert rdkit_mol is molecule.rdkit
    assert Chem.MolToSmiles(rdkit_mol, isomericSmiles=False) == "CCCCO"


def test_append_requires_backend(tmp_path, monkeypatch):
    from xenosite.predict.errors import BackendNotConfigured

    monkeypatch.chdir(tmp_path)
    with pytest.raises(BackendNotConfigured):
        predict("CCCCO", models=["epoxidation"], env={})
