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


def test_append_requires_backend(tmp_path, monkeypatch):
    from xenosite.predict.errors import BackendNotConfigured

    monkeypatch.chdir(tmp_path)
    with pytest.raises(BackendNotConfigured):
        predict("CCCCO", models=["epoxidation"], env={})
