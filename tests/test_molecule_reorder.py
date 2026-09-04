"""Unit tests for molecule reorder / strip presentation helpers."""

from __future__ import annotations

import pytest

from xenosite.predict.errors import InvalidMolecule
from xenosite.predict.molecule import (
    apply_presentation,
    in_input_order,
    parse_smiles,
    strip_details,
)
from xenosite.predict.types import (
    AtomResult,
    Atoms,
    Bonds,
    Metabolite,
    MolAtomPairResult,
    MolBondResult,
    Molecule,
)


def test_parse_captures_reordered_and_input_smiles():
    mol_rd, molecule = parse_smiles("OCCCC", detailed=True)
    assert molecule.smiles == "CCCCO"
    assert molecule.atoms.reordered == [4, 3, 2, 1, 0]
    assert molecule._input_smiles == "OCCCC"
    assert molecule.atoms.z == [6, 6, 6, 6, 8]


def test_in_input_order_atom_scores():
    _, molecule = parse_smiles("OCCCC", detailed=True)
    # Canonical CCCCO: O is last → high score on last atom
    molecule.results.append(
        AtomResult(
            model="ugt",
            model_version="1",
            atom=[0.0, 0.0, 0.0, 0.0, 0.9],
        )
    )
    in_input_order(molecule, input_smiles="OCCCC")
    assert molecule.smiles == "OCCCC"
    # Input OCCCC: O is first
    assert molecule.results[0].atom[0] == pytest.approx(0.9)
    assert molecule.atoms.z[0] == 8
    assert molecule.atoms.reordered == [0, 1, 2, 3, 4]


def test_in_input_order_bonds_and_pairs():
    _, molecule = parse_smiles("OCCCC", detailed=True)
    n_bonds = len(molecule.bonds.idx)
    molecule.results.append(
        MolBondResult(
            model="epoxidation",
            model_version="1",
            mol=0.1,
            bond=[float(i) for i in range(n_bonds)],
        )
    )
    molecule.results.append(
        MolAtomPairResult(
            model="quinone",
            model_version="1",
            mol=0.2,
            atom=[0.0] * 5,
            pair=[0.5],
            pair_idx=[(0, 4)],  # C0–O in canonical CCCCO
        )
    )
    in_input_order(molecule, input_smiles="OCCCC")
    # pair (0,4) canonical → input (reordered[0], reordered[4]) = (4, 0)
    assert molecule.results[1].pair_idx == [(4, 0)]
    assert len(molecule.bonds.idx) == n_bonds


def test_in_input_order_metabolite_atom_map():
    _, molecule = parse_smiles("OCCCC", detailed=True)
    molecule.results.append(
        AtomResult(
            model="ugt",
            model_version="1",
            atom=[0.1] * 5,
            metabolite=[
                Metabolite(
                    smiles="CCCCO",
                    atom=[4],
                    map_idx=[1, 2, 3, 4, 5],
                    pathway="UGT",
                    score=0.5,
                )
            ],
        )
    )
    in_input_order(molecule, input_smiles="OCCCC")
    met = molecule.results[0].metabolite[0]
    assert met.atom == [0]  # canonical 4 → input 0
    # map_idx 1-based canonical parents → input
    assert met.map_idx == [5, 4, 3, 2, 1]


def test_strip_details():
    _, molecule = parse_smiles("CCO", detailed=True)
    strip_details(molecule)
    assert molecule.atoms.z is None
    assert molecule.atoms.reordered is None
    assert molecule.bonds.order is None
    assert molecule.atoms.num == 3


def test_apply_presentation_combos():
    _, molecule = parse_smiles("OCCCC", detailed=True)
    molecule.results.append(
        AtomResult(model="ugt", model_version="1", atom=[0.0, 0.0, 0.0, 0.0, 1.0])
    )
    apply_presentation(molecule, canonicalize=False, detailed=False, input_smiles="OCCCC")
    assert molecule.smiles == "OCCCC"
    assert molecule.results[0].atom[0] == pytest.approx(1.0)
    assert molecule.atoms.z is None


def test_in_input_order_requires_reordered():
    molecule = Molecule(
        smiles="CCCCO",
        atoms=Atoms(num=5),
        bonds=Bonds(idx=[(0, 1), (1, 2), (2, 3), (3, 4)]),
        results=[],
    )
    with pytest.raises(InvalidMolecule, match="reordered"):
        in_input_order(molecule, input_smiles="OCCCC")
