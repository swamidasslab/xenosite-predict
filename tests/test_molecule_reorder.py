"""Unit tests for molecule reorder / strip presentation helpers."""

from __future__ import annotations

import copy

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


def _assert_atom_vector_roundtrip(can_vals, inp_vals, reordered):
    """``can_vals[i]`` and ``inp_vals[reordered[i]]`` describe the same atom."""
    assert len(can_vals) == len(inp_vals) == len(reordered)
    for can_i, inp_i in enumerate(reordered):
        assert inp_vals[inp_i] == pytest.approx(can_vals[can_i])


def test_parse_captures_reordered_and_input_smiles():
    _, molecule = parse_smiles("OCCCC", detailed=True)
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
            pair=[0.5, 0.25],
            pair_idx=[(0, 4), (1, 3)],  # canonical pairs
        )
    )
    in_input_order(molecule, input_smiles="OCCCC")
    # pair indices remap; pair *scores* stay aligned with each remapped entry.
    assert molecule.results[1].pair_idx == [(4, 0), (3, 1)]
    assert molecule.results[1].pair == [0.5, 0.25]
    assert len(molecule.bonds.idx) == n_bonds


def test_detailed_fields_roundtrip_with_bonds_order():
    _, can = parse_smiles("OCCCC", detailed=True)
    reordered = list(can.atoms.reordered)
    can_fields = {
        "z": list(can.atoms.z),
        "chrg": list(can.atoms.chrg),
        "impHs": list(can.atoms.impHs),
        "cipRank": list(can.atoms.cipRank),
    }
    can_order = list(can.bonds.order)
    can_bonds = list(can.bonds.idx)
    inp = copy.deepcopy(can)
    in_input_order(inp, input_smiles="OCCCC")
    for name, can_vals in can_fields.items():
        _assert_atom_vector_roundtrip(can_vals, getattr(inp.atoms, name), reordered)
    # Bond order stays parallel to remapped idx (same chemical bonds).
    assert inp.bonds.order == can_order
    for i, (a, b) in enumerate(can_bonds):
        assert set(inp.bonds.idx[i]) == {reordered[a], reordered[b]}


def test_canonicalize_false_preserves_stereo_input_smiles():
    """Presentation restores the exact input string; stereo is not re-emitted from RDKit."""
    stereo = "O[C@H](C)CC"
    _, molecule = parse_smiles(stereo, detailed=True)
    assert "@" not in molecule.smiles
    assert molecule.smiles == "CCC(C)O"
    molecule.results.append(
        AtomResult(model="ugt", model_version="1", atom=[0.0] * molecule.atoms.num)
    )
    in_input_order(molecule, input_smiles=stereo)
    assert molecule.smiles == stereo
    assert "@" in molecule.smiles


def test_bond_scores_stay_aligned_with_remapped_idx():
    """Bond score arrays are not permuted; they stay parallel to remapped ``bonds.idx``."""
    _, molecule = parse_smiles("OCCCC", detailed=True)
    reordered = list(molecule.atoms.reordered)
    # Canonical bond C–O is (3, 4).
    bond_i = next(
        i for i, (a, b) in enumerate(molecule.bonds.idx) if {a, b} == {3, 4}
    )
    scores = [0.0] * len(molecule.bonds.idx)
    scores[bond_i] = 0.77
    molecule.results.append(
        MolBondResult(model="epoxidation", model_version="1", mol=0.1, bond=scores)
    )
    in_input_order(molecule, input_smiles="OCCCC")
    assert molecule.results[0].bond[bond_i] == pytest.approx(0.77)
    assert set(molecule.bonds.idx[bond_i]) == {reordered[3], reordered[4]}
    # Input order: O at 0, last C at 1 → C–O is (0, 1).
    assert set(molecule.bonds.idx[bond_i]) == {0, 1}


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


def test_metabolite_map_idx_keeps_zero_for_new_atoms():
    _, molecule = parse_smiles("OCCCC", detailed=True)
    molecule.results.append(
        AtomResult(
            model="ugt",
            model_version="1",
            atom=[0.1] * 5,
            metabolite=[
                Metabolite(
                    smiles="X",
                    atom=[4],
                    map_idx=[0, 5, 0],  # new, parent can-4, new
                    pathway="UGT",
                    score=0.5,
                )
            ],
        )
    )
    in_input_order(molecule, input_smiles="OCCCC")
    assert molecule.results[0].metabolite[0].map_idx == [0, 1, 0]


def test_strip_details():
    _, molecule = parse_smiles("CCO", detailed=True)
    strip_details(molecule)
    assert molecule.atoms.z is None
    assert molecule.atoms.reordered is None
    assert molecule.bonds.order is None
    assert molecule.atoms.num == 3


@pytest.mark.parametrize(
    ("canonicalize", "detailed", "expect_smiles", "expect_z0", "expect_reordered"),
    [
        (True, True, "CCCCO", 6, [4, 3, 2, 1, 0]),
        (True, False, "CCCCO", None, None),
        (False, True, "OCCCC", 8, [0, 1, 2, 3, 4]),
        (False, False, "OCCCC", None, None),
    ],
)
def test_apply_presentation_all_flag_combos(
    canonicalize, detailed, expect_smiles, expect_z0, expect_reordered
):
    _, molecule = parse_smiles("OCCCC", detailed=True)
    molecule.results.append(
        AtomResult(model="ugt", model_version="1", atom=[0.0, 0.0, 0.0, 0.0, 1.0])
    )
    apply_presentation(
        molecule,
        canonicalize=canonicalize,
        detailed=detailed,
        input_smiles="OCCCC",
    )
    assert molecule.smiles == expect_smiles
    if expect_z0 is None:
        assert molecule.atoms.z is None
        assert molecule.atoms.reordered is None
        assert molecule.bonds.order is None
    else:
        assert molecule.atoms.z[0] == expect_z0
        assert molecule.atoms.reordered == expect_reordered
    # High score is always on oxygen, wherever O sits after presentation.
    o_idx = 4 if canonicalize else 0
    assert molecule.results[0].atom[o_idx] == pytest.approx(1.0)


def test_atom_score_roundtrip_via_reordered():
    _, can = parse_smiles("OCCCC", detailed=True)
    reordered = list(can.atoms.reordered)
    can.results.append(
        AtomResult(
            model="ugt",
            model_version="1",
            atom=[0.1, 0.2, 0.3, 0.4, 0.9],
        )
    )
    inp = copy.deepcopy(can)
    in_input_order(inp, input_smiles="OCCCC")
    _assert_atom_vector_roundtrip(can.results[0].atom, inp.results[0].atom, reordered)
    _assert_atom_vector_roundtrip(can.atoms.z, inp.atoms.z, reordered)


@pytest.mark.parametrize(
    "smiles",
    [
        "OCCCC",  # chain, reverse map
        "O[C@H](C)CC",  # branched + stereo stripped in canonical
        "c1ccc(O)cc1",  # aromatic ring
        "O1CCCCC1",  # aliphatic heterocycle
    ],
)
def test_in_input_order_roundtrip_varied_topologies(smiles):
    _, can = parse_smiles(smiles, detailed=True)
    assert can.smiles != smiles or can.atoms.reordered != list(range(can.atoms.num))
    reordered = list(can.atoms.reordered)
    n = can.atoms.num
    can.results.append(
        AtomResult(
            model="ugt",
            model_version="1",
            atom=[float(i) for i in range(n)],
        )
    )
    # Distinct bond scores parallel to idx.
    n_bonds = len(can.bonds.idx)
    can.results.append(
        MolBondResult(
            model="epoxidation",
            model_version="1",
            mol=0.0,
            bond=[10.0 + i for i in range(n_bonds)],
        )
    )
    can_z = list(can.atoms.z)
    can_atom = list(can.results[0].atom)
    can_bonds = list(can.bonds.idx)
    can_bond_scores = list(can.results[1].bond)

    inp = copy.deepcopy(can)
    in_input_order(inp, input_smiles=smiles)

    assert inp.smiles == smiles
    _assert_atom_vector_roundtrip(can_atom, inp.results[0].atom, reordered)
    _assert_atom_vector_roundtrip(can_z, inp.atoms.z, reordered)
    for i, (a, b) in enumerate(can_bonds):
        assert set(inp.bonds.idx[i]) == {reordered[a], reordered[b]}
        assert inp.results[1].bond[i] == pytest.approx(can_bond_scores[i])


def test_in_input_order_identity_when_input_already_canonical():
    _, molecule = parse_smiles("CCCCO", detailed=True)
    assert molecule.atoms.reordered == [0, 1, 2, 3, 4]
    atom = [0.1, 0.2, 0.3, 0.4, 0.5]
    molecule.results.append(AtomResult(model="ugt", model_version="1", atom=list(atom)))
    bonds_before = list(molecule.bonds.idx)
    in_input_order(molecule, input_smiles="CCCCO")
    assert molecule.results[0].atom == atom
    assert molecule.bonds.idx == bonds_before
    assert molecule.smiles == "CCCCO"


def test_in_input_order_requires_reordered():
    molecule = Molecule(
        smiles="CCCCO",
        atoms=Atoms(num=5),
        bonds=Bonds(idx=[(0, 1), (1, 2), (2, 3), (3, 4)]),
        results=[],
    )
    with pytest.raises(InvalidMolecule, match="reordered"):
        in_input_order(molecule, input_smiles="OCCCC")


def test_in_input_order_rejects_reordered_length_mismatch():
    _, molecule = parse_smiles("OCCCC", detailed=True)
    molecule.atoms.reordered = [4, 3, 2]
    with pytest.raises(InvalidMolecule, match="reordered length"):
        in_input_order(molecule, input_smiles="OCCCC")


def test_in_input_order_requires_input_smiles():
    _, molecule = parse_smiles("OCCCC", detailed=True)
    molecule._input_smiles = None
    with pytest.raises(InvalidMolecule, match="input SMILES"):
        in_input_order(molecule)

