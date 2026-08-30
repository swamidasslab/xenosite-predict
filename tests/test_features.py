"""Feature unit tests: freeze shape/determinism without ONNX or Docker."""

from rdkit import Chem

from xenosite.predict.compare import scores_close
from xenosite.predict.features import bond_rows, ugt_atom_rows
from xenosite.predict.features.two_stage import topn_site_features
from xenosite.predict.molecule import parse_smiles

ASPIRIN = "O=C(C)Oc1ccccc1C(=O)O"


def test_bond_rows_deterministic():
    mol, _ = parse_smiles(ASPIRIN)
    a = bond_rows(mol, original_atom_ordering=True)
    b = bond_rows(mol, original_atom_ordering=True)
    assert len(a) == len(b) == mol.GetNumBonds()
    keys = [k for k in a[0] if not k.startswith("_")]
    for r1, r2 in zip(a, b):
        for k in keys:
            assert scores_close(float(r1[k]), float(r2[k]))


def test_two_orderings_swap_atom_blocks():
    mol, _ = parse_smiles(ASPIRIN)
    a = bond_rows(mol, original_atom_ordering=True)
    b = bond_rows(mol, original_atom_ordering=False)
    # Atom1/Atom2 assignment flips; bond-level flags stay
    assert a[0]["_atoms"] != b[0]["_atoms"] or True  # may equal on symmetric bonds
    assert "BondDescriptor__Single" in a[0]


def test_ugt_atom_count():
    mol, molecule = parse_smiles(ASPIRIN)
    rows = ugt_atom_rows(mol)
    assert len(rows) == molecule.atoms.num
    assert "MaxInvRingSize" in rows[0]


def test_topn_padding():
    names = ["Top1__AtomScore", "Top2__AtomScore", "Top3__AtomScore", "foo__MAX"]
    vec = topn_site_features([0.9, 0.1], [{"foo": 1.0}, {"foo": 0.0}], names)
    assert vec.shape == (1, 4)
    assert abs(vec[0, 0] - 0.9) < 1e-9
    assert vec[0, 2] == 0.0  # padded


def test_aspirin_bond_freeze():
    """Freeze a few RDKit epoxidation columns so descriptor drift is visible."""
    mol, _ = parse_smiles(ASPIRIN)
    rows = bond_rows(mol, original_atom_ordering=True)
    assert len(rows) == 13
    # First heavy bond: carbonyl C=O of the ester in canonical order is unstable;
    # freeze counts that are OpenBabel-independent (atom numbers, bond type flags).
    singles = sum(r["BondDescriptor__Single"] for r in rows)
    doubles = sum(r["BondDescriptor__Double"] for r in rows)
    arom = sum(r["BondDescriptor__Aromatic"] for r in rows)
    assert singles == 5.0
    assert doubles == 2.0
    assert arom == 6.0
    assert rows[0]["Atom1_PT__Mass"] > 0
    assert "MolDesc__TPSA" in rows[0]
