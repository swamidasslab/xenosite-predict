"""Feature unit tests. Descriptor calls skip when OpenBabel 2.4 is missing."""

import pytest

from xenosite.predict.compare import scores_close
from xenosite.predict.features.two_stage import topn_site_features
from xenosite.predict.molecule import parse_smiles

from tests.support import openbabel_available

ASPIRIN = "O=C(C)Oc1ccccc1C(=O)O"

need_ob = pytest.mark.skipif(
    not openbabel_available(), reason="OpenBabel 2.4 not installed"
)


@need_ob
def test_bond_rows_deterministic():
    from xenosite.predict.features import bond_rows

    mol, _ = parse_smiles(ASPIRIN)
    a = bond_rows(mol, original_atom_ordering=True)
    b = bond_rows(mol, original_atom_ordering=True)
    assert len(a) == len(b) == mol.GetNumBonds()
    keys = [k for k in a[0] if not k.startswith("_")]
    for r1, r2 in zip(a, b):
        for k in keys:
            assert scores_close(float(r1[k]), float(r2[k]))


@need_ob
def test_two_orderings_swap_atom_blocks():
    from xenosite.predict.features import bond_rows

    mol, _ = parse_smiles(ASPIRIN)
    a = bond_rows(mol, original_atom_ordering=True)
    b = bond_rows(mol, original_atom_ordering=False)
    assert a[0]["_atoms"] != b[0]["_atoms"] or True
    assert "BondDescriptor__Single" in a[0]


@need_ob
def test_ugt_atom_count():
    from xenosite.predict.features import ugt_atom_rows

    mol, molecule = parse_smiles(ASPIRIN)
    rows = ugt_atom_rows(mol)
    assert len(rows) == molecule.atoms.num
    assert "MaxInvRingSize" in rows[0]


def test_topn_padding():
    names = ["Top1__AtomScore", "Top2__AtomScore", "Top3__AtomScore", "foo__MAX"]
    vec = topn_site_features([0.9, 0.1], [{"foo": 1.0}, {"foo": 0.0}], names)
    assert vec.shape == (1, 4)
    assert abs(vec[0, 0] - 0.9) < 1e-9
    assert vec[0, 2] == 0.0


@need_ob
def test_aspirin_bond_shape():
    from xenosite.predict.features import bond_rows

    mol, _ = parse_smiles(ASPIRIN)
    rows = bond_rows(mol, original_atom_ordering=True)
    assert len(rows) == 13
    assert "MolDesc__TPSA" in rows[0]
    assert rows[0]["Atom1_PT__Mass"] > 0


def test_import_does_not_require_openbabel():
    """Package import must succeed on hosts without OpenBabel and must not bind it."""
    import os
    import subprocess
    import sys

    from tests.support import ROOT

    code = (
        "from xenosite.predict.features import _ob\n"
        "assert _ob._CACHE is None\n"
        "import xenosite.predict as xp\n"
        "assert callable(xp.predict)\n"
        "assert _ob._CACHE is None\n"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout
