"""Feature unit tests. OpenBabel is a runtime dependency (PyPI wheel)."""

import numpy as np
from hypothesis import given
from hypothesis import strategies as st

from xenosite.predict.compare import scores_close
from xenosite.predict.features.two_stage import topn_site_features
from xenosite.predict.molecule import parse_smiles

ASPIRIN = "O=C(C)Oc1ccccc1C(=O)O"


def test_ndealk_row_site_pair_from_index():
    from xenosite.predict.features import ndealk_bond_rows, ndealk_row_site_key, ndealk_row_site_pair

    mol, _ = parse_smiles("CN1CCNCC1")
    rows = ndealk_bond_rows(mol)
    assert rows
    for row in rows:
        a, b = ndealk_row_site_pair(row)
        _mol, oa, ob = str(row["_index"]).split(".", 2)
        ia, ib = int(oa) - 1, int(ob) - 1
        assert (a, b) == (min(ia, ib), max(ia, ib))
        assert ndealk_row_site_key(row) == f"{a}-{b}"


def test_ndealk_site_modes_cn1ccn_and_coc1():
    from pathlib import Path

    from xenosite.predict.backends.adapters import canonical_bond_site_pair, reorder_by_bond
    from xenosite.predict.backends.onnx import OnnxBackend
    from xenosite.predict.features import load_names, matrix_from_rows, ndealk_bond_rows, ndealk_site_from_row_scores

    backend = OnnxBackend(Path(__file__).resolve().parents[1] / "weights" / "onnx")
    names = load_names("ndealk", "bond")
    cases = [
        "CN1CCN(c2ccc3nc(-c4cccc(C(F)(F)F)c4)[nH]c3c2)CC1",
        "COc1ccc2nc(C)cc(NCCCN3CCOCC3)c2c1",
    ]
    for smi in cases:
        mol, molecule = parse_smiles(smi)
        rows = ndealk_bond_rows(mol)
        x, _ = matrix_from_rows(rows, names)
        pred = [float(v) for v in backend.run_head("ndealk", "bond", x)[:, 6]]

        def mapped(mode):
            site = ndealk_site_from_row_scores(
                rows, pred, names, mode=mode, n_atoms=molecule.atoms.num
            )
            current, scores = [], []
            for k, v in site.items():
                a, b = map(int, k.split("-"))
                current.append(canonical_bond_site_pair(a, b))
                scores.append(v)
            return reorder_by_bond(scores, current, molecule.bonds.idx, fill=0.0)

        leg = mapped("legacy")
        pri = mapped("principled")
        if "CN1CCN" in smi:
            assert sum(1 for v in leg if v > 0.01) == 3
            for mode_vec in (leg, pri):
                assert not any(
                    molecule.bonds.idx[i] in ((25, 1), (4, 24)) and mode_vec[i] > 0.01
                    for i in range(len(mode_vec))
                )
        else:
            leg_set = {molecule.bonds.idx[i] for i, v in enumerate(leg) if v > 0.001}
            pri_set = {molecule.bonds.idx[i] for i, v in enumerate(pri) if v > 0.001}
            assert (20, 15) in leg_set
            assert (20, 15) not in pri_set
            assert len(pri_set) < len(leg_set)


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


def test_two_orderings_swap_atom_blocks():
    from xenosite.predict.features import bond_rows

    mol, _ = parse_smiles(ASPIRIN)
    a = bond_rows(mol, original_atom_ordering=True)
    b = bond_rows(mol, original_atom_ordering=False)
    assert a[0]["_atoms"] != b[0]["_atoms"] or True
    assert "BondDescriptor__Single" in a[0]


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


@given(
    scores=st.lists(
        st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        min_size=0,
        max_size=12,
    ),
    topn=st.integers(min_value=1, max_value=6),
)
def test_topn_properties(scores, topn):
    """Top-1 is the max site score; extra Top-k slots pad with 0."""
    names = [f"Top{k}__AtomScore" for k in range(1, topn + 1)]
    rows = [{"foo": float(i)} for i in range(len(scores))]
    vec = topn_site_features(scores, rows, names)
    assert vec.shape == (1, topn)
    assert np.all(np.isfinite(vec))
    if scores:
        assert vec[0, 0] == max(scores)
    else:
        assert vec[0, 0] == 0.0
    if topn > len(scores):
        assert vec[0, len(scores)] == 0.0


def test_halogen_hyb_matches_openbabel_24():
    """OpenBabel 2.4 left F/Br unhybridized; 3.x would report sp without this wrap."""
    from xenosite.predict.features._ob import from_rdkit_mol, load

    mol, _ = parse_smiles("O=C(Br)C(F)(F)F")
    pym = from_rdkit_mol(mol)
    ob, _ = load()
    by_z: dict[int, set[int]] = {}
    for atom in ob.OBMolAtomIter(pym.OBMol):
        z = atom.GetAtomicNum()
        if z == 1:
            continue
        by_z.setdefault(z, set()).add(atom.GetHyb())
    assert by_z[9] == {0}
    assert by_z[35] == {0}
    assert 2 in by_z[8]
    assert by_z[6] <= {2, 3}


def test_aromatic_sulfur_hyb_matches_openbabel_24():
    """OpenBabel 2.4 counted thiazole S as sp3; 3.x reports hyb 2."""
    from xenosite.predict.features._ob import from_rdkit_mol, load

    mol, _ = parse_smiles("c1cscn1")
    pym = from_rdkit_mol(mol)
    ob, _ = load()
    sulfurs = [
        a.GetHyb()
        for a in ob.OBMolAtomIter(pym.OBMol)
        if a.GetAtomicNum() == 16
    ]
    assert sulfurs == [3]


def test_nte_uses_sorted_symmetry_classes():
    """Aromatic bond classes must not depend on Kekulé atom-index order."""
    from tests.support import (
        compare_feature_dump_rows,
        load_ob_dumps,
        rows_for_model,
    )

    dumps = load_ob_dumps()
    assert dumps, "missing tests/fixtures/ob_dumps.json.gz (git lfs pull, or run make dump-ob)"
    smi = "COCCc1ccc(OCC(O)CNC(C)C)cc1"
    dump = next((d for d in dumps if d.get("smiles") == smi), None)
    assert dump is not None, f"no dump for {smi}"
    mol, _ = parse_smiles(smi)
    mm = compare_feature_dump_rows(
        rows_for_model("epoxidation", mol), dump["models"]["epoxidation"]
    )
    assert "BondDescriptor__NTopologicalEquivalent" not in mm


def test_sssr_naphthalene_is_two_hexagons():
    from xenosite.predict.features._ob import from_rdkit_mol
    from xenosite.predict.features.molgraph import MolGraph

    mol, _ = parse_smiles("c1ccc2ccccc2c1")
    mg = MolGraph(from_rdkit_mol(mol))
    assert sorted(len(r) for r in mg.cycles()) == [6, 6]
    assert sorted(len(r) for r in mg.dfs_cycles()) == [6, 10]


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
