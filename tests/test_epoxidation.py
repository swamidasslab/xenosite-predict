"""Epoxidation golden / ONNX tests — skip without weights. 13 example molecules."""

from __future__ import annotations

import pytest

from xenosite.predict import WeightsNotFound, predict
from xenosite.predict.backends.onnx import OnnxBackend
from xenosite.predict.molecule import parse_smiles

from tests.support import ROOT, load_golden, onnx_weights_present

EXAMPLE_SMILES = [
    r"C/C(=C\c1ccc(CO)cc1)c1ccc2c(c1)C(C)(C)C(O)CC2(C)C",
    "O=C(C)Oc1ccccc1C(=O)O",
    "O(c1ccc(cc1)CCOC)CC(O)CNC(C)C",
    "CN1C(C(=O)NC2=NC=CS2)=C(O)C2=CC=CC=C2S1(=O)=O",
    "O=CC(=C)c1ccccc1",
    "Clc1cc2nccc(c2cc1)Nc3cc(c(O)cc3)CN(CC)CC",
    "CN(C)CCOC(C1=CC=CC=C1)C1=CC=CC=C1",
    "CS(=O)(=O)c1ccc(-c2cn3ccccc3n2)cc1",
    "O=C(Br)C(F)(F)F",
    "CC(C)C(Br)C(=O)NC(=O)N",
    "CCC(=O)O[C@H]1CC[C@H]2[C@@H]3CCC4=CC(=O)CC[C@]4(C)[C@H]3CC[C@]12C",
    r"c1c(cccc1)C(c2ccccc2)N3CCN(CC3)C\C=C\c4ccccc4",
    r"C(#C\C=C\CN(C)Cc2cccc1ccccc12)C(C)(C)C",
]


def test_epoxidation_parse_all_examples():
    for smi in EXAMPLE_SMILES:
        _, molecule = parse_smiles(smi)
        assert molecule.atoms.num >= 2


def test_epoxidation_onnx_skips_without_weights():
    if onnx_weights_present("epoxidation"):
        be = OnnxBackend(ROOT / "weights" / "onnx")
        mol = predict(EXAMPLE_SMILES[1], models=["epoxidation"], backend=be)
        assert mol.results
        assert mol.results[0].model == "epoxidation"
        assert len(mol.results[0].bond) == len(mol.bonds.idx)
    else:
        be = OnnxBackend(ROOT / "weights" / "onnx")
        with pytest.raises(WeightsNotFound):
            predict(EXAMPLE_SMILES[1], models=["epoxidation"], backend=be)


def test_epoxidation_golden_if_present():
    rows = [g for g in load_golden() if g.get("model") == "epoxidation"]
    if not rows:
        pytest.skip("no golden scores committed yet (gather after ONNX convert)")
    if not onnx_weights_present("epoxidation"):
        pytest.skip("no epoxidation ONNX")
    be = OnnxBackend(ROOT / "weights" / "onnx")
    from xenosite.predict.compare import assert_equiv_results

    for g in rows:
        mol = predict(g["smiles"], models=["epoxidation"], backend=be)
        got = mol.results[0].model_dump()
        assert_equiv_results({"bond": g["bond"], "mol": g["mol"]}, {"bond": got["bond"], "mol": got["mol"]})
