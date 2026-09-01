"""Epoxidation parse and ONNX smoke tests. 13 example molecules."""

from __future__ import annotations

import pytest

from xenosite.predict import predict
from xenosite.predict.backends.onnx import OnnxBackend
from xenosite.predict.molecule import parse_smiles

from tests.support import ROOT, golden_name_by_smiles, onnx_weights_present, onnx_root

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


def _example_ids():
    names = golden_name_by_smiles()
    return [names.get(s) or s[:24] for s in EXAMPLE_SMILES]


@pytest.mark.parametrize("smiles", EXAMPLE_SMILES, ids=_example_ids())
def test_epoxidation_parse(smiles):
    _, molecule = parse_smiles(smiles)
    assert molecule.atoms.num >= 2


def test_epoxidation_onnx_predicts():
    assert onnx_weights_present("epoxidation"), "no epoxidation ONNX under weights/onnx"
    mol = predict(
        EXAMPLE_SMILES[1],
        models=["epoxidation"],
        backend=OnnxBackend(onnx_root()),
    )
    assert mol.results
    assert mol.results[0].model == "epoxidation"
    assert len(mol.results[0].bond) == len(mol.bonds.idx)
