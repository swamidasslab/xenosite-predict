"""Per-model ONNX predict smoke tests (quinone, reactivity, ugt, ndealk, isozyme, phase1)."""

import pytest

from xenosite.predict import predict
from xenosite.predict.backends.onnx import OnnxBackend
from xenosite.predict.errors import ModelNotAvailable
from xenosite.predict.molecule import parse_smiles

from tests.support import ROOT, onnx_weights_present

ASPIRIN = "O=C(C)Oc1ccccc1C(=O)O"
NULL_PAIR = "O=C(Br)C(F)(F)F"  # quinone null-pair molecule
NDEALK_OFFBY = "CCCC1CCCNC1C=O"


@pytest.mark.parametrize(
    "model",
    ["epoxidation", "quinone", "ugt", "ndealk", "isozyme", "reactivity", "phase1"],
)
def test_onnx_predicts(model):
    key = "ndealk" if model == "isozyme" else model
    assert onnx_weights_present(key), f"no ONNX for {key} under weights/onnx"
    mol = predict(ASPIRIN, models=[model], backend=OnnxBackend(ROOT / "weights" / "onnx"))
    assert mol.results


def test_quinone_null_pair_molecule_parses():
    _, m = parse_smiles(NULL_PAIR)
    assert m.atoms.num >= 2


def test_ndealk_offby1_molecule_parses():
    _, m = parse_smiles(NDEALK_OFFBY)
    assert m.atoms.num >= 2


def test_bioactivation_onnx_blocked():
    be = OnnxBackend(ROOT / "weights" / "onnx")
    with pytest.raises(ModelNotAvailable):
        predict(ASPIRIN, models=["bioactivation"], backend=be)
