"""Per-model skip-without-weights tests (quinone, reactivity, ugt, ndealk, isozyme)."""

import pytest

from xenosite.predict import WeightsNotFound, predict
from xenosite.predict.backends.onnx import OnnxBackend
from xenosite.predict.errors import ModelNotAvailable, OpenBabelNotAvailable

from tests.support import ROOT, onnx_weights_present

ASPIRIN = "O=C(C)Oc1ccccc1C(=O)O"
NULL_PAIR = "O=C(Br)C(F)(F)F"  # quinone null-pair molecule
NDEALK_OFFBY = "CCCC1CCCNC1C=O"


@pytest.mark.parametrize("model", ["quinone", "ugt", "ndealk", "isozyme", "reactivity"])
def test_onnx_skips_or_runs(model):
    be = OnnxBackend(ROOT / "weights" / "onnx")
    key = "ndealk" if model == "isozyme" else model
    if onnx_weights_present(key):
        try:
            mol = predict(ASPIRIN, models=[model], backend=be)
        except (WeightsNotFound, OpenBabelNotAvailable) as exc:
            pytest.skip(str(exc))
        assert mol.results
    else:
        with pytest.raises((WeightsNotFound, ModelNotAvailable, OpenBabelNotAvailable)):
            predict(ASPIRIN, models=[model], backend=be)


def test_quinone_null_pair_molecule_parses():
    from xenosite.predict.molecule import parse_smiles

    _, m = parse_smiles(NULL_PAIR)
    assert m.atoms.num >= 2


def test_ndealk_offby1_molecule_parses():
    from xenosite.predict.molecule import parse_smiles

    _, m = parse_smiles(NDEALK_OFFBY)
    assert m.atoms.num >= 2


def test_phase1_onnx_not_faked():
    be = OnnxBackend(ROOT / "weights" / "onnx")
    with pytest.raises(WeightsNotFound):
        predict(ASPIRIN, models=["phase1"], backend=be)


def test_bioactivation_onnx_blocked():
    be = OnnxBackend(ROOT / "weights" / "onnx")
    with pytest.raises(ModelNotAvailable):
        predict(ASPIRIN, models=["bioactivation"], backend=be)
