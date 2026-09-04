"""User-facing ``predict`` / ``list_models`` smoke tests."""

from __future__ import annotations

import pytest

from xenosite.predict import list_models, predict
from xenosite.predict.backends.onnx import OnnxBackend
from xenosite.predict.features import _ob

from tests.support import ROOT, onnx_weights_present, onnx_root

ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"

ALL_ONNX = (
    "epoxidation",
    "quinone",
    "reactivity",
    "ugt",
    "ndealk",
    "phase1",
)


@pytest.mark.parametrize("model", ALL_ONNX)
def test_predict_single_model(model):
    if not onnx_weights_present(model):
        pytest.skip(f"no ONNX for {model}")
    if not _ob.installed():
        pytest.skip("OpenBabel not installed")
    be = OnnxBackend(onnx_root())
    mol = predict(ASPIRIN, model=model, backend=be)
    assert mol.smiles
    assert mol.atoms.num >= 2
    assert len(mol.bonds.idx) >= 1
    assert mol.results
    assert all(r.model_version == "1" for r in mol.results)


def test_predict_multi_model_appends():
    if not all(onnx_weights_present(m) for m in ("epoxidation", "quinone")):
        pytest.skip("missing ONNX weights")
    if not _ob.installed():
        pytest.skip("OpenBabel not installed")
    be = OnnxBackend(onnx_root())
    mol = predict(ASPIRIN, models=["epoxidation", "quinone"], backend=be)
    heads = {r.model for r in mol.results}
    assert "epoxidation" in heads
    assert "quinone" in heads
    assert all(r.model_version == "1" for r in mol.results)


def test_predict_v0_stamps_legacy_version():
    if not onnx_weights_present("epoxidation"):
        pytest.skip("no epoxidation ONNX")
    if not _ob.installed():
        pytest.skip("OpenBabel not installed")
    be = OnnxBackend(onnx_root())
    mol = predict(ASPIRIN, models=[("epoxidation", "0")], backend=be)
    assert mol.results
    assert all(r.model_version == "0" for r in mol.results)
    dumped = mol.results[0].model_dump()
    assert dumped["model_version"] == "0"
    assert "version" not in dumped
    assert mol._parameter["ndealk_site_mode"] == "legacy"
    assert mol._parameter["symmetry_group_mode"] == "openbabel"


def test_predict_reuses_molecule_object():
    if not onnx_weights_present("epoxidation"):
        pytest.skip("no epoxidation ONNX")
    if not _ob.installed():
        pytest.skip("OpenBabel not installed")
    be = OnnxBackend(onnx_root())
    mol = predict(ASPIRIN, model="epoxidation", backend=be)
    n0 = len(mol.results)
    mol2 = predict(mol, model="quinone", backend=be)
    assert mol2 is mol
    assert len(mol.results) > n0


def test_list_models_reports_phase1():
    rows = list_models(env={"XENOSITE_MODELS_WEIGHTS": str(onnx_root())})
    versions = {(r["name"], r["version"]) for r in rows}
    assert ("phase1", "0") in versions
    assert ("phase1", "1") in versions
    by = {(r["name"], r["version"]): r for r in rows}
    v1 = by[("phase1", "1")]
    assert v1["two_stage"] is True
    assert "stable_oxygenation" in v1["heads"]
    if onnx_weights_present("phase1") and _ob.installed():
        assert v1["available"] is True
        assert by[("phase1", "0")]["available"] is True
