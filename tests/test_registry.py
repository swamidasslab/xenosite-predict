"""Registry, list_models, backend picker (env isolation)."""

import pytest

from xenosite.predict import UnknownModel, list_models
from xenosite.predict.backends import BackendNotConfigured, resolve_backend
from xenosite.predict.registry import get_info, load_runner, normalize_models, ensure_builtins


def test_registry_registers_scoring_versions():
    ensure_builtins()
    v0 = get_info("epoxidation", "0")
    v1 = get_info("epoxidation", "1")
    assert v0.default is False
    assert v1.default is True
    assert v0.factory is not v1.factory
    from xenosite.predict.v1.legacy import LegacyRunner

    assert isinstance(load_runner("epoxidation", "0"), LegacyRunner)
    assert not isinstance(load_runner("epoxidation", "1"), LegacyRunner)


def test_normalize_models_default_version():
    specs = normalize_models(["epoxidation", ("ugt", "0")])
    assert specs == [("epoxidation", "1"), ("ugt", "0")]


def test_normalize_rejects_global_version_pattern():
    # A lone string is one model, not a version applied to a list
    assert normalize_models("quinone") == [("quinone", "1")]


def test_unknown_model_predict(monkeypatch):
    from xenosite.predict import predict
    from xenosite.predict.backends.onnx import OnnxBackend

    be = OnnxBackend("/tmp/does-not-exist-onnx")
    with pytest.raises(UnknownModel):
        predict("CCCCO", models=["not-a-model"], backend=be)


def test_list_models_without_backend(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    rows = list_models(env={})
    names = {r["name"] for r in rows}
    assert "epoxidation" in names
    assert "bioactivation" in names
    assert all(r["available"] is False for r in rows)


def test_list_models_local_onnx():
    from tests.support import onnx_root
    from xenosite.predict.features import _ob

    rows = list_models(env={"XENOSITE_MODELS_WEIGHTS": str(onnx_root())})
    by = {r["name"]: r for r in rows}
    if (onnx_root() / "epoxidation").exists():
        if _ob.installed():
            assert by["epoxidation"]["available"] is True
        else:
            assert by["epoxidation"]["available"] is False
            assert "OpenBabel" in by["epoxidation"]["reason"]
    ba = [r for r in rows if r["name"] == "bioactivation"]
    assert ba
    assert all(r["available"] is False for r in ba)


def test_picker_http_url():
    be = resolve_backend(env={"XENOSITE_BACKEND": "https://example.invalid"})
    assert be.name == "http"


def test_picker_weights_dir(tmp_path):
    d = tmp_path / "onnx"
    d.mkdir()
    (d / "epoxidation").mkdir()
    (d / "epoxidation" / "bond.onnx").write_bytes(b"not-a-real-onnx")
    be = resolve_backend(env={"XENOSITE_MODELS_WEIGHTS": str(d)})
    assert be.name == "onnx"
    assert ("epoxidation", "0") in be.available_models()
    assert ("epoxidation", "1") in be.available_models()


def test_picker_error_when_empty():
    with pytest.raises(BackendNotConfigured):
        resolve_backend(env={}, cwd=tmp_path_missing())


def tmp_path_missing():
    from pathlib import Path

    return Path("/tmp/xenosite-predict-no-weights-dir-xyz")


def test_explicit_backend_string(tmp_path):
    d = tmp_path / "w"
    d.mkdir()
    (d / "x.onnx").write_bytes(b"x")
    be = resolve_backend("onnx", env={"XENOSITE_MODELS_WEIGHTS": str(d)})
    assert be.name == "onnx"
