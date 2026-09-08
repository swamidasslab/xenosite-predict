"""ORT SessionOptions: profiling off unless XENOSITE_ORT_PROFILE is a real path."""

from __future__ import annotations

from pathlib import Path

import pytest

from xenosite.predict.backends.onnx import (
    ENV_ORT_PROFILE,
    OnnxBackend,
    _ort_profile_prefix,
    _session_options,
    ort,
)

from tests.support import ROOT

STUB_ONNX = ROOT / "tests" / "fixtures" / "stub_onnx"


pytestmark = pytest.mark.skipif(ort is None, reason="onnxruntime is not installed")


def test_session_options_disable_profiling_by_default():
    opts = _session_options(env={})
    assert opts.enable_profiling is False
    assert opts.enable_mem_pattern is False
    assert opts.profile_file_prefix == "ort_profile"
    assert opts.intra_op_num_threads == 1
    assert opts.inter_op_num_threads == 1


def test_session_options_honor_ort_intra_env():
    opts = _session_options(env={"XENOSITE_ORT_INTRA_OP": "4"})
    assert opts.intra_op_num_threads == 4
    assert opts.inter_op_num_threads == 1


@pytest.mark.parametrize("value", ["", "  ", "1", "true", "TRUE", "yes", "on"])
def test_profile_env_flags_do_not_enable(value):
    assert _ort_profile_prefix({ENV_ORT_PROFILE: value}) is None
    opts = _session_options(env={ENV_ORT_PROFILE: value})
    assert opts.enable_profiling is False
    assert opts.profile_file_prefix == "ort_profile"


def test_profile_env_missing_parent_is_ignored(tmp_path):
    dest = tmp_path / "missing" / "ort.json"
    assert _ort_profile_prefix({ENV_ORT_PROFILE: str(dest)}) is None
    opts = _session_options(env={ENV_ORT_PROFILE: str(dest)})
    assert opts.enable_profiling is False


def test_profile_env_directory_is_ignored(tmp_path):
    opts = _session_options(env={ENV_ORT_PROFILE: str(tmp_path)})
    assert opts.enable_profiling is False


def test_profile_env_real_filename_enables(tmp_path):
    dest = tmp_path / "ort_profile.json"
    assert _ort_profile_prefix({ENV_ORT_PROFILE: str(dest)}) == str(dest)
    opts = _session_options(env={ENV_ORT_PROFILE: str(dest)})
    assert opts.enable_profiling is True
    assert opts.profile_file_prefix == str(dest)
    assert opts.enable_mem_pattern is False


def test_opening_session_does_not_dump_mem_sess(tmp_path, monkeypatch):
    if not (STUB_ONNX / "ugt" / "atom.onnx").is_file():
        pytest.skip("stub ONNX fixture missing")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv(ENV_ORT_PROFILE, raising=False)
    OnnxBackend(STUB_ONNX).session("ugt", "atom")
    dumped = [
        p
        for p in Path.cwd().iterdir()
        if p.is_file() and (":mem:" in p.name or p.suffix == ".sess" or p.name.endswith(".sess"))
    ]
    assert dumped == []
