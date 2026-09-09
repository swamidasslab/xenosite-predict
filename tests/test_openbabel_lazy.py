"""OpenBabel bindings load only when a v0/v1 ONNX predictor runs descriptors."""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

from tests.support import ROOT, onnx_weights_present

_PROBE = r"""
import sys

def openbabel_loaded():
    return any(
        name == "openbabel" or name.startswith("openbabel.") or name == "pybel"
        for name in sys.modules
    )
"""


def _run(code: str) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = (
        str(ROOT / "src") + os.pathsep + str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    )
    return subprocess.run(
        [sys.executable, "-c", _PROBE + "\n" + code],
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        text=True,
    )


def test_import_v1_v2_list_models_do_not_load_openbabel():
    code = r"""
assert not openbabel_loaded()
import xenosite.predict as xp
assert not openbabel_loaded()
import xenosite.predict.v1  # noqa: F401
assert not openbabel_loaded()
import importlib
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    importlib.import_module("xenosite.predict.v0")
assert not openbabel_loaded()
try:
    importlib.import_module("xenosite.predict.v2")
except NotImplementedError:
    pass
assert not openbabel_loaded()
from xenosite.predict.registry import ensure_builtins
ensure_builtins()
assert not openbabel_loaded()
xp.list_models(env={})
assert not openbabel_loaded()
from xenosite.predict.v1.features import _ob
assert _ob._CACHE is None
assert not openbabel_loaded()
"""
    proc = _run(code)
    assert proc.returncode == 0, proc.stderr or proc.stdout


def test_http_predict_does_not_load_openbabel():
    code = r"""
import httpx
import msgpack
from xenosite.predict import predict
from xenosite.predict.backends.http import HttpBackend

payload = {
    "smiles": "CCO",
    "name": {},
    "atoms": {"num": 3, "z": [6, 6, 8], "reordered": [2, 1, 0]},
    "bonds": {"idx": [[0, 1], [1, 2]], "order": [1.0, 1.0]},
    "results": [
        {"model": "epoxidation", "model_version": "1", "mol": 0.1, "bond": [0.01, 0.02]}
    ],
}

def handler(request):
    return httpx.Response(
        200,
        content=msgpack.packb(payload),
        headers={"content-type": "application/x-msgpack"},
    )

be = HttpBackend(
    "https://example.test",
    transport=httpx.MockTransport(handler),
    retry_max=1,
)
assert not openbabel_loaded()
predict("CCO", model="epoxidation", backend=be)
assert not openbabel_loaded()
from xenosite.predict.v1.features import _ob
assert _ob._CACHE is None
"""
    proc = _run(code)
    assert proc.returncode == 0, proc.stderr or proc.stdout


@pytest.mark.skipif(not onnx_weights_present("epoxidation"), reason="epoxidation ONNX missing")
def test_list_models_onnx_does_not_load_openbabel():
    code = r"""
from tests.support import onnx_root
from xenosite.predict import list_models
from xenosite.predict.backends.onnx import OnnxBackend

assert not openbabel_loaded()
list_models(backend=OnnxBackend(onnx_root()))
assert not openbabel_loaded()
from xenosite.predict.v1.features import _ob
assert _ob._CACHE is None
"""
    proc = _run(code)
    assert proc.returncode == 0, proc.stderr or proc.stdout


@pytest.mark.skipif(not onnx_weights_present("epoxidation"), reason="epoxidation ONNX missing")
def test_onnx_predict_loads_openbabel():
    code = r"""
from tests.support import onnx_root
from xenosite.predict import predict
from xenosite.predict.backends.onnx import OnnxBackend

assert not openbabel_loaded()
predict("CCO", model="epoxidation", backend=OnnxBackend(onnx_root()))
assert openbabel_loaded()
from xenosite.predict.v1.features import _ob
assert _ob._CACHE is not None
"""
    proc = _run(code)
    assert proc.returncode == 0, proc.stderr or proc.stdout
