"""Live XenoNet graphs vs legacy-test-api POST /xenonet."""

from __future__ import annotations

import pytest
import httpx

from xenosite.predict.backends.onnx import OnnxBackend
from xenosite.predict.features import _ob
from xenosite.predict.v1.xenonet import build_network

from tests.support import PARITY_ATOL, onnx_root, onnx_weights_present

pytestmark = pytest.mark.live


def test_legacy_xenonet_depth1_ethane(legacy_api_url):
    if not _ob.installed():
        pytest.skip("OpenBabel not installed")
    root = onnx_root()
    if not onnx_weights_present("phase1"):
        from xenosite.predict.weights import default_cache_dir

        cache = default_cache_dir()
        if cache is None or not any((cache / "phase1").glob("*.onnx")):
            pytest.skip("phase1 ONNX missing")
        root = cache
    payload = {
        "smiles": "CC",
        "depth_limit": 1,
        "beam_width": 1000,
        "max_time": 5,
        "weighted": True,
    }
    with httpx.Client(timeout=180.0) as client:
        r = client.post(f"{legacy_api_url}/xenonet", json=payload)
    if r.status_code == 501:
        pytest.skip(r.json().get("error") or "xenonet unavailable on legacy-test-api")
    r.raise_for_status()
    expect = r.json()
    if expect.get("error"):
        pytest.skip(str(expect["error"]))
    got = build_network(
        "CC",
        depth_limit=1,
        beam_width=1000,
        backend=OnnxBackend(root),
        scoring="0",
    ).to_dict()
    ge = {(e["parent"], e["child"], e["rule"], tuple(e["site"])) for e in got["edges"]}
    ee = {(e["parent"], e["child"], e["rule"], tuple(e["site"])) for e in expect["edges"]}
    assert ge == ee
    w_got = {
        (e["parent"], e["child"], e["rule"], tuple(e["site"])): e["weight"]
        for e in got["edges"]
    }
    for e in expect["edges"]:
        key = (e["parent"], e["child"], e["rule"], tuple(e["site"]))
        assert abs(w_got[key] - e["weight"]) <= PARITY_ATOL
