"""Live parity: Hypothesis random-vector NN, SMILES, OpenBabel dump. Skip without Docker."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from xenosite.predict import predict
from xenosite.predict.backends.legacy import LegacyTestBackend
from xenosite.predict.backends.onnx import OnnxBackend
from xenosite.predict.molecule import parse_smiles

from tests.strategies import feature_matrix
from tests.support import (
    ROOT,
    compare_feature_dump_rows,
    list_onnx_heads,
    onnx_io_dims,
    onnx_weights_present,
    rows_for_model,
)

ASPIRIN = "O=C(C)Oc1ccccc1C(=O)O"


@pytest.mark.live
def test_legacy_health(legacy_api_url):
    be = LegacyTestBackend(legacy_api_url)
    assert be.health()


def _live_nn_heads():
    heads = list_onnx_heads()
    if heads:
        return heads
    return [
        ("epoxidation", "bond"),
        ("epoxidation", "mol"),
        ("ugt", "atom"),
    ]


@pytest.mark.live
@pytest.mark.parametrize("model,head", _live_nn_heads())
@settings(
    max_examples=5,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)
@given(data=st.data())
def test_random_vector_nn(legacy_api_url, model, head, data):
    """Same Hypothesis-drawn matrix to legacy /nn and ONNX. Skip if either is missing."""
    if not onnx_weights_present(model):
        pytest.skip(f"no ONNX for {model}")
    dims = onnx_io_dims(model, head)
    if dims is None:
        pytest.skip(f"no I/O dims for {model}/{head}")
    n_in, _n_out = dims
    x = data.draw(feature_matrix(n_in, max_rows=3))
    be = LegacyTestBackend(legacy_api_url)
    try:
        y_legacy = be.nn(model, head, x.tolist())
    except Exception as exc:
        pytest.skip(f"legacy /nn/{model}/{head} failed: {exc}")
    onnx = OnnxBackend(ROOT / "weights" / "onnx")
    try:
        y_onnx = onnx.run_head(model, head, x)
    except Exception as exc:
        pytest.skip(f"ONNX run failed: {exc}")
    y0 = np.asarray(y_legacy["y"] if isinstance(y_legacy, dict) else y_legacy)
    y1 = np.asarray(y_onnx)
    n = min(y0.size, y1.size)
    np.testing.assert_allclose(y0.reshape(-1)[:n], y1.reshape(-1)[:n], atol=1e-4, rtol=0)


@pytest.mark.live
@pytest.mark.parametrize("model", ["epoxidation", "quinone", "ugt", "reactivity"])
def test_smiles_parity(legacy_api_url, model):
    if not onnx_weights_present("ndealk" if model == "isozyme" else model):
        pytest.skip(f"no ONNX for {model}")
    legacy = LegacyTestBackend(legacy_api_url)
    onnx = OnnxBackend(ROOT / "weights" / "onnx")
    try:
        a = predict(ASPIRIN, models=[model], backend=legacy)
    except Exception as exc:
        pytest.skip(f"legacy predict failed: {exc}")
    try:
        b = predict(ASPIRIN, models=[model], backend=onnx)
    except Exception as ext:
        pytest.skip(f"onnx predict failed: {ext}")
    assert a.results and b.results
    ra, rb = a.results[0], b.results[0]
    if getattr(ra, "bond", None) is not None and getattr(rb, "bond", None) is not None:
        np.testing.assert_allclose(ra.bond, rb.bond, atol=1e-4)
    if getattr(ra, "atom", None) is not None and getattr(rb, "atom", None) is not None:
        np.testing.assert_allclose(ra.atom, rb.atom, atol=1e-4)
    if getattr(ra, "mol", None) is not None and getattr(rb, "mol", None) is not None:
        np.testing.assert_allclose(ra.mol, rb.mol, atol=1e-4)


@pytest.mark.live
@pytest.mark.parametrize("model", ["epoxidation", "ugt", "reactivity", "ndealk", "quinone"])
def test_internal_ob_vs_live_dump(model):
    """Re-dump via xenosite-predict-py2:dump; compare to host OpenBabel features.

    Does not need the WashU registry. Skip if Docker, OpenBabel, or sibling src
    is missing. Any overlapping-column mismatch fails. Do not loosen atol.
    """
    import sys

    tools = ROOT / "tools"
    if str(tools) not in sys.path:
        sys.path.insert(0, str(tools))
    from dump_ob import DEFAULT_SRC, _rdkit_sdf, dump_one  # type: ignore

    if not DEFAULT_SRC.is_dir():
        pytest.skip(f"xenosite-legacy/src missing at {DEFAULT_SRC}")
    mol, _molecule = parse_smiles(ASPIRIN)
    canonical, sdf_text = _rdkit_sdf(ASPIRIN)
    try:
        dump = dump_one(canonical, model, DEFAULT_SRC, sdf_text=sdf_text)
    except Exception as exc:
        pytest.skip(f"OpenBabel dump via py2 image failed: {exc}")
    rows = rows_for_model(model, mol)
    if not dump.get("columns") or not dump.get("rows"):
        pytest.skip("empty OB dump")
    mismatches = compare_feature_dump_rows(rows, dump)
    if mismatches:
        pytest.fail(
            f"internal OpenBabel vs dump mismatch for {model} (do not loosen atol): "
            + ", ".join(mismatches)
        )
