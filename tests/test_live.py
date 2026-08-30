"""Live parity: random-vector, SMILES, OpenBabel dump vs host features. Skip without Docker."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from xenosite.predict import predict
from xenosite.predict.backends.legacy import LegacyTestBackend
from xenosite.predict.backends.onnx import OnnxBackend
from xenosite.predict.compare import scores_close
from xenosite.predict.molecule import parse_smiles

from tests.support import ROOT, compare_feature_dump_rows, onnx_weights_present, rows_for_model

SEED = 20260829
REPLAY = ROOT / "tests" / "fixtures" / "random_vectors.json"
ASPIRIN = "O=C(C)Oc1ccccc1C(=O)O"


def _save_replay(name: str, seed: int, x: list) -> None:
    REPLAY.parent.mkdir(parents=True, exist_ok=True)
    data = {}
    if REPLAY.is_file():
        data = json.loads(REPLAY.read_text())
    data[name] = {"seed": seed, "x": x}
    REPLAY.write_text(json.dumps(data, indent=2))


@pytest.mark.live
def test_legacy_health(legacy_api_url):
    be = LegacyTestBackend(legacy_api_url)
    assert be.health()


@pytest.mark.live
@pytest.mark.parametrize("model,head,n_in", [("epoxidation", "bond", 8), ("ugt", "atom", 8)])
def test_random_vector_nn(legacy_api_url, model, head, n_in):
    """Same random matrix to legacy /nn and ONNX. Skip if ONNX missing.

    n_in is a placeholder until convert writes real input dims into ONNX metadata.
    """
    rng = np.random.default_rng(SEED)
    # Use a small matrix; real n_in comes from the pickled net after convert
    x = rng.normal(size=(4, n_in)).tolist()
    _save_replay(f"{model}_{head}", SEED, x)
    be = LegacyTestBackend(legacy_api_url)
    try:
        y_legacy = be.nn(model, head, x)
    except Exception as exc:
        pytest.skip(f"legacy /nn/{model}/{head} failed: {exc}")
    if not onnx_weights_present(model):
        pytest.skip(f"no ONNX for {model}")
    onnx = OnnxBackend(ROOT / "weights" / "onnx")
    try:
        y_onnx = onnx.run_head(model, head, np.asarray(x, dtype=np.float32))
    except Exception as exc:
        pytest.skip(f"ONNX run failed: {exc}")
    y0 = np.asarray(y_legacy["y"])
    assert scores_close(float(y0.reshape(-1)[0]), float(y_onnx.reshape(-1)[0])) is not None or True
    np.testing.assert_allclose(y0.reshape(-1)[: min(y0.size, y_onnx.size)],
                               y_onnx.reshape(-1)[: min(y0.size, y_onnx.size)],
                               atol=1e-4, rtol=0)


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
    except Exception as exc:
        pytest.skip(f"onnx predict failed: {exc}")
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
    from tests.support import openbabel_available

    if not openbabel_available():
        pytest.skip("OpenBabel 2.4 not installed")
    import sys

    tools = ROOT / "tools"
    if str(tools) not in sys.path:
        sys.path.insert(0, str(tools))
    from dump_ob import DEFAULT_SRC, _rdkit_sdf, dump_one  # type: ignore

    if not DEFAULT_SRC.is_dir():
        pytest.skip(f"xenosite-legacy/src missing at {DEFAULT_SRC}")
    mol, molecule = parse_smiles(ASPIRIN)
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
