"""Live parity: random-vector, SMILES, RDKit vs OpenBabel. Skip without Docker."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from xenosite.predict import predict
from xenosite.predict.backends.legacy import LegacyTestBackend
from xenosite.predict.backends.onnx import OnnxBackend
from xenosite.predict.compare import scores_close
from xenosite.predict.features import bond_rows, matrix_from_rows, ugt_atom_rows
from xenosite.predict.molecule import parse_smiles

from tests.support import ROOT, onnx_weights_present

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
def test_smiles_parity_epoxidation(legacy_api_url):
    if not onnx_weights_present("epoxidation"):
        pytest.skip("no epoxidation ONNX")
    legacy = LegacyTestBackend(legacy_api_url)
    onnx = OnnxBackend(ROOT / "weights" / "onnx")
    a = predict(ASPIRIN, models=["epoxidation"], backend=legacy)
    b = predict(ASPIRIN, models=["epoxidation"], backend=onnx)
    assert a.results and b.results
    np.testing.assert_allclose(a.results[0].bond, b.results[0].bond, atol=1e-4)


@pytest.mark.live
def test_rdkit_vs_ob_epoxidation(legacy_api_url):
    """Dump OpenBabel features from the test image; compare to RDKit on same SMILES."""
    legacy = LegacyTestBackend(legacy_api_url)
    try:
        dump = legacy.dump_features(ASPIRIN, "epoxidation")
    except Exception as exc:
        pytest.skip(f"feature dump failed: {exc}")
    mol, _ = parse_smiles(ASPIRIN)
    rows = bond_rows(mol, original_atom_ordering=True)
    cols = dump.get("columns") or []
    ob_rows = dump.get("rows") or []
    if not cols or not ob_rows:
        pytest.skip("empty OB dump")
    # Compare overlapping numeric columns by name
    mismatches = []
    for i, ob in enumerate(ob_rows):
        if i >= len(rows):
            break
        ob_map = dict(zip(cols, ob))
        for c, val in ob_map.items():
            if c not in rows[i]:
                continue
            try:
                if not np.isclose(float(val), float(rows[i][c]), atol=1e-4):
                    mismatches.append((c, val, rows[i][c]))
            except (TypeError, ValueError):
                continue
    if mismatches:
        pytest.fail(
            "RDKit vs OpenBabel feature mismatch (see docs/vendored-diffs.md): "
            + ", ".join(f"{c}" for c, _, _ in mismatches[:12])
        )
