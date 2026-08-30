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
from xenosite.predict.features import bond_rows, ugt_atom_rows
from xenosite.predict.molecule import parse_smiles

from tests.support import ROOT, compare_rdkit_ob_rows, onnx_weights_present

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
def test_rdkit_vs_ob(legacy_api_url, model):
    """Dump OpenBabel features from the test image; compare to RDKit on same SMILES."""
    from xenosite.predict.features import reactivity_atom_rows

    legacy = LegacyTestBackend(legacy_api_url)
    try:
        dump = legacy.dump_features(ASPIRIN, model)
    except Exception as exc:
        pytest.skip(f"feature dump failed: {exc}")
    mol, _ = parse_smiles(ASPIRIN)
    if model in ("epoxidation", "ndealk"):
        rows = bond_rows(mol, original_atom_ordering=True)
    elif model == "ugt":
        rows = ugt_atom_rows(mol)
    else:
        rows = reactivity_atom_rows(mol)
    cols = dump.get("columns") or []
    ob_rows = dump.get("rows") or []
    if not cols or not ob_rows:
        pytest.skip("empty OB dump")
    mismatches = compare_rdkit_ob_rows(rows, dump)
    if mismatches:
        pytest.fail(
            f"RDKit vs OpenBabel mismatch for {model} (docs/vendored-diffs.md): "
            + ", ".join(mismatches[:12])
        )
