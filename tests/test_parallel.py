"""Parallel / async predict helpers: correctness and throughput."""

from __future__ import annotations

import asyncio
import os
import shutil
import time
from pathlib import Path

import pytest

from xenosite.predict import apredict, apredict_many, predict, predict_many
from xenosite.predict.backends.onnx import OnnxBackend
from xenosite.predict.features import _ob
from xenosite.predict.parallel import default_workers, reset_pools_for_tests

from tests.support import ROOT

STUB_ONNX = ROOT / "tests" / "fixtures" / "stub_onnx"

# Diverse SMILES so descriptor work is non-trivial.
BATCH_SMILES = [
    "CC(=O)Oc1ccccc1C(=O)O",
    "CCO",
    "c1ccccc1",
    "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
    "CC(=O)NC1=CC=C(C=C1)O",
    "C1=CC=C(C=C1)C(=O)O",
    "CC1=CC=CC=C1C(=O)O",
    "CCCCCCCCCCCCCCCC(=O)O",
    "CC(C)NCC(O)c1ccc(O)c(CO)c1",
    "Clc1ccc(cc1)C(=O)Nc2ccc(Cl)c(Cl)c2",
    "O=C1NC(=O)C(c2ccccc2)(c2ccccc2)N1",
    "CCN(CC)CC",
    "Nc1ncnc2n(cnc12)C3OC(CO)C(O)C3O",
    "CC(=O)OCCN(C)C",
    "Fc1ccc(cc1)C(O)(C(F)(F)F)C(F)(F)F",
]


@pytest.fixture
def ugt_onnx_root(tmp_path):
    if not _ob.installed():
        pytest.skip("OpenBabel not installed")
    if not (STUB_ONNX / "ugt" / "atom.onnx").is_file():
        pytest.skip("stub ONNX fixture missing")
    reset_pools_for_tests()
    root = tmp_path / "onnx"
    shutil.copytree(STUB_ONNX, root)
    yield root
    reset_pools_for_tests()


def test_predict_many_matches_sequential(ugt_onnx_root):
    be = OnnxBackend(ugt_onnx_root)
    smiles = BATCH_SMILES[:6]
    sequential = [predict(s, model="ugt", backend=be) for s in smiles]
    parallel = predict_many(smiles, model="ugt", backend=be, workers=2)
    assert len(parallel) == len(sequential)
    for a, b in zip(sequential, parallel):
        assert a.smiles == b.smiles
        assert len(a.results) == len(b.results) == 1
        assert a.results[0].atom == pytest.approx(b.results[0].atom)


def test_apredict_matches_predict(ugt_onnx_root):
    be = OnnxBackend(ugt_onnx_root)
    smi = BATCH_SMILES[0]
    sync = predict(smi, model="ugt", backend=be)

    async def _run():
        return await apredict(smi, model="ugt", backend=be, workers=2)

    async_mol = asyncio.run(_run())
    assert async_mol.smiles == sync.smiles
    assert async_mol.results[0].atom == pytest.approx(sync.results[0].atom)


def test_apredict_many_matches_predict_many(ugt_onnx_root):
    be = OnnxBackend(ugt_onnx_root)
    smiles = BATCH_SMILES[:8]
    sync = predict_many(smiles, model="ugt", backend=be, workers=2)

    async def _run():
        return await apredict_many(smiles, model="ugt", backend=be, workers=2)

    async_out = asyncio.run(_run())
    assert [m.smiles for m in async_out] == [m.smiles for m in sync]
    for a, b in zip(sync, async_out):
        assert a.results[0].atom == pytest.approx(b.results[0].atom)


def test_apredict_gather_concurrent(ugt_onnx_root):
    be = OnnxBackend(ugt_onnx_root)
    smiles = BATCH_SMILES[:8]

    async def _run():
        return await asyncio.gather(
            *[apredict(s, model="ugt", backend=be, workers=4) for s in smiles]
        )

    out = asyncio.run(_run())
    assert len(out) == len(smiles)
    assert all(m.results for m in out)


def test_predict_many_throughput_beats_sequential(ugt_onnx_root):
    """Process workers must improve wall-clock throughput on multi-core hosts."""
    cpus = os.cpu_count() or 1
    if cpus < 2:
        pytest.skip("need at least 2 CPUs to measure parallel speedup")

    be = OnnxBackend(ugt_onnx_root)
    # Repeat the batch so wall time dwarfs process-pool startup.
    smiles = BATCH_SMILES * 3
    workers = min(4, cpus)

    # Warm workers / sessions so the timed region is steady-state.
    predict_many(smiles[:workers], model="ugt", backend=be, workers=workers)
    for s in smiles[:2]:
        predict(s, model="ugt", backend=be)

    t0 = time.perf_counter()
    sequential = [predict(s, model="ugt", backend=be) for s in smiles]
    t_seq = time.perf_counter() - t0

    t0 = time.perf_counter()
    parallel = predict_many(smiles, model="ugt", backend=be, workers=workers)
    t_par = time.perf_counter() - t0

    assert len(parallel) == len(sequential)
    speedup = t_seq / t_par if t_par > 0 else 0.0
    # Require a clear win; on 2–4 cores descriptor work should scale well.
    assert speedup >= 1.3, (
        f"expected predict_many speedup>=1.3, got {speedup:.2f}x "
        f"(seq={t_seq:.3f}s par={t_par:.3f}s workers={workers} cpus={cpus})"
    )


def test_default_workers_respects_env(monkeypatch):
    monkeypatch.setenv("XENOSITE_WORKERS", "3")
    assert default_workers() == 3
