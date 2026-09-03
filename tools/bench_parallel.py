"""Benchmark predict_many vs sequential predict (ugt stub ONNX)."""

from __future__ import annotations

import os
import shutil
import time
from pathlib import Path

from xenosite.predict import predict, predict_many
from xenosite.predict.backends.onnx import OnnxBackend
from xenosite.predict.parallel import reset_pools_for_tests

SMILES = [
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


def main() -> None:
    reset_pools_for_tests()
    root = Path("/tmp/ugt_stub_onnx_bench")
    if root.exists():
        shutil.rmtree(root)
    shutil.copytree(Path(__file__).resolve().parents[1] / "tests/fixtures/stub_onnx", root)
    be = OnnxBackend(root)
    smiles = SMILES * 3
    workers = min(4, os.cpu_count() or 1)

    predict_many(smiles[:workers], model="ugt", backend=be, workers=workers)
    for s in smiles[:2]:
        predict(s, model="ugt", backend=be)

    t0 = time.perf_counter()
    for s in smiles:
        predict(s, model="ugt", backend=be)
    t_seq = time.perf_counter() - t0

    t0 = time.perf_counter()
    predict_many(smiles, model="ugt", backend=be, workers=workers)
    t_par = time.perf_counter() - t0

    print("xenosite.predict parallel throughput")
    print(f"n={len(smiles)} workers={workers} cpus={os.cpu_count()} model=ugt (stub ONNX)")
    print(f"sequential={t_seq:.3f}s ({len(smiles) / t_seq:.1f} mol/s)")
    print(f"predict_many={t_par:.3f}s ({len(smiles) / t_par:.1f} mol/s)")
    print(f"speedup={t_seq / t_par:.2f}x")


if __name__ == "__main__":
    main()
