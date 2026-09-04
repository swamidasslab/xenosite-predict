"""Benchmark ``predict_many`` wall time: local ONNX vs HTTP backend.

Usage::

    uv run python tools/bench_http_vs_onnx.py
    XENOSITE_BACKEND=https://swami.wustl.edu/xenosite-api \\
      uv run python tools/bench_http_vs_onnx.py --n 32 --model ugt

Skips the HTTP side cleanly when the origin is unreachable.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import httpx

from xenosite.predict import predict_many
from xenosite.predict.backends.http import HttpBackend
from xenosite.predict.backends.onnx import OnnxBackend
from xenosite.predict.parallel import reset_pools_for_tests
from xenosite.predict.weights import generation_root

DEFAULT_HTTP = "https://swami.wustl.edu/xenosite-api"
ROOT = Path(__file__).resolve().parents[1]

SMILES = [
    "CC(=O)Oc1ccccc1C(=O)O",
    "CCO",
    "c1ccccc1",
    "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
    "CC(=O)NC1=CC=C(C=C1)O",
    "C1=CC=C(C=C1)C(=O)O",
    "CC1=CC=CC=C1C(=O)O",
    "CC(C)NCC(O)c1ccc(O)c(CO)c1",
    "Nc1ncnc2n(cnc12)C3OC(CO)C(O)C3O",
]


def _onnx_root() -> Path:
    return generation_root(ROOT / "weights" / "onnx")


def _onnx_weights_present(model: str) -> bool:
    return any((_onnx_root() / model).glob("*.onnx"))


def _bench(label: str, fn) -> float:
    t0 = time.perf_counter()
    fn()
    return time.perf_counter() - t0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=20, help="number of molecules")
    parser.add_argument("--model", default="ugt")
    parser.add_argument(
        "--http",
        default=os.environ.get("XENOSITE_BACKEND", DEFAULT_HTTP),
        help="xenosite-api origin",
    )
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()

    reset_pools_for_tests()
    smiles = (SMILES * ((args.n // len(SMILES)) + 1))[: args.n]
    print(f"n={len(smiles)} model={args.model}")

    root = _onnx_root()
    if _onnx_weights_present(args.model):
        be = OnnxBackend(root)
        # warmup
        predict_many(smiles[:2], model=args.model, backend=be, workers=1)
        dt = _bench(
            "onnx",
            lambda: predict_many(
                smiles, model=args.model, backend=be, workers=args.workers
            ),
        )
        print(f"ONNX  predict_many: {dt:.3f}s ({len(smiles) / dt:.1f} mol/s)")
    else:
        print(f"ONNX  skipped (no weights for {args.model} under {root})")

    url = (args.http or "").rstrip("/")
    if not url:
        print("HTTP  skipped (no XENOSITE_BACKEND / --http)")
        return
    try:
        r = httpx.get(f"{url}/v1/canonize", params={"smiles": "CCO"}, timeout=10.0)
        r.raise_for_status()
    except Exception as exc:
        print(f"HTTP  skipped (unreachable {url}: {exc})")
        return

    env = dict(os.environ)
    be_h = HttpBackend(url, api_key=env.get("XENOSITE_API_KEY"), env=env)
    predict_many(smiles[:2], model=args.model, backend=be_h)
    dt = _bench(
        "http",
        lambda: predict_many(smiles, model=args.model, backend=be_h),
    )
    print(f"HTTP  predict_many: {dt:.3f}s ({len(smiles) / dt:.1f} mol/s) origin={url}")


if __name__ == "__main__":
    main()
