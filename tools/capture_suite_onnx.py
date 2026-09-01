#!/usr/bin/env python3
"""Incremental ONNX score cache for the golden suite (327×models).

Writes ``tests/fixtures/suite_onnx_cache.json``. Skips pairs already present;
use ``--force`` to refresh one model or SMILES. Run once, then use
``tools/report_suite_drift.py`` for instant classification.

Example::

  uv run python tools/capture_suite_onnx.py              # fill missing only
  uv run python tools/capture_suite_onnx.py --model reactivity --force
  uv run python tools/report_suite_drift.py              # analyze from cache
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.support import (  # noqa: E402
    SUITE_MODELS,
    load_golden_suite,
    serialize_molecule_results,
)
from tools.suite_drift_lib import DEFAULT_CACHE, cache_index, load_cache, save_cache  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    p.add_argument(
        "--models",
        default=",".join(m for m in SUITE_MODELS if m != "phase1"),
        help="comma-separated models",
    )
    p.add_argument("--smiles", default="", help="single SMILES only")
    p.add_argument("--model", default="", help="single model only")
    p.add_argument("--force", action="store_true", help="redo cached pairs")
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args(argv)

    from xenosite.predict import predict
    from xenosite.predict.backends.onnx import OnnxBackend

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    if args.model:
        models = [args.model]

    golden = load_golden_suite(merge_smoke=True)
    pairs: list[tuple[str, str, dict]] = []
    seen: set[tuple[str, str]] = set()
    for g in golden:
        model, smiles = g.get("model") or "", g.get("smiles") or ""
        if model not in models or not smiles:
            continue
        key = (model, smiles)
        if key in seen:
            continue
        seen.add(key)
        if args.smiles and smiles != args.smiles:
            continue
        pairs.append((model, smiles, g))
    if args.limit:
        pairs = pairs[: args.limit]

    rows = load_cache(args.cache)
    done = cache_index(rows)
    added = 0
    backend = OnnxBackend(ROOT / "weights" / "onnx")

    for model, smiles, g in pairs:
        key = (model, smiles)
        if key in done and not args.force:
            continue
        if args.force:
            rows = [r for r in rows if (r.get("model"), r.get("smiles")) != key]
        t0 = time.time()
        rec: dict = {
            "model": model,
            "smiles": smiles,
            "name": g.get("name") or smiles[:40],
        }
        try:
            mol = predict(smiles, models=[model], backend=backend)
            rec["results"] = serialize_molecule_results(mol)
            rec["error"] = None
        except Exception as exc:
            rec["results"] = []
            rec["error"] = str(exc)
        rows.append(rec)
        added += 1
        save_cache(args.cache, rows)
        dt = time.time() - t0
        status = "ok" if not rec["error"] else f"ERR {rec['error'][:40]}"
        print(f"[{added}] {model} {rec['name'][:32]} ({dt:.1f}s) {status}", flush=True)

    print(f"cache {args.cache} total={len(rows)} new={added}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
