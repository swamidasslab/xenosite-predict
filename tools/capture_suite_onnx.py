#!/usr/bin/env python3
"""Incremental ONNX score cache for the golden suite (327×models).

Writes ``tests/fixtures/suite_onnx_cache.json``. Skips pairs already present;
use ``--force`` to refresh one model or SMILES. Run once, then use
``tools/report_suite_drift.py`` for instant classification.

Example::

  uv run python tools/capture_suite_onnx.py              # fill missing only
  uv run python tools/capture_suite_onnx.py --workers 8
  uv run python tools/capture_suite_onnx.py --model reactivity --force
  uv run python tools/report_suite_drift.py              # analyze from cache
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.support import (  # noqa: E402
    GOLDEN_PARAMETER,
    SUITE_MODELS,
    load_golden_suite,
    serialize_molecule_results,
)
from tools.progress import iter_progress, map_progress, worker_quiet  # noqa: E402
from tools.suite_drift_lib import DEFAULT_CACHE, cache_index, default_workers, load_cache, save_cache  # noqa: E402

_WEIGHTS_ROOT: str = ""


def _worker_init(weights_root: str) -> None:
    worker_quiet()
    global _WEIGHTS_ROOT
    _WEIGHTS_ROOT = weights_root


def _capture_one(task: tuple[str, str, str]) -> dict:
    """Run one (model, smiles, name) in a worker process."""
    model, smiles, name = task
    from xenosite.predict import predict
    from xenosite.predict.backends.onnx import OnnxBackend

    rec: dict = {
        "model": model,
        "smiles": smiles,
        "name": name or smiles[:40],
    }
    kwargs = {}
    if model in ("ndealk", "isozyme", "quinone"):
        kwargs["_parameter"] = dict(GOLDEN_PARAMETER)
    try:
        backend = OnnxBackend(_WEIGHTS_ROOT)
        mol = predict(smiles, models=[model], backend=backend, **kwargs)
        rec["results"] = serialize_molecule_results(mol)
        rec["error"] = None
    except Exception as exc:
        rec["results"] = []
        rec["error"] = str(exc)
    return rec


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
    p.add_argument(
        "--workers",
        type=int,
        default=default_workers(),
        help="parallel worker processes (default: min(cpu_count, 24) or XENOSITE_WORKERS)",
    )
    args = p.parse_args(argv)

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
    pending: list[tuple[str, str, str]] = []
    for model, smiles, g in pairs:
        key = (model, smiles)
        if key in done and not args.force:
            continue
        if args.force:
            rows = [r for r in rows if (r.get("model"), r.get("smiles")) != key]
            done.pop(key, None)
        name = g.get("name") or smiles[:40]
        pending.append((model, smiles, name))

    if not pending:
        print(f"cache {args.cache} total={len(rows)} new=0 (nothing pending)")
        return 0

    weights_root = str(ROOT / "weights" / "onnx")
    workers = max(1, args.workers)
    added = 0
    t0 = time.time()

    if workers == 1:
        _worker_init(weights_root)
        for task in iter_progress(pending, desc="capture ONNX", unit="pair"):
            rec = _capture_one(task)
            rows.append(rec)
            added += 1
            save_cache(args.cache, rows)
    else:
        def _on_capture(rec: dict, bar) -> None:
            if rec.get("error"):
                bar.write(
                    f"ERR {rec['model']} {rec['name'][:32]}: {rec['error'][:60]}"
                )

        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_worker_init,
            initargs=(weights_root,),
        ) as pool:
            for rec in map_progress(
                _capture_one,
                pending,
                pool=pool,
                desc=f"capture ONNX ({workers}w, {len(pending)} pairs)",
                unit="pair",
                on_result=_on_capture,
            ):
                rows.append(rec)
                added += 1
                save_cache(args.cache, rows)

    dt = time.time() - t0
    print(f"cache {args.cache} total={len(rows)} new={added} ({dt:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
