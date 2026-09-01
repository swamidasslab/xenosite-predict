#!/usr/bin/env python3
"""Bulk-capture legacy prediction scores for the 327-molecule descriptor suite.

Writes ``tests/fixtures/golden_descriptor_suite.json`` incrementally (skips pairs
already present; use ``--force`` to redo one model or SMILES).

Uses ``legacy-test-api`` (``POST /predict/<model>``) via :class:`LegacyTestBackend`.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.support import (  # noqa: E402
    GOLDEN_SUITE,
    SUITE_MODELS,
    load_descriptor_smiles,
    load_golden_suite,
    serialize_molecule_results,
)
from tools.progress import iter_progress, map_progress, worker_quiet  # noqa: E402
from tools.suite_drift_lib import DEFAULT_CACHE, FAILING_SMILES_JSON, analyze_suite, default_workers, load_cache  # noqa: E402

_LEGACY_URL: str = ""
_NAMES: dict[str, str] = {}


def _load_out(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def _save_out(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")


def _name_for(smiles: str) -> str:
    return _NAMES.get(smiles) or smiles[:40]


def _worker_init(url: str, names: dict[str, str]) -> None:
    worker_quiet()
    global _LEGACY_URL, _NAMES
    _LEGACY_URL = url
    _NAMES = names


def _gather_one(task: tuple[str, str]) -> dict:
    """Run one (smiles, model) pair in a worker process."""
    smiles, model = task
    from xenosite.predict import predict
    from xenosite.predict.backends.legacy import LegacyTestBackend

    rec: dict = {
        "smiles": smiles,
        "name": _name_for(smiles),
        "model": model,
        "error": None,
        "results": [],
    }
    try:
        mol = predict(smiles, models=[model], backend=LegacyTestBackend(_LEGACY_URL))
        rec["smiles"] = mol.smiles
        rec["results"] = serialize_molecule_results(mol)
    except Exception as exc:
        rec["error"] = str(exc)
    return rec


def _failing_smiles(models: list[str] | None = None) -> list[str]:
    model_set = set(models) if models else None
    if FAILING_SMILES_JSON.is_file() and not model_set:
        return json.loads(FAILING_SMILES_JSON.read_text(encoding="utf-8"))
    golden = load_golden_suite(merge_smoke=True)
    cache = load_cache(DEFAULT_CACHE)
    suite_models = list(model_set) if model_set else [
        m for m in SUITE_MODELS if m not in ("phase1", "bioactivation")
    ]
    report = analyze_suite(golden, cache, models=suite_models, workers=default_workers())
    if model_set:
        smiles = {ff.smiles for ff in report.field_failures if ff.model in model_set}
        return sorted(smiles)
    return sorted(report.unique_failing_smiles)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--url",
        default="http://127.0.0.1:8099",
        help="legacy-test-api base URL",
    )
    p.add_argument("--out", type=Path, default=GOLDEN_SUITE)
    p.add_argument(
        "--models",
        default=",".join(SUITE_MODELS),
        help="comma-separated model names",
    )
    p.add_argument("--limit", type=int, default=0, help="max SMILES (0 = all)")
    p.add_argument("--smiles", default="", help="single SMILES only")
    p.add_argument("--force", action="store_true", help="redo even if present")
    p.add_argument(
        "--failing-only",
        action="store_true",
        help="regather SMILES that fail drift vs ONNX cache (implies --force)",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=default_workers(),
        help="parallel worker processes (default: min(cpu_count, 24) or XENOSITE_WORKERS)",
    )
    p.add_argument(
        "--save-every",
        type=int,
        default=50,
        help="flush golden JSON every N successful pairs (0 = end only)",
    )
    p.add_argument(
        "--merge-smoke",
        action="store_true",
        help="also copy rows from golden_smiles.json at the end",
    )
    args = p.parse_args(argv)

    from xenosite.predict.backends.legacy import LegacyTestBackend

    try:
        from tests.support import golden_name_by_smiles

        names = golden_name_by_smiles()
    except Exception:
        names = {}

    if args.failing_only:
        args.force = True

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    if args.smiles:
        smiles_list = [args.smiles]
    elif args.failing_only:
        model_filter = models if set(models) != set(SUITE_MODELS) else None
        smiles_list = _failing_smiles(model_filter)
        if not smiles_list:
            print("no failing SMILES from drift cache", file=sys.stderr)
            return 0
        label = f"regather {len(smiles_list)} failing SMILES"
        if model_filter:
            label += f" ({','.join(model_filter)})"
        print(label, flush=True)
    else:
        smiles_list = load_descriptor_smiles()
    if args.limit:
        smiles_list = smiles_list[: args.limit]

    backend = LegacyTestBackend(args.url)
    if not backend.health():
        print(f"legacy-test-api not healthy at {args.url}", file=sys.stderr)
        return 1

    rows = _load_out(args.out)
    done = {(r.get("model"), r.get("smiles")) for r in rows}
    pending: list[tuple[str, str]] = []
    for smiles in smiles_list:
        for model in models:
            key = (model, smiles)
            if key in done and not args.force:
                continue
            pending.append((smiles, model))

    if not pending:
        print(f"{args.out} total={len(rows)} new=0 (nothing pending)")
        return 0

    if args.force:
        pending_keys = {(m, s) for s, m in pending}
        rows = [r for r in rows if (r.get("model"), r.get("smiles")) not in pending_keys]
        done = {(r.get("model"), r.get("smiles")) for r in rows}

    workers = max(1, args.workers)
    save_every = max(0, int(args.save_every))
    added = 0
    errors: list[str] = []
    t0 = time.time()

    def _maybe_save() -> None:
        if save_every and added % save_every == 0:
            _save_out(args.out, rows)

    def _apply(rec: dict) -> None:
        nonlocal added
        key = (rec["model"], rec["smiles"])
        if rec.get("error"):
            errors.append(f"{rec['model']} {rec['smiles'][:32]}: {rec['error']}")
            return
        rows.append(
            {
                "smiles": rec["smiles"],
                "name": rec["name"],
                "model": rec["model"],
                "results": rec["results"],
            }
        )
        done.add(key)
        added += 1
        _maybe_save()

    if workers == 1:
        _worker_init(args.url, names)
        for task in iter_progress(pending, desc="gather golden", unit="pair"):
            rec = _gather_one(task)
            _apply(rec)
            if rec.get("error"):
                print(f"ERROR {errors[-1]}", file=sys.stderr, flush=True)
    else:
        def _on_gather(rec: dict, bar) -> None:
            if rec.get("error"):
                bar.write(f"ERR {rec['model']} {rec['name'][:32]}: {rec['error'][:60]}")

        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_worker_init,
            initargs=(args.url, names),
        ) as pool:
            for rec in map_progress(
                _gather_one,
                pending,
                pool=pool,
                desc=f"gather golden ({workers}w, {len(pending)} pairs)",
                unit="pair",
                on_result=_on_gather,
            ):
                _apply(rec)

    _save_out(args.out, rows)

    if args.merge_smoke:
        from tests.support import load_golden

        for r in load_golden():
            key = (r.get("model"), r.get("smiles"))
            if key not in done:
                rows.append(r)
                done.add(key)
        _save_out(args.out, rows)

    dt = time.time() - t0
    print(f"wrote {args.out} total={len(rows)} new={added} errors={len(errors)} ({dt:.1f}s)")
    if errors:
        print("first errors:", file=sys.stderr)
        for e in errors[:10]:
            print(f"  {e}", file=sys.stderr)
        return 2 if not added else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
