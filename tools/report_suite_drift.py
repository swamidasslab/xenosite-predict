#!/usr/bin/env python3
"""Report ONNX vs golden-suite drift from cached predictions (fast).

Capture once::

  uv run python tools/capture_suite_onnx.py

Then analyze instantly (no model inference)::

  uv run python tools/report_suite_drift.py
  uv run python tools/report_suite_drift.py --model quinone --fail-only

Use ``--refresh`` to capture missing/stale rows before reporting.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.support import (  # noqa: E402
    SUITE_MODELS,
    load_golden_suite,
)
from tools.suite_drift_lib import (  # noqa: E402
    DEFAULT_CACHE,
    analyze_suite,
    default_workers,
    format_report,
    load_cache,
    save_failing_smiles,
)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    p.add_argument("--model", default="", help="filter to one model")
    p.add_argument("--smiles", default="", help="filter to one SMILES")
    p.add_argument("--fail-only", action="store_true", help="only print if failures")
    p.add_argument(
        "--refresh",
        action="store_true",
        help="run capture_suite_onnx for missing pairs first",
    )
    p.add_argument("--top", type=int, default=15, help="examples in report")
    p.add_argument(
        "--workers",
        type=int,
        default=default_workers(),
        help="parallel worker processes (default: min(cpu_count, 24) or XENOSITE_WORKERS)",
    )
    args = p.parse_args(argv)

    if args.refresh:
        cmd = [sys.executable, str(ROOT / "tools" / "capture_suite_onnx.py"), "--cache", str(args.cache)]
        if args.model:
            cmd.extend(["--model", args.model])
        if args.smiles:
            cmd.extend(["--smiles", args.smiles])
        cmd.extend(["--workers", str(max(1, args.workers))])
        subprocess.check_call(cmd)

    cache = load_cache(args.cache)
    if not cache:
        print(
            f"No cache at {args.cache}. Run:\n"
            f"  uv run python tools/capture_suite_onnx.py",
            file=sys.stderr,
        )
        return 2

    models = [args.model] if args.model else [m for m in SUITE_MODELS if m != "phase1"]
    golden = load_golden_suite(merge_smoke=True)
    if args.smiles:
        golden = [g for g in golden if g.get("smiles") == args.smiles]

    report = analyze_suite(
        golden,
        cache,
        models=models,
        workers=max(1, args.workers),
    )
    save_failing_smiles(report)

    if args.fail_only and report.pytest_fail_rows == 0:
        return 0

    # Report on stdout so it does not clobber the tqdm bar on stderr.
    sys.stdout.write(format_report(report, top_n=args.top) + "\n")
    uncached = report.by_field.get("uncached", 0)
    if uncached:
        sys.stdout.write(
            f"\n({uncached} golden rows missing from cache — run capture_suite_onnx.py)\n"
        )
    return 1 if report.pytest_fail_rows else 0


if __name__ == "__main__":
    raise SystemExit(main())
