#!/usr/bin/env python3
"""Classify drift-report failures by mechanism and [nH] correlation."""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.support import SUITE_MODELS, load_golden_suite  # noqa: E402
from tools.suite_drift_lib import (  # noqa: E402
    DEFAULT_CACHE,
    analyze_suite,
    default_workers,
    load_cache,
)

NH = re.compile(r"\[[nN][Hh]")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    p.add_argument("--workers", type=int, default=default_workers())
    args = p.parse_args(argv)

    golden = load_golden_suite(merge_smoke=True)
    cache = load_cache(args.cache)
    models = [m for m in SUITE_MODELS if m != "phase1"]
    report = analyze_suite(golden, cache, models=models, workers=max(1, args.workers))

    by_mechanism: Counter[str] = Counter()
    by_nh: Counter[str] = Counter()
    for ff in report.field_failures:
        key = "nh" if NH.search(ff.smiles) else "other"
        by_nh[key] += 1
        if ff.field == "pair_idx" and ff.max_diff in (1.0, 2.0):
            mech = "pair_idx_off_by_one"
        elif ff.max_diff >= 0.5:
            mech = "large_score_drift"
        elif ff.max_diff >= 0.02:
            mech = "medium_drift"
        elif ff.max_diff >= 0.001:
            mech = "small_above_atol"
        else:
            mech = "tiny_above_atol"
        by_mechanism[mech] += 1

    nh_smiles = {s for s in report.unique_failing_smiles if NH.search(s)}
    other_smiles = report.unique_failing_smiles - nh_smiles

    nh_row_fails: Counter[str] = Counter()
    other_row_fails: Counter[str] = Counter()
    seen_rows: set[tuple[str, str]] = set()
    for ff in report.field_failures:
        key = (ff.model, ff.smiles)
        if key in seen_rows:
            continue
        seen_rows.add(key)
        if NH.search(ff.smiles):
            nh_row_fails[ff.model] += 1
        else:
            other_row_fails[ff.model] += 1
    # uncached / predict_error rows (no field failures)
    for model, _smiles, _err in report.predict_errors:
        key = (model, _smiles)
        if key in seen_rows:
            continue
        seen_rows.add(key)
        if NH.search(_smiles):
            nh_row_fails[model] += 1
        else:
            other_row_fails[model] += 1
    uncached = report.by_field.get("uncached", 0)
    print("=== ROW FAILURES ===")
    print(f"(by_mol_class from report: {dict(report.by_mol_class)})")
    if uncached:
        print(f"uncached rows: {uncached}")
    print(f"pytest fail rows: {report.pytest_fail_rows}")
    print(
        f"unique failing SMILES: {len(report.unique_failing_smiles)} "
        f"({len(nh_smiles)} [nH], {len(other_smiles)} other)"
    )
    print(f"row fails on [nH]: {sum(nh_row_fails.values())}")
    print(f"row fails on other: {sum(other_row_fails.values())}")
    print()
    print("=== BY MODEL ([nH] / other row fails) ===")
    for m in sorted(set(nh_row_fails) | set(other_row_fails)):
        print(f"  {m}: [nH]={nh_row_fails[m]} other={other_row_fails[m]}")
    print()
    print(f"=== FIELD FAILURES ({len(report.field_failures)} head hits) ===")
    for k, v in by_mechanism.most_common():
        print(f"  {k}: {v}")
    print(f"  [nH] field hits: {by_nh['nh']}  other: {by_nh['other']}")
    print()
    print("=== KEY COUNTS ===")
    print(
        "pair_idx off-by-one:",
        sum(1 for f in report.field_failures if f.field == "pair_idx"),
    )
    print(
        "reactivity atom large (>=0.05) on [nH]:",
        sum(
            1
            for f in report.field_failures
            if f.model == "reactivity"
            and f.field == "atom"
            and f.max_diff >= 0.05
            and NH.search(f.smiles)
        ),
    )
    print(
        "isozyme bond on [nH] / other:",
        sum(
            1
            for f in report.field_failures
            if f.model == "isozyme" and f.field == "bond" and NH.search(f.smiles)
        ),
        "/",
        sum(
            1
            for f in report.field_failures
            if f.model == "isozyme" and f.field == "bond" and not NH.search(f.smiles)
        ),
    )
    if other_smiles:
        print()
        print("=== sample non-[nH] failing SMILES ===")
        for s in sorted(other_smiles)[:8]:
            print(f"  {s[:70]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
