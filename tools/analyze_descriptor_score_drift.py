#!/usr/bin/env python3
"""Cross-tab OB descriptor drift vs ONNX/golden score drift."""

from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.support import (  # noqa: E402
    SUITE_MODELS,
    descriptor_mismatch_columns,
    descriptor_omp_only,
    descriptor_passes,
    load_golden_suite,
)
from tools.suite_drift_lib import (  # noqa: E402
    DEFAULT_CACHE,
    analyze_suite,
    default_workers,
    load_cache,
)


def descriptor_status(smiles: str, model: str, *, dump_by_smiles: dict[str, dict]) -> str:
    cols = descriptor_mismatch_columns_cached(smiles, model, dump_by_smiles)
    if cols == ["_missing_dump"]:
        return "missing_dump"
    if not cols:
        return "pass"
    if all("Ortho_" in c or "Meta_" in c or "Para_" in c for c in cols):
        return "omp_only"
    return "fail"


def load_dump_by_smiles() -> dict[str, dict]:
    from tests.support import load_ob_dumps

    return {d["smiles"]: d for d in load_ob_dumps() if d.get("smiles")}


def descriptor_mismatch_columns_cached(
    smiles: str, model: str, dump_by_smiles: dict[str, dict]
) -> list[str]:
    from tests.support import compare_feature_dump_rows, dump_compare_skip_columns, rows_for_model
    from xenosite.predict.molecule import parse_smiles

    dump_mol = dump_by_smiles.get(smiles)
    if dump_mol is None or model not in (dump_mol.get("models") or {}):
        return ["_missing_dump"]
    mol, _ = parse_smiles(smiles)
    mm = compare_feature_dump_rows(
        rows_for_model(model, mol),
        dump_mol["models"][model],
        skip_columns=dump_compare_skip_columns(model),
    )
    return [c for c in mm if not c.startswith("_")]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    p.add_argument("--model", default="quinone")
    p.add_argument("--workers", type=int, default=default_workers())
    args = p.parse_args(argv)

    golden = load_golden_suite(merge_smoke=True)
    cache = load_cache(args.cache)
    models = [args.model] if args.model else [m for m in SUITE_MODELS if m != "phase1"]
    report = analyze_suite(
        golden, cache, models=models, workers=max(1, args.workers)
    )

    dump_by_smiles = load_dump_by_smiles()
    print(f"(loaded {len(dump_by_smiles)} OB dumps)\n")

    by_smiles: dict[str, dict] = defaultdict(lambda: {"ordering": 0, "real": 0})
    for d in report.alignment_diagnostics:
        key = "ordering" if d.ordering_only else "real"
        by_smiles[d.smiles][key] += 1

    failing = sorted(report.unique_failing_smiles)
    model = args.model or models[0]
    print(f"=== {model}: failing SMILES {len(failing)} ===\n")

    counts: Counter[str] = Counter()
    col_hits: Counter[str] = Counter()
    xtab: Counter[tuple[str, str]] = Counter()
    rows: list[tuple] = []

    for smi in failing:
        status = descriptor_status(smi, model, dump_by_smiles=dump_by_smiles)
        counts[status] += 1
        cols = descriptor_mismatch_columns_cached(smi, model, dump_by_smiles)
        if status not in ("pass", "missing_dump"):
            for c in cols:
                col_hits[c] += 1
        align = by_smiles.get(smi, {"ordering": 0, "real": 0})
        o, r = align["ordering"], align["real"]
        if o and not r:
            al = "ordering_only"
        elif r and not o:
            al = "real_drift"
        elif o and r:
            al = "mixed"
        else:
            al = "no_align_diag"
        xtab[(status, al)] += 1
        rows.append((smi, status, o, r, cols))

    print("descriptor status (failing SMILES):")
    for k in ("pass", "omp_only", "fail", "missing_dump"):
        print(f"  {k}: {counts[k]}")

    print("\ndescriptor × score alignment:")
    for key, n in sorted(xtab.items(), key=lambda x: -x[1]):
        print(f"  {key[0]:12} + {key[1]:16}: {n}")

    real_smiles = {smi for smi, _, o, r, _ in rows if r > 0}
    ord_smiles = {smi for smi, _, o, r, _ in rows if o > 0 and r == 0}

    def _passes(s: str) -> bool:
        return not descriptor_mismatch_columns_cached(s, model, dump_by_smiles)

    def _omp_only(s: str) -> bool:
        cols = descriptor_mismatch_columns_cached(s, model, dump_by_smiles)
        return bool(cols) and all("Ortho_" in c or "Meta_" in c or "Para_" in c for c in cols)

    print(f"\nreal_drift SMILES (n={len(real_smiles)}):")
    for label, fn in [
        ("pass", _passes),
        ("omp_only", _omp_only),
    ]:
        print(f"  descriptor {label}: {sum(1 for s in real_smiles if fn(s))}")
    print(
        "  descriptor fail: "
        f"{sum(1 for s in real_smiles if not _passes(s) and not _omp_only(s))}"
    )

    print(f"\nordering_only SMILES (n={len(ord_smiles)}):")
    print(f"  descriptor pass: {sum(1 for s in ord_smiles if _passes(s))}")
    print(f"  descriptor fail/omp: {len(ord_smiles) - sum(1 for s in ord_smiles if _passes(s))}")

    all_smiles = {g["smiles"] for g in golden if g.get("model") == model}
    print(f"\nall {model} suite SMILES (n={len(all_smiles)}):")
    print(f"  descriptor pass: {sum(1 for s in all_smiles if _passes(s))}")
    print(f"  omp_only:        {sum(1 for s in all_smiles if _omp_only(s))}")
    print(
        "  other fail:      "
        f"{sum(1 for s in all_smiles if not _passes(s) and not _omp_only(s))}"
    )

    if col_hits:
        print("\ntop mismatch columns (failing SMILES):")
        for c, n in col_hits.most_common(12):
            print(f"  {c}: {n}")

    print("\nreal drift + descriptor pass (score path, not OB):")
    shown = 0
    for smi, status, o, r, _ in rows:
        if r > 0 and status == "pass":
            print(f"  {smi[:72]}")
            shown += 1
            if shown >= 6:
                break

    print("\nreal drift + non-OMP descriptor fail:")
    shown = 0
    for smi, status, o, r, cols in rows:
        if r > 0 and status == "fail":
            print(f"  {smi[:60]} cols={cols[:4]}")
            shown += 1
            if shown >= 6:
                break

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
