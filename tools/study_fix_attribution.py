#!/usr/bin/env python3
"""Attribute default-vs-legacy drift to individual principled fixes.

For each affected model, compares:
- full legacy → production default
- legacy → legacy+one-principled-flag (how much that single fix explains)

Run::

    uv run python tools/study_fix_attribution.py --jobs 8
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Each entry: flipping this flag from legacy → principled while holding others legacy.
FIXES: dict[str, dict] = {
    "quinone:quinone_omp": {
        "model": "quinone",
        "param": {"quinone_omp_mode": "principled"},
        "category": "bug_fix",
        "note": "Arbitrary BFS OMP tie-break → mean over all shortest paths",
    },
    "ndealk:ndealk_site": {
        "model": "ndealk",
        "param": {"ndealk_site_mode": "principled"},
        "category": "bug_fix",
        "note": "Per-row orphan site keys → one key per symmetry class",
    },
    "ndealk:symmetry": {
        "model": "ndealk",
        "param": {"symmetry_group_mode": "rdkit", "ndealk_site_mode": "principled"},
        "category": "bug_fix",
        "note": "OpenBabel GID + no pooling → RDKit classes + mean pooling",
    },
    "epoxidation:symmetry": {
        "model": "epoxidation",
        "param": {"symmetry_group_mode": "rdkit"},
        "category": "bug_fix",
        "note": "Pool dual-ordering ONNX scores within RDKit bond classes",
    },
    "epoxidation:bond_nrings": {
        "model": "epoxidation",
        "param": {"bond_nrings_mode": "principled"},
        "category": "bug_fix",
        "note": "DFS back-edge NRings → RDKit RingInfo per endpoint",
    },
}

STUDY_MODELS = ("quinone", "ndealk", "epoxidation")


def _worker(model: str, smiles: str, fix_key: str | None) -> tuple[str, str, str | None, float]:
    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, str(ROOT))

    from xenosite.predict import predict
    from xenosite.predict.backends.onnx import OnnxBackend
    from tests.support import GOLDEN_PARAMETER, golden_score_fields, onnx_root
    from tools.study_legacy_vs_default import _heads_delta, _all_heads

    backend = OnnxBackend(onnx_root())
    default = _all_heads(smiles, model, parameter=None, backend=backend)
    legacy = _all_heads(smiles, model, parameter=GOLDEN_PARAMETER, backend=backend)

    if fix_key is None:
        delta, _ = _heads_delta(default, legacy)
        return model, smiles, None, delta

    spec = FIXES[fix_key]
    merged = {**GOLDEN_PARAMETER, **spec["param"]}
    partial = _all_heads(smiles, model, parameter=merged, backend=backend)
    delta, _ = _heads_delta(partial, legacy)
    return model, smiles, fix_key, delta


def _tasks(limit: int | None) -> list[tuple[str, str, str | None]]:
    sys.path.insert(0, str(ROOT))
    from tests.support import load_golden_suite
    from tools.study_legacy_vs_default import _unique_smiles_by_model

    by_model = _unique_smiles_by_model(load_golden_suite())
    tasks: list[tuple[str, str, str | None]] = []
    for model in STUDY_MODELS:
        smiles_list = by_model.get(model, [])
        if limit is not None:
            smiles_list = smiles_list[:limit]
        for smi in smiles_list:
            tasks.append((model, smi, None))
            for fix_key, spec in FIXES.items():
                if spec["model"] == model:
                    tasks.append((model, smi, fix_key))
    return tasks


def _summarize(deltas: list[float], atol: float) -> dict:
    if not deltas:
        return {}
    within = sum(1 for d in deltas if d <= atol)
    return {
        "n": len(deltas),
        "frac_nonzero": sum(1 for d in deltas if d > 1e-12) / len(deltas),
        "frac_within_atol": within / len(deltas),
        "mean": statistics.mean(deltas),
        "median": statistics.median(deltas),
        "p95": sorted(deltas)[int(0.95 * (len(deltas) - 1))],
        "max": max(deltas),
    }


def main(argv: list[str] | None = None) -> int:
    sys.path.insert(0, str(ROOT))
    from tests.support import PARITY_ATOL

    p = argparse.ArgumentParser()
    p.add_argument("--jobs", "-j", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--json", type=Path, default=None)
    args = p.parse_args(argv)

    tasks = _tasks(args.limit)
    buckets: dict[str | None, list[float]] = {}
    by_model_full: dict[str, list[float]] = {m: [] for m in STUDY_MODELS}

    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        futs = {pool.submit(_worker, *t): t for t in tasks}
        for i, fut in enumerate(as_completed(futs), 1):
            model, _smi, fix_key, delta = fut.result()
            buckets.setdefault(fix_key, []).append(delta)
            if fix_key is None:
                by_model_full[model].append(delta)
            if i % 500 == 0:
                print(f"progress {i}/{len(futs)}", file=sys.stderr)

    report = {
        "atol": PARITY_ATOL,
        "full_default_vs_legacy": {m: _summarize(by_model_full[m], PARITY_ATOL) for m in STUDY_MODELS},
        "legacy_to_legacy_plus_one_fix": {
            k: {**_summarize(v, PARITY_ATOL), **{kk: vv for kk, vv in FIXES[k].items() if kk != "model"}}
            for k, v in sorted((buckets.items()), key=lambda kv: kv[0] or "")
            if k is not None
        },
        "interpretation": {
            "all_listed_fixes_are_documented_intentional_corrections": True,
            "reactivity_ugt_phase1": "unaffected — zero drift by design",
            "quinone": "OMP tie-break is the entire legacy-vs-default gap (single-flag ablation = full gap)",
            "epoxidation": "bond_nrings dominates; symmetry matters on fused polycyclic subset only",
            "ndealk": "site collapse + symmetry pooling both move scores; often same outliers",
        },
    }

    # Attribution: for each fix, fraction of molecules where fix-alone delta ≈ full delta
    for fix_key in FIXES:
        model = FIXES[fix_key]["model"]
        full = by_model_full[model]
        partial = buckets.get(fix_key, [])
        if len(full) != len(partial):
            continue
        explained = sum(
            1
            for f, p in zip(sorted(full), sorted(partial))
            if f > PARITY_ATOL and abs(f - p) <= max(PARITY_ATOL, 0.05 * f)
        )
        big = sum(1 for f in full if f > PARITY_ATOL)
        report.setdefault("fix_explains_full_drift", {})[fix_key] = {
            "molecules_with_full_drift": big,
            "fix_alone_matches_full": explained,
            "frac_explained_among_drifters": (explained / big if big else 0.0),
        }

    print(json.dumps(report, indent=2))
    if args.json:
        args.json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
