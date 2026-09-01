#!/usr/bin/env python3
"""Compare default (production) vs legacy ``_parameter`` ONNX scores on golden molecules.

Usage::

    uv run python tools/study_legacy_vs_default.py
    uv run python tools/study_legacy_vs_default.py --jobs 8
    uv run python tools/study_legacy_vs_default.py --limit 50 --json /tmp/parity.json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

STUDY_MODELS = ("epoxidation", "quinone", "reactivity", "ugt", "ndealk", "isozyme", "phase1")

FLAG_ABLATIONS = {
    "ndealk_site": {"ndealk_site_mode": "legacy"},
    "quinone_omp": {"quinone_omp_mode": "legacy"},
    "symmetry": {"symmetry_group_mode": "openbabel"},
    "bond_nrings": {"bond_nrings_mode": "legacy"},
}


@dataclass
class RowDelta:
    smiles: str
    model: str
    max_delta: float
    deltas: dict[str, float] = field(default_factory=dict)
    principled_warn: bool = False


def _score_delta(default_fields: dict, other_fields: dict) -> tuple[float, dict[str, float]]:
    deltas: dict[str, float] = {}
    max_delta = 0.0
    for key in ("mol", "atom", "bond", "pair"):
        va, vb = default_fields.get(key), other_fields.get(key)
        if va is None or vb is None:
            continue
        if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
            d = abs(float(va) - float(vb))
            deltas[key] = d
            max_delta = max(max_delta, d)
        elif isinstance(va, list) and isinstance(vb, list):
            if len(va) != len(vb):
                deltas[f"{key}_len"] = float(abs(len(va) - len(vb)))
                max_delta = max(max_delta, deltas[f"{key}_len"])
            for i, (x, y) in enumerate(zip(va, vb)):
                d = abs(float(x) - float(y))
                deltas[f"{key}[{i}]"] = d
                max_delta = max(max_delta, d)
    return max_delta, deltas


def _heads_delta(default_heads: dict[str, dict], other_heads: dict[str, dict]) -> tuple[float, dict[str, float]]:
    max_delta = 0.0
    deltas: dict[str, float] = {}
    for name in sorted(set(default_heads) | set(other_heads)):
        if name not in default_heads or name not in other_heads:
            deltas[f"missing:{name}"] = 1.0
            max_delta = max(max_delta, 1.0)
            continue
        d, parts = _score_delta(default_heads[name], other_heads[name])
        max_delta = max(max_delta, d)
        for k, v in parts.items():
            deltas[f"{name}.{k}"] = v
    return max_delta, deltas


def _all_heads(smiles: str, model: str, *, parameter: dict | None, backend) -> dict[str, dict]:
    from xenosite.predict import predict
    from tests.support import golden_score_fields

    kwargs: dict = {"models": [model], "backend": backend}
    if parameter is not None:
        kwargs["_parameter"] = parameter
    mol = predict(smiles, **kwargs)
    return {r.model: golden_score_fields(r) for r in mol.results}


def _study_one(model: str, smiles: str) -> tuple[RowDelta, dict[str, float]]:
    """Worker: compare default vs legacy for one (model, smiles) pair."""
    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, str(ROOT))

    from xenosite.predict.backends.onnx import OnnxBackend
    from tests.support import GOLDEN_PARAMETER, PRINCIPLED_PARAMETER, onnx_root

    backend = OnnxBackend(onnx_root())

    default = _all_heads(smiles, model, parameter=None, backend=backend)
    principled = _all_heads(smiles, model, parameter=PRINCIPLED_PARAMETER, backend=backend)
    legacy = _all_heads(smiles, model, parameter=GOLDEN_PARAMETER, backend=backend)

    p_delta, _ = _heads_delta(default, principled)
    max_delta, deltas = _heads_delta(default, legacy)
    row = RowDelta(
        smiles=smiles,
        model=model,
        max_delta=max_delta,
        deltas=deltas,
        principled_warn=p_delta > 1e-12,
    )

    ablation: dict[str, float] = {}
    for flag_name, param in FLAG_ABLATIONS.items():
        if model in ("reactivity", "ugt"):
            continue
        if model == "isozyme" and flag_name == "quinone_omp":
            continue
        ablated = _all_heads(smiles, model, parameter=param, backend=backend)
        d, _ = _heads_delta(default, ablated)
        ablation[f"{model}:{flag_name}"] = d

    return row, ablation


def _unique_smiles_by_model(rows: list[dict]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = defaultdict(list)
    seen: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        model = row.get("model")
        smiles = row.get("smiles")
        if not model or not smiles or model not in STUDY_MODELS:
            continue
        if smiles in seen[model]:
            continue
        seen[model].add(smiles)
        out[model].append(smiles)
    return out


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    k = (len(values) - 1) * p / 100.0
    f = int(k)
    c = min(f + 1, len(values) - 1)
    if f == c:
        return values[f]
    return values[f] + (values[c] - values[f]) * (k - f)


def _build_tasks(*, limit: int | None) -> list[tuple[str, str]]:
    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, str(ROOT))
    from tests.support import onnx_weights_present, load_golden_suite

    by_model = _unique_smiles_by_model(load_golden_suite())
    tasks: list[tuple[str, str]] = []
    for model in STUDY_MODELS:
        weight_key = "ndealk" if model == "isozyme" else model
        if not onnx_weights_present(weight_key):
            continue
        smiles_list = by_model.get(model, [])
        if limit is not None:
            smiles_list = smiles_list[:limit]
        tasks.extend((model, smi) for smi in smiles_list)
    return tasks


def run_study(*, limit: int | None = None, jobs: int = 1) -> dict:
    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, str(ROOT))
    from tests.support import PARITY_ATOL

    tasks = _build_tasks(limit=limit)
    rows_out: list[RowDelta] = []
    ablation_rows: dict[str, list[float]] = defaultdict(list)

    if jobs <= 1:
        for model, smiles in tasks:
            row, ablation = _study_one(model, smiles)
            if row.principled_warn:
                print(
                    f"WARN default != principled {model} {smiles[:40]}",
                    file=sys.stderr,
                )
            rows_out.append(row)
            for k, v in ablation.items():
                ablation_rows[k].append(v)
    else:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            futures = {pool.submit(_study_one, model, smi): (model, smi) for model, smi in tasks}
            done = 0
            for fut in as_completed(futures):
                model, smiles = futures[fut]
                row, ablation = fut.result()
                if row.principled_warn:
                    print(
                        f"WARN default != principled {model} {smiles[:40]}",
                        file=sys.stderr,
                    )
                rows_out.append(row)
                for k, v in ablation.items():
                    ablation_rows[k].append(v)
                done += 1
                if done % 50 == 0 or done == len(tasks):
                    print(f"progress {done}/{len(tasks)}", file=sys.stderr)

    summary: dict[str, dict] = {}
    for model in STUDY_MODELS:
        model_rows = [r for r in rows_out if r.model == model]
        if not model_rows:
            continue
        deltas = [r.max_delta for r in model_rows]
        within = sum(1 for d in deltas if d <= PARITY_ATOL)
        summary[model] = {
            "n": len(model_rows),
            "within_atol": within,
            "frac_within_atol": within / len(model_rows),
            "max_delta": max(deltas),
            "mean_max_delta": statistics.mean(deltas),
            "median_max_delta": statistics.median(deltas),
            "p95_max_delta": _percentile(deltas, 95),
            "atol": PARITY_ATOL,
            "top5": sorted(
                [{"smiles": r.smiles, "max_delta": r.max_delta} for r in model_rows],
                key=lambda x: x["max_delta"],
                reverse=True,
            )[:5],
        }

    ablation_summary = {}
    for key, vals in sorted(ablation_rows.items()):
        ablation_summary[key] = {
            "n": len(vals),
            "mean_max_delta": statistics.mean(vals) if vals else 0.0,
            "median_max_delta": statistics.median(vals) if vals else 0.0,
            "p95_max_delta": _percentile(vals, 95),
            "max_delta": max(vals) if vals else 0.0,
        }

    return {
        "summary": summary,
        "ablation": ablation_summary,
        "n_rows": len(rows_out),
        "jobs": jobs,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--limit", type=int, default=None, help="Max molecules per model")
    p.add_argument("--jobs", "-j", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    p.add_argument("--json", type=Path, default=None, help="Write full JSON report")
    args = p.parse_args(argv)

    report = run_study(limit=args.limit, jobs=args.jobs)
    print(json.dumps(report["summary"], indent=2))
    print("\n# Single-flag ablation (default vs default+one legacy flag)\n")
    print(json.dumps(report["ablation"], indent=2))

    if args.json:
        args.json.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nWrote {args.json}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
