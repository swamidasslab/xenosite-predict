"""Shared ONNX-vs-golden suite drift: capture, compare, classify (no predict in report)."""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE = ROOT / "tests" / "fixtures" / "suite_onnx_cache.json"

NH_SMILES = re.compile(r"\[[nN][Hh]")


@dataclass
class FieldFailure:
    model: str
    smiles: str
    name: str
    head: str
    field: str
    max_diff: float
    atol: float


@dataclass
class DriftReport:
    pytest_fail_rows: int = 0
    pytest_pass_rows: int = 0
    head_failures: int = 0
    by_model: Counter = field(default_factory=Counter)
    by_field: Counter = field(default_factory=Counter)
    by_model_field: Counter = field(default_factory=Counter)
    by_mol_class: Counter = field(default_factory=Counter)
    by_magnitude: Counter = field(default_factory=Counter)
    unique_failing_smiles: set[str] = field(default_factory=set)
    field_failures: list[FieldFailure] = field(default_factory=list)
    predict_errors: list[tuple[str, str, str]] = field(default_factory=list)
    stats_by_model: dict[str, list[float]] = field(default_factory=dict)


def load_cache(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def save_cache(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")


def cache_index(rows: list[dict]) -> dict[tuple[str, str], dict]:
    return {(r.get("model"), r.get("smiles")): r for r in rows}


def mol_class(smiles: str) -> str:
    return "[nH]" if NH_SMILES.search(smiles) else "other"


def mag_bucket(d: float) -> str:
    if d == float("inf"):
        return "shape_mismatch"
    if d >= 0.5:
        return ">=0.5"
    if d >= 0.05:
        return "0.05-0.5"
    if d >= 0.02:
        return "0.02-0.05"
    if d >= 0.001:
        return "0.001-0.02"
    return "<0.001"


def max_diff(want: dict, have: dict) -> float:
    m = 0.0
    for k in want:
        if k not in have or have[k] is None:
            continue
        wa, ha = np.asarray(want[k], dtype=float), np.asarray(have[k], dtype=float)
        if wa.shape != ha.shape:
            return float("inf")
        if wa.size:
            m = max(m, float(np.max(np.abs(wa - ha))))
    return m


def failing_fields(
    want: dict, have: dict, atol: float
) -> list[tuple[str, float]]:
    out: list[tuple[str, float]] = []
    for k in want:
        if k not in have or have.get(k) is None:
            continue
        wa, ha = np.asarray(want[k], dtype=float), np.asarray(have[k], dtype=float)
        if wa.shape != ha.shape:
            out.append((k, float("inf")))
            continue
        if wa.size:
            d = float(np.max(np.abs(wa - ha)))
            if d > atol:
                out.append((k, d))
    return out


def analyze_suite(
    golden_rows: list[dict],
    onnx_rows: list[dict],
    *,
    models: list[str],
    normalize_quinone,
    golden_score_fields,
    parity_atol,
) -> DriftReport:
    """Compare cached ONNX rows to golden; no inference."""
    onnx_by = cache_index(onnx_rows)
    report = DriftReport()

    for g in golden_rows:
        model = g.get("model") or ""
        if model not in models:
            continue
        smiles = g.get("smiles") or ""
        key = (model, smiles)
        atol = parity_atol(smiles, model)
        cached = onnx_by.get(key)

        if cached is None:
            report.pytest_fail_rows += 1
            report.by_model[model] += 1
            report.by_mol_class[mol_class(smiles)] += 1
            report.by_field["uncached"] += 1
            report.unique_failing_smiles.add(smiles)
            continue

        if cached.get("error"):
            report.pytest_fail_rows += 1
            report.by_model[model] += 1
            report.by_mol_class[mol_class(smiles)] += 1
            report.by_field["predict_error"] += 1
            report.predict_errors.append((model, smiles, cached["error"]))
            report.unique_failing_smiles.add(smiles)
            continue

        onnx_results = {r["model"]: r for r in cached.get("results") or []}
        row_failed = False
        row_max = 0.0

        for gr in g.get("results") or []:
            head = gr.get("model") or ""
            if head not in onnx_results:
                report.head_failures += 1
                report.by_field["missing_head"] += 1
                if not row_failed:
                    report.pytest_fail_rows += 1
                    report.by_model[model] += 1
                    report.by_mol_class[mol_class(smiles)] += 1
                    report.unique_failing_smiles.add(smiles)
                    row_failed = True
                continue

            want = golden_score_fields(gr)
            have = golden_score_fields(onnx_results[head])
            if head == "quinone":
                normalize_quinone(want, smiles)
                normalize_quinone(have, smiles)

            d = max_diff(want, have)
            row_max = max(row_max, d)
            report.stats_by_model.setdefault(model, []).append(d)

            bad = failing_fields(want, have, atol)
            if bad:
                report.head_failures += len(bad)
                if not row_failed:
                    report.pytest_fail_rows += 1
                    report.by_model[model] += 1
                    report.by_mol_class[mol_class(smiles)] += 1
                    report.unique_failing_smiles.add(smiles)
                    row_failed = True
                for fld, fd in bad:
                    report.by_field[fld] += 1
                    report.by_model_field[(model, fld)] += 1
                    report.by_magnitude[mag_bucket(fd)] += 1
                    report.field_failures.append(
                        FieldFailure(
                            model=model,
                            smiles=smiles,
                            name=g.get("name") or smiles[:40],
                            head=head,
                            field=fld,
                            max_diff=fd,
                            atol=atol,
                        )
                    )
            elif d <= atol:
                pass

        if not row_failed:
            report.pytest_pass_rows += 1
            if model not in report.stats_by_model:
                report.stats_by_model[model] = [row_max]

    return report


def format_report(report: DriftReport, *, top_n: int = 15) -> str:
    lines: list[str] = []
    lines.append("=== pytest-equivalent (one row per model×SMILES) ===")
    lines.append(f"fail: {report.pytest_fail_rows}")
    lines.append(f"pass: {report.pytest_pass_rows}")
    lines.append(f"unique failing SMILES: {len(report.unique_failing_smiles)}")
    lines.append(f"head-level failures (legacy report count): {report.head_failures}")
    lines.append("")
    lines.append("=== by model (failed rows) ===")
    for m, c in sorted(report.by_model.items(), key=lambda x: -x[1]):
        vals = report.stats_by_model.get(m, [])
        if vals:
            arr = np.asarray(vals)
            lines.append(
                f"  {m}: fail_rows={c} n={len(vals)} max={arr.max():.6g} "
                f"p95={np.percentile(arr, 95):.6g}"
            )
        else:
            lines.append(f"  {m}: fail_rows={c}")
    lines.append("")
    lines.append("=== by failing score field ===")
    for k, c in sorted(report.by_field.items(), key=lambda x: -x[1]):
        lines.append(f"  {k}: {c}")
    lines.append("")
    lines.append("=== model × field (top) ===")
    for (m, k), c in sorted(report.by_model_field.items(), key=lambda x: -x[1])[:top_n]:
        lines.append(f"  {m}/{k}: {c}")
    lines.append("")
    lines.append("=== molecule class (failed rows) ===")
    for k, c in report.by_mol_class.most_common():
        lines.append(f"  {k}: {c}")
    lines.append("")
    lines.append("=== magnitude (field hits) ===")
    for k in ("shape_mismatch", ">=0.5", "0.05-0.5", "0.02-0.05", "0.001-0.02", "<0.001"):
        if report.by_magnitude.get(k):
            lines.append(f"  {k}: {report.by_magnitude[k]}")
    if report.predict_errors:
        lines.append("")
        lines.append(f"=== predict errors ({len(report.predict_errors)}) ===")
        for row in report.predict_errors[:5]:
            lines.append(f"  {row}")
    lines.append("")
    lines.append(f"=== largest diffs (top {top_n}) ===")
    ranked = sorted(report.field_failures, key=lambda f: -f.max_diff)[:top_n]
    for f in ranked:
        lines.append(
            f"  {f.model}/{f.field} {f.name[:30]} diff={f.max_diff:.6g} atol={f.atol:g}"
        )
    return "\n".join(lines)
