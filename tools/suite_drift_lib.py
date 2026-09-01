"""Shared ONNX-vs-golden suite drift: capture, compare, classify (no predict in report)."""

from __future__ import annotations

import json
import os
import re
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE = ROOT / "tests" / "fixtures" / "suite_onnx_cache.json"
FAILING_SMILES_JSON = ROOT / "tests" / "fixtures" / "failing_suite_smiles.json"

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
class AlignmentDiagnostic:
    """Positional compare failed but index-free matching may explain the gap."""

    model: str
    smiles: str
    name: str
    head: str
    field: str
    positional_diff: float
    matched_diff: float
    atol: float

    @property
    def ordering_only(self) -> bool:
        return self.positional_diff > self.atol and self.matched_diff <= self.atol


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
    alignment_ordering_only: Counter = field(default_factory=Counter)
    alignment_real_drift: Counter = field(default_factory=Counter)
    alignment_diagnostics: list[AlignmentDiagnostic] = field(default_factory=list)


def load_cache(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def save_cache(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")


def cache_index(rows: list[dict]) -> dict[tuple[str, str], dict]:
    return {(r.get("model"), r.get("smiles")): r for r in rows}


def default_workers() -> int:
    """Default pool size: ``XENOSITE_WORKERS``, else min(cpu_count, 24)."""
    env = os.environ.get("XENOSITE_WORKERS", "").strip()
    if env.isdigit():
        return max(1, int(env))
    n = os.cpu_count() or 4
    return max(1, min(n, 24))


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


def index_free_max_diff(want: np.ndarray, have: np.ndarray) -> float:
    """Minimax score gap over bijections, ignoring vector order (sorted matching).

    For equal-length vectors this equals ``min_π max_i |want[i] - have[π(i)]|``.
    Different lengths return ``inf`` (cannot align as a pure reordering issue).
    """
    a = np.asarray(want, dtype=float).reshape(-1)
    b = np.asarray(have, dtype=float).reshape(-1)
    if a.size != b.size:
        return float("inf")
    if a.size == 0:
        return 0.0
    return float(np.max(np.abs(np.sort(a) - np.sort(b))))


def marriage_assignment_max_diff(want: np.ndarray, have: np.ndarray) -> float:
    """Maximum edge cost in a minimum-sum bipartite matching (Hungarian).

    Ignores index alignment when pairing scores. Falls back to
    :func:`index_free_max_diff` when SciPy is unavailable.
    """
    a = np.asarray(want, dtype=float).reshape(-1)
    b = np.asarray(have, dtype=float).reshape(-1)
    if a.size != b.size:
        return float("inf")
    if a.size == 0:
        return 0.0
    if a.size == 1:
        return float(abs(a[0] - b[0]))
    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError:
        return index_free_max_diff(a, b)
    cost = np.abs(a[:, None] - b[None, :])
    row, col = linear_sum_assignment(cost)
    return float(np.max(cost[row, col]))


def classify_alignment(positional_diff: float, matched_diff: float, atol: float) -> str | None:
    """``ordering_only``, ``real_drift``, or None when positional compare passed."""
    if positional_diff <= atol:
        return None
    if matched_diff <= atol:
        return "ordering_only"
    return "real_drift"


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


def _analyze_golden_row(task: tuple[dict, dict | None]) -> DriftReport:
    """Compare one golden row to its cached ONNX result (worker-safe)."""
    import sys

    root = str(ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)

    from tests.support import golden_score_fields, parity_atol
    from xenosite.predict.numbering import normalize_quinone_pair_fields

    g, cached = task
    report = DriftReport()
    model = g.get("model") or ""
    smiles = g.get("smiles") or ""
    atol = parity_atol(smiles, model)

    if cached is None:
        report.pytest_fail_rows += 1
        report.by_model[model] += 1
        report.by_mol_class[mol_class(smiles)] += 1
        report.by_field["uncached"] += 1
        report.unique_failing_smiles.add(smiles)
        return report

    if cached.get("error"):
        report.pytest_fail_rows += 1
        report.by_model[model] += 1
        report.by_mol_class[mol_class(smiles)] += 1
        report.by_field["predict_error"] += 1
        report.predict_errors.append((model, smiles, cached["error"]))
        report.unique_failing_smiles.add(smiles)
        return report

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
            from xenosite.predict.molecule import parse_smiles

            mol, _ = parse_smiles(smiles)
            normalize_quinone_pair_fields(want, have, mol=mol)

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
                if fld in ("pair_idx", "mol") or not np.isfinite(fd):
                    continue
                wa = np.asarray(want.get(fld, []), dtype=float).reshape(-1)
                ha = np.asarray(have.get(fld, []), dtype=float).reshape(-1)
                if wa.size != ha.size:
                    continue
                matched = marriage_assignment_max_diff(wa, ha)
                kind = classify_alignment(fd, matched, atol)
                if kind is None:
                    continue
                key = (model, fld)
                if kind == "ordering_only":
                    report.alignment_ordering_only[key] += 1
                else:
                    report.alignment_real_drift[key] += 1
                report.alignment_diagnostics.append(
                    AlignmentDiagnostic(
                        model=model,
                        smiles=smiles,
                        name=g.get("name") or smiles[:40],
                        head=head,
                        field=fld,
                        positional_diff=fd,
                        matched_diff=matched,
                        atol=atol,
                    )
                )

    if not row_failed:
        report.pytest_pass_rows += 1
        if model not in report.stats_by_model:
            report.stats_by_model[model] = [row_max]

    return report


def merge_reports(reports: list[DriftReport]) -> DriftReport:
    out = DriftReport()
    for r in reports:
        out.pytest_fail_rows += r.pytest_fail_rows
        out.pytest_pass_rows += r.pytest_pass_rows
        out.head_failures += r.head_failures
        out.by_model.update(r.by_model)
        out.by_field.update(r.by_field)
        out.by_model_field.update(r.by_model_field)
        out.by_mol_class.update(r.by_mol_class)
        out.by_magnitude.update(r.by_magnitude)
        out.unique_failing_smiles |= r.unique_failing_smiles
        out.field_failures.extend(r.field_failures)
        out.predict_errors.extend(r.predict_errors)
        out.alignment_ordering_only.update(r.alignment_ordering_only)
        out.alignment_real_drift.update(r.alignment_real_drift)
        out.alignment_diagnostics.extend(r.alignment_diagnostics)
        for model, vals in r.stats_by_model.items():
            out.stats_by_model.setdefault(model, []).extend(vals)
    return out


def analyze_suite(
    golden_rows: list[dict],
    onnx_rows: list[dict],
    *,
    models: list[str],
    workers: int = 1,
) -> DriftReport:
    """Compare cached ONNX rows to golden; no inference."""
    from tools.progress import iter_progress, map_progress, worker_quiet

    onnx_by = cache_index(onnx_rows)
    model_set = set(models)
    tasks = [
        (g, onnx_by.get((g.get("model"), g.get("smiles"))))
        for g in golden_rows
        if (g.get("model") or "") in model_set
    ]
    if not tasks:
        return DriftReport()

    workers = max(1, workers)
    label = f"analyze drift ({workers}w, {len(tasks)} rows)"
    if workers == 1:
        worker_quiet()
        parts = [
            _analyze_golden_row(task)
            for task in iter_progress(tasks, desc=label, unit="row")
        ]
        return merge_reports(parts)

    with ProcessPoolExecutor(max_workers=workers, initializer=worker_quiet) as pool:
        parts = map_progress(
            _analyze_golden_row,
            tasks,
            pool=pool,
            desc=label,
            unit="row",
        )
    return merge_reports(parts)


def save_failing_smiles(report: DriftReport, path: Path = FAILING_SMILES_JSON) -> None:
    """Write unique failing SMILES for ``gather_golden_suite.py --failing-only``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(sorted(report.unique_failing_smiles), indent=2) + "\n",
        encoding="utf-8",
    )


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
    if report.alignment_diagnostics:
        ordering = [d for d in report.alignment_diagnostics if d.ordering_only]
        real = [d for d in report.alignment_diagnostics if not d.ordering_only]
        lines.append("")
        lines.append("=== score alignment (index-free marriage match) ===")
        lines.append(
            "Positional fail but matched≤atol ⇒ ordering/alignment; "
            "matched>atol ⇒ real score drift."
        )
        lines.append(
            f"  ordering_only field hits: {sum(report.alignment_ordering_only.values())}"
        )
        lines.append(f"  real_drift field hits: {sum(report.alignment_real_drift.values())}")
        if report.alignment_ordering_only:
            lines.append("  ordering_only by model/field:")
            for (m, fld), c in sorted(
                report.alignment_ordering_only.items(), key=lambda x: -x[1]
            )[:top_n]:
                lines.append(f"    {m}/{fld}: {c}")
        if report.alignment_real_drift:
            lines.append("  real_drift by model/field:")
            for (m, fld), c in sorted(
                report.alignment_real_drift.items(), key=lambda x: -x[1]
            )[:top_n]:
                lines.append(f"    {m}/{fld}: {c}")
        if ordering:
            lines.append("")
            lines.append(f"=== ordering-only examples (top {min(top_n, len(ordering))}) ===")
            for d in sorted(
                ordering, key=lambda x: -(x.positional_diff - x.matched_diff)
            )[:top_n]:
                lines.append(
                    f"  {d.model}/{d.field} {d.name[:28]} "
                    f"pos={d.positional_diff:.6g} matched={d.matched_diff:.6g} "
                    f"atol={d.atol:g}"
                )
        if real:
            lines.append("")
            lines.append(
                f"=== real drift (matched still fails, top {min(top_n, len(real))}) ==="
            )
            for d in sorted(real, key=lambda x: -x.matched_diff)[:top_n]:
                lines.append(
                    f"  {d.model}/{d.field} {d.name[:28]} "
                    f"pos={d.positional_diff:.6g} matched={d.matched_diff:.6g} "
                    f"atol={d.atol:g}"
                )
    return "\n".join(lines)
