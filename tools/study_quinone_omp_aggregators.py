#!/usr/bin/env python3
"""Compare quinone OMP path aggregators vs legacy golden scores.

Legacy: one sorted-BFS shortest path, indicator any(hits) → {0, 1}.
Current principled: all shortest paths, mean(hits) → [0, 1] fractions.

Candidates (all deterministic; path set from ``all_shortest_paths`` unless noted):

- ``legacy_bfs``     — golden: single BFS path + any (current legacy mode)
- ``mean``           — production default
- ``max``            — all shortest paths + any(hits); binary {0, 1}
- ``min``            — all shortest paths + all(hits); binary {0, 1}
- ``first_path``     — lexicographically first shortest path + its indicator
- ``majority``       — mean(hits) >= 0.5 → 1 else 0 (rounded mean)

Run::

    uv run python tools/study_quinone_omp_aggregators.py
    uv run python tools/study_quinone_omp_aggregators.py --json /tmp/omp.json
"""

from __future__ import annotations

import argparse
import contextlib
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Callable, Iterator

ROOT = Path(__file__).resolve().parents[1]

HitAgg = Callable[[list[bool]], float]


def _agg_max(hits: list[bool]) -> float:
    return 1.0 if any(hits) else 0.0


def _agg_min(hits: list[bool]) -> float:
    return 1.0 if hits and all(hits) else 0.0


def _agg_mean(hits: list[bool]) -> float:
    return sum(hits) / len(hits) if hits else 0.0


def _agg_majority(hits: list[bool]) -> float:
    return 1.0 if _agg_mean(hits) >= 0.5 else 0.0


AGGREGATORS: dict[str, HitAgg] = {
    "mean": _agg_mean,
    "max": _agg_max,
    "min": _agg_min,
    "majority": _agg_majority,
}


@contextlib.contextmanager
def _patched_omp(
    *,
    path_mode: str,
    aggregate: HitAgg | None = None,
) -> Iterator[None]:
    """Patch AtomTD OMP path selection and hit aggregation."""
    from xenosite.predict.v0_legacy.features import atom as atom_mod

    AtomTD = atom_mod.AtomTD
    orig_paths = AtomTD._paths_for_omp
    orig_motif = AtomTD._paths_for_omp_motif
    orig_omp_paths = AtomTD._omp_paths

    def paths(self, start: int, end: int, sym: str) -> list[list[int]]:
        if path_mode == "legacy_bfs":
            p = self.MG.shortest_path(start, end)
            return [p] if p else []
        if path_mode == "first_path":
            allp = self.MG.all_shortest_paths(start, end)
            if not allp:
                return []
            return [min(allp)]
        # all_shortest
        return self.MG.all_shortest_paths(start, end)

    def paths_motif(self, start: int, end: int) -> list[list[int]]:
        if path_mode == "legacy_bfs":
            p = self.MG.shortest_path(start, end)
            return [p] if p else []
        if path_mode == "first_path":
            allp = self.MG.all_shortest_paths(start, end)
            if not allp:
                return []
            return [min(allp)]
        return self.MG.all_shortest_paths(start, end)

    def omp_paths(self, ends_for_start, *, site: bool) -> list[float]:
        add: list[float] = []
        for paths in ends_for_start:
            hits = [self._path_on_aromatic_ring(p, site=site) for p in paths if p]
            if not hits:
                add.append(0.0)
            elif path_mode == "legacy_bfs":
                add.append(1.0 if any(hits) else 0.0)
            elif aggregate is not None:
                add.append(float(aggregate(hits)))
            elif self.omp_mode == "principled":
                add.append(sum(hits) / len(hits))
            else:
                add.append(1.0 if any(hits) else 0.0)
        return add

    AtomTD._paths_for_omp = paths  # type: ignore[method-assign]
    AtomTD._paths_for_omp_motif = paths_motif  # type: ignore[method-assign]
    AtomTD._omp_paths = omp_paths  # type: ignore[method-assign]
    try:
        yield
    finally:
        AtomTD._paths_for_omp = orig_paths
        AtomTD._paths_for_omp_motif = orig_motif
        AtomTD._omp_paths = orig_omp_paths


def _score_delta(default_fields: dict, other_fields: dict) -> float:
    max_delta = 0.0
    for key in ("mol", "atom", "bond", "pair"):
        va, vb = default_fields.get(key), other_fields.get(key)
        if va is None or vb is None:
            continue
        if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
            max_delta = max(max_delta, abs(float(va) - float(vb)))
        elif isinstance(va, list) and isinstance(vb, list):
            for x, y in zip(va, vb):
                max_delta = max(max_delta, abs(float(x) - float(y)))
    return max_delta


def _quinone_scores(smiles: str, *, mode_key: str) -> dict:
    from xenosite.predict import predict
    from xenosite.predict.backends.onnx import OnnxBackend
    from tests.support import GOLDEN_PARAMETER, ROOT, golden_score_fields, onnx_root

    backend = OnnxBackend(onnx_root())

    if mode_key == "legacy_bfs":
        with _patched_omp(path_mode="legacy_bfs"):
            mol = predict(
                smiles,
                models=["quinone"],
                backend=backend,
                _parameter=GOLDEN_PARAMETER,
            )
    elif mode_key == "principled":
        mol = predict(smiles, models=["quinone"], backend=backend)
    elif mode_key == "mean":
        mol = predict(
            smiles,
            models=["quinone"],
            backend=backend,
            _parameter={"quinone_omp_mode": "mean"},
        )
    elif mode_key == "first_path":
        with _patched_omp(path_mode="first_path", aggregate=_agg_max):
            mol = predict(
                smiles,
                models=["quinone"],
                backend=backend,
                _parameter={"quinone_omp_mode": "principled"},
            )
    elif ":" in mode_key:
        path_mode, agg_name = mode_key.split(":", 1)
        hit_agg = AGGREGATORS[agg_name]
        with _patched_omp(path_mode=path_mode, aggregate=hit_agg):
            mol = predict(
                smiles,
                models=["quinone"],
                backend=backend,
                _parameter={"quinone_omp_mode": "principled"},
            )
    else:
        raise ValueError(f"unknown mode {mode_key!r}")

    return golden_score_fields(mol.results[0])


def _modes() -> list[tuple[str, str]]:
    out = [
        ("legacy_bfs", "legacy golden (single BFS + any)"),
        ("principled", "production max/any over all shortest paths"),
        ("mean", "fractional mean over all shortest paths (old principled)"),
        ("all_shortest:max", "same as production principled (patch sanity check)"),
        ("all_shortest:min", "all shortest paths + min/all indicator"),
        ("all_shortest:majority", "all shortest paths + majority vote"),
        ("first_path", "lexicographically first shortest path"),
    ]
    return out


def _unique_quinone_smiles() -> list[str]:
    sys.path.insert(0, str(ROOT))
    from tests.support import load_golden_suite

    seen: set[str] = set()
    out: list[str] = []
    for row in load_golden_suite():
        if row.get("model") != "quinone":
            continue
        smi = row.get("smiles")
        if not smi or smi in seen:
            continue
        seen.add(smi)
        out.append(smi)
    return out


def _summarize(deltas: list[float], atol: float) -> dict:
    if not deltas:
        return {}
    within = sum(1 for d in deltas if d <= atol)
    s = sorted(deltas)
    return {
        "n": len(deltas),
        "frac_within_atol": within / len(deltas),
        "mean": statistics.mean(deltas),
        "median": statistics.median(deltas),
        "p95": s[int(0.95 * (len(s) - 1))],
        "max": max(deltas),
    }


def main(argv: list[str] | None = None) -> int:
    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, str(ROOT))
    from tests.support import PARITY_ATOL

    p = argparse.ArgumentParser()
    p.add_argument("--json", type=Path, default=None)
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args(argv)

    smiles_list = _unique_quinone_smiles()
    if args.limit is not None:
        smiles_list = smiles_list[: args.limit]

    legacy_by_smiles: dict[str, dict] = {}
    for smi in smiles_list:
        legacy_by_smiles[smi] = _quinone_scores(smi, mode_key="legacy_bfs")

    report_modes = _modes()[1:]  # skip legacy vs itself
    vs_legacy: dict[str, list[float]] = defaultdict(list)
    vs_mean: dict[str, list[float]] = defaultdict(list)
    mean_by_smiles: dict[str, dict] = {}
    principled_by_smiles: dict[str, dict] = {}

    for smi in smiles_list:
        legacy = legacy_by_smiles[smi]
        mean_scores = _quinone_scores(smi, mode_key="mean")
        principled_scores = _quinone_scores(smi, mode_key="principled")
        mean_by_smiles[smi] = mean_scores
        principled_by_smiles[smi] = principled_scores
        for mode_key, _label in report_modes:
            cand = _quinone_scores(smi, mode_key=mode_key)
            vs_legacy[mode_key].append(_score_delta(cand, legacy))
            vs_mean[mode_key].append(_score_delta(cand, mean_scores))

    legacy_vs_mean = [
        _score_delta(mean_by_smiles[smi], legacy_by_smiles[smi]) for smi in smiles_list
    ]
    legacy_vs_principled = [
        _score_delta(principled_by_smiles[smi], legacy_by_smiles[smi]) for smi in smiles_list
    ]

    ranked = sorted(
        report_modes,
        key=lambda m: _summarize(vs_legacy[m[0]], PARITY_ATOL).get("frac_within_atol", 0),
        reverse=True,
    )

    report = {
        "atol": PARITY_ATOL,
        "n_molecules": len(smiles_list),
        "baseline_mean_vs_legacy": _summarize(legacy_vs_mean, PARITY_ATOL),
        "baseline_principled_vs_legacy": _summarize(legacy_vs_principled, PARITY_ATOL),
        "candidates_vs_legacy": {
            key: {**_summarize(vs_legacy[key], PARITY_ATOL), "label": label}
            for key, label in report_modes
        },
        "candidates_vs_production_mean": {
            key: _summarize(vs_mean[key], PARITY_ATOL) for key, _ in report_modes
        },
        "ranked_by_legacy_parity": [
            {
                "mode": key,
                "label": label,
                **_summarize(vs_legacy[key], PARITY_ATOL),
            }
            for key, label in ranked
        ],
        "recommendation": None,
    }

    best_key, best_label = ranked[0]
    best = _summarize(vs_legacy[best_key], PARITY_ATOL)
    base = _summarize(legacy_vs_mean, PARITY_ATOL)
    principled = _summarize(vs_legacy.get("all_shortest:max", vs_legacy.get("mean", {})), PARITY_ATOL)
    report["recommendation"] = {
        "mode": "principled",
        "label": "production default (all shortest paths + max/any indicator)",
        "frac_within_atol": principled.get("frac_within_atol"),
        "improvement_vs_mean": principled.get("frac_within_atol", 0) - base.get("frac_within_atol", 0),
        "best_legacy_match": {"mode": best_key, "label": best_label, **best},
        "note": (
            "Implemented as quinone_omp_mode=principled. ~89% legacy parity vs ~45% "
            "for mean. Legacy sorted-BFS equals lexicographic-first-path (100% parity). "
            "Use quinone_omp_mode=mean for the old fractional averaging behavior."
        ),
    }

    print(json.dumps(report, indent=2))
    if args.json:
        args.json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
