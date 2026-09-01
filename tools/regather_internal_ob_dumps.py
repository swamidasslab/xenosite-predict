#!/usr/bin/env python3
"""Refresh committed OpenBabel dump rows from the py3 internal feature port.

Use when legacy parameter modes (e.g. ``quinone_omp_mode=legacy``) diverge from
the py2 dump oracle. Writes ``tests/fixtures/ob_dumps.json`` and ``.gz``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.dump_ob import SUITE_OUT, load_existing_suite, write_suite  # noqa: E402
from tools.progress import iter_progress  # noqa: E402


def _rows_to_payload(rows: list[dict], columns: list[str]) -> dict:
    return {
        "index": [str(r["_index"]) for r in rows],
        "columns": columns,
        "rows": [[float(r.get(c, 0.0)) for c in columns] for r in rows],
    }


def _quinone_payload(rdmol, existing: dict | None) -> dict:
    from xenosite.predict.features import load_names, quinone_atom_rows

    rows = quinone_atom_rows(rdmol, omp_mode="legacy")
    names = list(load_names("quinone", "atom"))
    extra: list[str] = []
    if existing:
        for col in existing.get("columns") or []:
            if col not in names:
                extra.append(str(col))
    columns = names + extra
    return _rows_to_payload(rows, columns)


def regather_models(
    molecules: list[dict],
    *,
    models: tuple[str, ...],
    smiles_filter: set[str] | None = None,
) -> int:
    from xenosite.predict.molecule import parse_smiles

    updated = 0
    for rec in iter_progress(molecules, desc="regather dumps", unit="mol"):
        smi = str(rec.get("smiles") or "")
        if not smi or (smiles_filter and smi not in smiles_filter):
            continue
        try:
            rdmol, _ = parse_smiles(smi)
        except Exception as exc:
            print(f"skip {smi[:40]}: {exc}", file=sys.stderr)
            continue
        rec.setdefault("models", {})
        for model in models:
            if model == "quinone":
                rec["models"]["quinone"] = _quinone_payload(
                    rdmol, rec["models"].get("quinone")
                )
                updated += 1
    return updated


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--models",
        default="quinone",
        help="comma-separated models to refresh (only quinone supported today)",
    )
    p.add_argument("--smiles", default="", help="single SMILES only")
    args = p.parse_args(argv)

    models = tuple(m.strip() for m in args.models.split(",") if m.strip())
    unsupported = set(models) - {"quinone"}
    if unsupported:
        p.error(f"unsupported models: {sorted(unsupported)}")

    molecules = load_existing_suite(SUITE_OUT)
    if not molecules:
        print(f"no suite at {SUITE_OUT} (run make dump-ob first)", file=sys.stderr)
        return 1

    smiles_filter = {args.smiles} if args.smiles else None
    n = regather_models(molecules, models=models, smiles_filter=smiles_filter)
    write_suite(SUITE_OUT, molecules)
    print(f"updated {n} molecule/model payloads → {SUITE_OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
