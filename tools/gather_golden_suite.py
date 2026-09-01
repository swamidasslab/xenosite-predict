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


def _load_out(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def _save_out(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")


def _name_for(smiles: str, names: dict[str, str]) -> str:
    return names.get(smiles) or smiles[:40]


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
        "--merge-smoke",
        action="store_true",
        help="also copy rows from golden_smiles.json at the end",
    )
    args = p.parse_args(argv)

    from xenosite.predict import predict
    from xenosite.predict.backends.legacy import LegacyTestBackend

    try:
        from tests.support import golden_name_by_smiles

        names = golden_name_by_smiles()
    except Exception:
        names = {}

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    smiles_list = [args.smiles] if args.smiles else load_descriptor_smiles()
    if args.limit:
        smiles_list = smiles_list[: args.limit]

    backend = LegacyTestBackend(args.url)
    if not backend.health():
        print(f"legacy-test-api not healthy at {args.url}", file=sys.stderr)
        return 1

    rows = _load_out(args.out)
    done = {(r.get("model"), r.get("smiles")) for r in rows}
    added = 0
    errors: list[str] = []

    for i, smiles in enumerate(smiles_list):
        for model in models:
            key = (model, smiles)
            if key in done and not args.force:
                continue
            if args.force and key in done:
                rows = [r for r in rows if (r.get("model"), r.get("smiles")) != key]
                done.discard(key)
            t0 = time.time()
            try:
                mol = predict(smiles, models=[model], backend=backend)
                rec = {
                    "smiles": mol.smiles,
                    "name": _name_for(smiles, names),
                    "model": model,
                    "results": serialize_molecule_results(mol),
                }
                rows.append(rec)
                done.add(key)
                added += 1
                _save_out(args.out, rows)
                dt = time.time() - t0
                print(f"[{added}] {model} {rec['name'][:32]} ({dt:.1f}s)", flush=True)
            except Exception as exc:
                msg = f"{model} {smiles[:32]}: {exc}"
                errors.append(msg)
                print(f"ERROR {msg}", file=sys.stderr, flush=True)

    if args.merge_smoke:
        smoke = load_golden_suite(merge_smoke=False)
        from tests.support import load_golden

        for r in load_golden():
            key = (r.get("model"), r.get("smiles"))
            if key not in done:
                rows.append(r)
                done.add(key)
        _save_out(args.out, rows)

    print(f"wrote {args.out} total={len(rows)} new={added} errors={len(errors)}")
    if errors:
        print("first errors:", file=sys.stderr)
        for e in errors[:10]:
            print(f"  {e}", file=sys.stderr)
        return 2 if not added else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
