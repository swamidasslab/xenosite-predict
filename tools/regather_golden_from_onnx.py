#!/usr/bin/env python3
"""Regather golden score rows from ONNX predict with legacy ``_parameter`` modes.

Updates ``tests/fixtures/golden_descriptor_suite.json`` (and optionally
``golden_smiles.json``) so golden parity tests match ONNX + legacy site/OMP.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.support import (  # noqa: E402
    GOLDEN,
    GOLDEN_PARAMETER,
    GOLDEN_SUITE,
    SUITE_MODELS,
    load_descriptor_smiles,
    load_golden,
    load_golden_suite,
    serialize_molecule_results,
    onnx_root,
)
from tools.progress import iter_progress, map_progress, worker_quiet  # noqa: E402
from tools.suite_drift_lib import default_workers  # noqa: E402

_WEIGHTS_ROOT: str = ""
_MODELS: tuple[str, ...] = ()
_FORCE: bool = False


def _load_rows(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def _save_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")


def _golden_kwargs(model: str) -> dict:
    if model in ("ndealk", "isozyme", "quinone"):
        return {"_parameter": dict(GOLDEN_PARAMETER)}
    return {}


def _worker_init(weights_root: str) -> None:
    worker_quiet()
    global _WEIGHTS_ROOT
    _WEIGHTS_ROOT = weights_root


def _capture_one(task: tuple[str, str, str]) -> dict:
    model, smiles, name = task
    from xenosite.predict import predict
    from xenosite.predict.backends.onnx import OnnxBackend

    rec: dict = {
        "model": model,
        "smiles": smiles,
        "name": name or smiles[:40],
        "error": None,
        "results": [],
    }
    try:
        mol = predict(
            smiles,
            models=[model],
            backend=OnnxBackend(_WEIGHTS_ROOT),
            **_golden_kwargs(model),
        )
        rec["smiles"] = mol.smiles
        rec["results"] = serialize_molecule_results(mol)
    except Exception as exc:
        rec["error"] = str(exc)
    return rec


def _merge_row(rows: list[dict], fresh: dict) -> None:
    key = (fresh.get("model"), fresh.get("smiles"))
    for i, row in enumerate(rows):
        if (row.get("model"), row.get("smiles")) == key:
            rows[i] = fresh
            return
    rows.append(fresh)


def _tasks(
    *,
    models: list[str],
    smiles_list: list[str],
    names: dict[str, str],
    existing: set[tuple[str, str]],
) -> list[tuple[str, str, str]]:
    pending: list[tuple[str, str, str]] = []
    for smi in smiles_list:
        for model in models:
            if model in ("phase1", "bioactivation"):
                continue
            key = (model, smi)
            if key in existing and not _FORCE:
                continue
            pending.append((model, smi, names.get(smi) or smi[:40]))
    return pending


def main(argv: list[str] | None = None) -> int:
    global _MODELS, _FORCE
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=GOLDEN_SUITE)
    p.add_argument(
        "--models",
        default="quinone,ndealk,isozyme",
        help="comma-separated models (default: quinone,ndealk,isozyme)",
    )
    p.add_argument("--smiles", default="", help="single SMILES only")
    p.add_argument("--force", action="store_true", help="redo all matching pairs")
    p.add_argument("--include-smoke", action="store_true", help="also refresh golden_smiles.json")
    p.add_argument("--workers", type=int, default=default_workers())
    args = p.parse_args(argv)

    _FORCE = args.force
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    _MODELS = tuple(models)

    from tests.support import golden_name_by_smiles

    names = golden_name_by_smiles()
    if args.smiles:
        smiles_list = [args.smiles]
    else:
        smiles_list = load_descriptor_smiles()

    rows = _load_rows(args.out)
    existing = {(r.get("model"), r.get("smiles")) for r in rows}
    pending = _tasks(
        models=models,
        smiles_list=smiles_list,
        names=names,
        existing=existing,
    )
    if not pending:
        print(f"nothing pending for {args.out}")
    else:
        weights_root = str(onnx_root())
        workers = max(1, args.workers)
        t0 = time.time()
        if workers == 1:
            _worker_init(weights_root)
            for task in iter_progress(pending, desc="regather golden", unit="pair"):
                rec = _capture_one(task)
                if rec.get("error"):
                    print(f"ERR {rec['model']} {rec['name']}: {rec['error']}", file=sys.stderr)
                else:
                    _merge_row(rows, rec)
        else:
            with ProcessPoolExecutor(
                max_workers=workers,
                initializer=_worker_init,
                initargs=(weights_root,),
            ) as pool:
                for rec in map_progress(
                    _capture_one,
                    pending,
                    pool=pool,
                    desc=f"regather golden ({workers}w)",
                    unit="pair",
                ):
                    if rec.get("error"):
                        print(
                            f"ERR {rec['model']} {rec['name']}: {rec['error']}",
                            file=sys.stderr,
                        )
                    else:
                        _merge_row(rows, rec)
        _save_rows(args.out, rows)
        print(f"updated {len(pending)} pairs in {args.out} ({time.time()-t0:.1f}s)")

    if args.include_smoke and GOLDEN.is_file():
        smoke = _load_rows(GOLDEN)
        smoke_existing = {(r.get("model"), r.get("smiles")) for r in smoke}
        smoke_models = models or [m for m in SUITE_MODELS if m not in ("phase1", "bioactivation")]
        smoke_pending = _tasks(
            models=smoke_models,
            smiles_list=[g.get("smiles") for g in load_golden() if g.get("smiles")],
            names=names,
            existing=smoke_existing,
        )
        if smoke_pending:
            weights_root = str(onnx_root())
            _worker_init(weights_root)
            for task in iter_progress(smoke_pending, desc="regather smoke golden", unit="pair"):
                rec = _capture_one(task)
                if not rec.get("error"):
                    _merge_row(smoke, rec)
            _save_rows(GOLDEN, smoke)
            print(f"updated smoke golden {GOLDEN}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
