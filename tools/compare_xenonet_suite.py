#!/usr/bin/env python3
"""Compare ONNX XenoNet vs legacy-test-api POST /xenonet on the descriptor suite.

Writes incremental JSON to ``artifacts/xenonet_suite_parity.json`` and a mismatch
report on stdout. Progress bar on tty/stderr. Needs ``make legacy-test-api``.

Default: 327 descriptor SMILES, depth 1, beam 1000, 8 workers.
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.support import (  # noqa: E402
    PARITY_ATOL,
    load_descriptor_smiles,
    onnx_root,
    onnx_weights_present,
)
from tools.progress import map_progress, worker_quiet  # noqa: E402

OUT_DEFAULT = ROOT / "artifacts" / "xenonet_suite_parity.json"

_URL = ""
_TIMEOUT = 180.0
_DEPTH = 1
_BEAM = 1000
_ATOL = PARITY_ATOL


def _edge_key(e: dict) -> tuple:
    return (e["parent"], e["child"], e["rule"], tuple(e.get("site") or []))


def _inchikey(smiles: str) -> str:
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return f"raw:{smiles}"
    try:
        key = Chem.MolToInchiKey(mol)
    except Exception:
        key = None
    if not key:
        return f"smi:{Chem.MolToSmiles(mol, canonical=True)}"
    return key


def _struct_key(e: dict) -> tuple:
    return (
        _inchikey(e["parent"]),
        _inchikey(e["child"]),
        e["rule"],
        tuple(e.get("site") or []),
    )


def _compare(got: dict, py2: dict) -> dict:
    ge = {_edge_key(e) for e in got.get("edges") or []}
    ee = {_edge_key(e) for e in py2.get("edges") or []}
    extra = sorted(ge - ee)
    missing = sorted(ee - ge)
    gs = {_struct_key(e) for e in got.get("edges") or []}
    es = {_struct_key(e) for e in py2.get("edges") or []}
    extra_s = sorted(gs - es)
    missing_s = sorted(es - gs)
    w_got = {_edge_key(e): float(e["weight"]) for e in got.get("edges") or []}
    w_exp = {_edge_key(e): float(e["weight"]) for e in py2.get("edges") or []}
    max_dw = 0.0
    n_w = 0
    for k in ge & ee:
        n_w += 1
        max_dw = max(max_dw, abs(w_got[k] - w_exp[k]))
    w_got_s = {_struct_key(e): float(e["weight"]) for e in got.get("edges") or []}
    w_exp_s = {_struct_key(e): float(e["weight"]) for e in py2.get("edges") or []}
    max_dw_s = 0.0
    for k in gs & es:
        max_dw_s = max(max_dw_s, abs(w_got_s[k] - w_exp_s[k]))
    n_got = {n["smiles"]: n.get("metabolism_score") for n in got.get("nodes") or []}
    n_exp = {n["smiles"]: n.get("metabolism_score") for n in py2.get("nodes") or []}
    max_dn = 0.0
    for smi, s in n_exp.items():
        if s is None or n_got.get(smi) is None:
            continue
        max_dn = max(max_dn, abs(float(n_got[smi]) - float(s)))
    n_got_i = {
        _inchikey(n["smiles"]): n.get("metabolism_score") for n in got.get("nodes") or []
    }
    n_exp_i = {
        _inchikey(n["smiles"]): n.get("metabolism_score") for n in py2.get("nodes") or []
    }
    max_dn_s = 0.0
    for key, s in n_exp_i.items():
        if s is None or n_got_i.get(key) is None:
            continue
        max_dn_s = max(max_dn_s, abs(float(n_got_i[key]) - float(s)))
    from collections import Counter

    rs_got = Counter(
        (e["rule"], tuple(e.get("site") or [])) for e in got.get("edges") or []
    )
    rs_exp = Counter(
        (e["rule"], tuple(e.get("site") or [])) for e in py2.get("edges") or []
    )
    rs_ok = rs_got == rs_exp
    rw_got = Counter(
        (e["rule"], round(float(e["weight"]), 6)) for e in got.get("edges") or []
    )
    rw_exp = Counter(
        (e["rule"], round(float(e["weight"]), 6)) for e in py2.get("edges") or []
    )
    rw_ok = rw_got == rw_exp
    smiles_ok = not extra and not missing and max_dw <= _ATOL and max_dn <= _ATOL
    struct_ok = (
        not extra_s and not missing_s and max_dw_s <= _ATOL and max_dn_s <= _ATOL
    )
    if smiles_ok:
        kind = "ok"
    elif struct_ok or rs_ok:
        kind = "smiles_form"
    elif rw_ok:
        kind = "site_shift"
    else:
        kind = "topology"
    rec = {
        "ok": smiles_ok and not py2.get("error"),
        "struct_ok": struct_ok and not py2.get("error"),
        "rs_ok": rs_ok,
        "rw_ok": rw_ok,
        "kind": kind,
        "extra": extra[:8],
        "missing": missing[:8],
        "extra_struct": extra_s[:8],
        "missing_struct": missing_s[:8],
        "n_extra": len(ge - ee),
        "n_missing": len(ee - ge),
        "n_extra_struct": len(gs - es),
        "n_missing_struct": len(es - gs),
        "n_py2_edges": len(ee),
        "n_onnx_edges": len(ge),
        "max_dw": max_dw,
        "max_dn": max_dn,
        "max_dw_struct": max_dw_s,
        "max_dn_struct": max_dn_s,
        "shared_edges": n_w,
    }
    if py2.get("error"):
        rec["py2_error"] = py2["error"]
        rec["ok"] = False
        rec["struct_ok"] = False
        rec["kind"] = "py2_error"
    return rec


def _worker_init(url: str, timeout: float, depth: int, beam: int, atol: float) -> None:
    worker_quiet()
    global _URL, _TIMEOUT, _DEPTH, _BEAM, _ATOL
    _URL = url
    _TIMEOUT = timeout
    _DEPTH = depth
    _BEAM = beam
    _ATOL = atol


def _one(smiles: str) -> dict:
    import httpx
    from xenosite.predict.backends.onnx import OnnxBackend
    from xenosite.predict.v1.xenonet import build_network
    from xenosite.predict.weights import default_cache_dir

    rec: dict = {"smiles": smiles, "ok": False, "error": None}
    try:
        with httpx.Client(timeout=_TIMEOUT) as client:
            r = client.post(
                f"{_URL.rstrip('/')}/xenonet",
                json={
                    "smiles": smiles,
                    "depth_limit": _DEPTH,
                    "beam_width": _BEAM,
                    "max_time": 5,
                    "weighted": True,
                },
            )
            r.raise_for_status()
            py2 = r.json()
        root = onnx_root()
        if not onnx_weights_present("phase1"):
            cache = default_cache_dir()
            if cache is None:
                rec["error"] = "no phase1 ONNX"
                return rec
            root = cache
        got = build_network(
            smiles,
            depth_limit=_DEPTH,
            beam_width=_BEAM,
            backend=OnnxBackend(root),
            scoring="0",
        ).to_dict()
        rec.update(_compare(got, py2))
    except Exception as exc:
        rec["error"] = str(exc)
        rec["ok"] = False
    return rec


def _load_done(path: Path) -> dict[str, dict]:
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data.get("rows") or []
    return {r["smiles"]: r for r in rows if r.get("smiles")}


def _save(path: Path, args, rows: list[dict]) -> None:
    payload = {
        "depth_limit": args.depth,
        "beam_width": args.beam,
        "atol": args.atol,
        "n": len(rows),
        "rows": rows,
    }
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    tmp.replace(path)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url", default="http://127.0.0.1:8099")
    p.add_argument("--timeout", type=float, default=180.0)
    p.add_argument("--depth", type=int, default=1)
    p.add_argument("--beam", type=int, default=1000)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--limit", type=int, default=0, help="Cap molecule count (0 = all)")
    p.add_argument("--force", action="store_true")
    p.add_argument("--out", type=Path, default=OUT_DEFAULT)
    p.add_argument("--atol", type=float, default=PARITY_ATOL)
    p.add_argument("--save-every", type=int, default=10)
    args = p.parse_args()

    import httpx

    health = httpx.get(f"{args.url.rstrip('/')}/health", timeout=10.0)
    health.raise_for_status()

    smiles = load_descriptor_smiles()
    if args.limit:
        smiles = smiles[: args.limit]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    done = {} if args.force else _load_done(args.out)
    todo = [s for s in smiles if s not in done]
    print(
        f"suite={len(smiles)} done={len(done)} todo={len(todo)} "
        f"depth={args.depth} beam={args.beam} workers={args.workers} atol={args.atol}",
        flush=True,
    )

    rows = [done[s] for s in smiles if s in done]
    since_save = 0

    def _on_result(rec, _bar) -> None:
        nonlocal since_save
        rows.append(rec)
        since_save += 1
        if args.save_every and since_save >= args.save_every:
            _save(args.out, args, rows)
            since_save = 0

    if todo:
        with ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=_worker_init,
            initargs=(args.url, args.timeout, args.depth, args.beam, args.atol),
        ) as pool:
            map_progress(
                _one,
                todo,
                pool=pool,
                desc="xenonet vs py2",
                unit="mol",
                on_result=_on_result,
            )
        _save(args.out, args, rows)

    ok = sum(1 for r in rows if r.get("ok"))
    struct_ok = sum(1 for r in rows if r.get("struct_ok"))
    rs_ok = sum(1 for r in rows if r.get("rs_ok"))
    rw_ok = sum(1 for r in rows if r.get("rw_ok"))
    err = [r for r in rows if r.get("error") or r.get("py2_error")]
    bad = [r for r in rows if not r.get("ok")]
    kinds: dict[str, int] = {}
    for r in rows:
        kinds[r.get("kind") or "unknown"] = kinds.get(r.get("kind") or "unknown", 0) + 1
    print(
        f"ok={ok}/{len(rows)} struct_ok={struct_ok}/{len(rows)} "
        f"rs_ok={rs_ok}/{len(rows)} rw_ok={rw_ok}/{len(rows)} "
        f"mismatches={len(bad)} errors={len(err)} kinds={kinds}",
        flush=True,
    )
    for r in bad[:40]:
        smi = r.get("smiles") or ""
        print(
            f"  {r.get('kind')} {smi[:60]} extra={r.get('n_extra')} missing={r.get('n_missing')} "
            f"extra_s={r.get('n_extra_struct')} missing_s={r.get('n_missing_struct')} "
            f"max_dw={r.get('max_dw_struct')} max_dn={r.get('max_dn_struct')} "
            f"err={r.get('error') or r.get('py2_error')}",
            flush=True,
        )
        if r.get("kind") == "topology" and r.get("extra_struct"):
            print(f"    extra_struct {r['extra_struct'][:2]}", flush=True)
        if r.get("kind") == "topology" and r.get("missing_struct"):
            print(f"    missing_struct {r['missing_struct'][:2]}", flush=True)
        if r.get("kind") != "topology" and r.get("extra"):
            print(f"    extra sample {r['extra'][:2]}", flush=True)
        if r.get("kind") != "topology" and r.get("missing"):
            print(f"    missing sample {r['missing'][:2]}", flush=True)
    if len(bad) > 40:
        print(f"  ... {len(bad) - 40} more", flush=True)
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
