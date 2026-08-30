#!/usr/bin/env python3
"""Extract score-only golden fixtures from xenosite-api ``.test_data/test_data.mpk.gz``.

Writes ``tests/fixtures/golden_smiles.json``. Compare scores on RDKit indices, not
msgpack identity (adapter quirks).
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

MODEL_MAP = {
    "EpoxidationModel": "epoxidation",
    "ReactivityModel": "reactivity",
    "QuinoneModel": "quinone",
    "NdealkModel": "ndealk",
    "UgtModel": "ugt",
    "IsozymeModel": "isozyme",
    "Phase1Model": "phase1",
}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--src",
        type=Path,
        default=Path("../.test_data/test_data.mpk.gz"),
    )
    p.add_argument("--out", type=Path, default=Path("tests/fixtures/golden_smiles.json"))
    args = p.parse_args()
    if not args.src.is_file():
        print(f"missing {args.src}")
        return 1
    import msgpack

    with gzip.open(args.src, "rb") as f:
        rows = msgpack.load(f, raw=False, strict_map_key=False)

    out = []
    for row in rows:
        raw = row.get("endpoint_result")
        if not raw:
            continue
        content = raw[0] if isinstance(raw, (list, tuple)) else raw
        if not content:
            continue
        mol = msgpack.unpackb(content, raw=False, strict_map_key=False)
        if not isinstance(mol, dict):
            continue
        rec = {
            "smiles": mol.get("smiles") or row.get("example"),
            "name": row.get("name"),
            "model": MODEL_MAP.get(row.get("model"), row.get("model")),
            "results": [],
        }
        for r in mol.get("results") or []:
            rec["results"].append(
                {
                    "model": r.get("model"),
                    "version": r.get("version"),
                    "mol": r.get("mol"),
                    "atom": r.get("atom"),
                    "bond": r.get("bond"),
                    "pair": r.get("pair"),
                    "pair_idx": r.get("pair_idx"),
                }
            )
        out.append(rec)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.out} n={len(out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
