#!/usr/bin/env python3
"""Diagnostic: compare legacy bioactivation golden / live vs doctest anchors.

Not a pytest gate. Forest + ported models are expected to drift from metabolite1;
this records MBS / top-PBS deltas for API-option discussion.

  uv run python tools/bioactivation_e2e_diag.py
  uv run python tools/bioactivation_e2e_diag.py --url http://127.0.0.1:8099
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Doctest anchors from libridass.bioactivation1.base
_DOCTEST = {
    "C1=C(C=CC=C1)C=C": {"name": "styrene", "mbs": 0.8432, "top_pbs": 0.70564},
    "C(C(C=CCl)(C#C)O)C": {"name": "ethchlorvynol", "mbs": 0.80764},
}


def _mbs_from_results(results: list[dict]) -> float | None:
    for r in results:
        if r.get("model") == "bioactivation" and "mol" in r:
            return float(r["mol"])
    return None


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url", default="", help="if set, also hit legacy-test-api live")
    p.add_argument(
        "--golden",
        type=Path,
        default=ROOT / "tests" / "fixtures" / "golden_smiles.json",
    )
    args = p.parse_args(argv)

    golden = json.loads(args.golden.read_text()) if args.golden.is_file() else []
    by_smi = {
        r["smiles"]: r
        for r in golden
        if r.get("model") == "bioactivation"
    }

    print("doctest vs golden_smiles (legacy Docker capture)")
    for smi, ref in _DOCTEST.items():
        # golden may canonicalize SMILES
        hit = by_smi.get(smi)
        if hit is None:
            for gs, row in by_smi.items():
                if ref["name"] in (row.get("name") or ""):
                    hit = row
                    smi = gs
                    break
        mbs_g = _mbs_from_results(hit["results"]) if hit else None
        print(
            f"  {ref['name']}: doctest_MBS={ref['mbs']} golden_MBS={mbs_g} "
            f"delta={None if mbs_g is None else round(mbs_g - ref['mbs'], 5)}"
        )

    if args.url:
        from xenosite.predict import predict
        from xenosite.predict.backends.legacy import LegacyTestBackend

        be = LegacyTestBackend(args.url, timeout=600.0)
        print(f"live legacy-test-api @ {args.url}")
        for smi, ref in _DOCTEST.items():
            mol = predict(smi, models=["bioactivation"], backend=be)
            res = next(r for r in mol.results if r.model == "bioactivation")
            print(
                f"  {ref['name']}: live_MBS={round(float(res.mol), 5)} "
                f"n_metabolites={len(res.metabolite or [])} "
                f"delta_vs_doctest={round(float(res.mol) - ref['mbs'], 5)}"
            )

    print(
        "Note: Tier C drift (forest vs metabolite1) is expected until a "
        "pipeline runner is chosen; head ONNX parity is separate (Tier A/B)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
