#!/usr/bin/env python3
"""Capture small XenoNet graphs from legacy-test-api POST /xenonet.

Writes ``tests/fixtures/xenonet/*.json``. Needs ``make legacy-test-api``.
Bioactivation PBS/MBS is not captured here.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tests" / "fixtures" / "xenonet"

JOBS = (
    {
        "name": "ethane_depth1.json",
        "smiles": "CC",
        "depth_limit": 1,
        "beam_width": 1000,
        "max_time": 5,
    },
    {
        "name": "ethylene_depth1.json",
        "smiles": "C=C",
        "depth_limit": 1,
        "beam_width": 1000,
        "max_time": 5,
    },
    {
        "name": "ethane_depth2.json",
        "smiles": "CC",
        "depth_limit": 2,
        "beam_width": 20,
        "max_time": 5,
    },
    {
        "name": "ethane_to_ethanol.json",
        "smiles": "CC",
        "depth_limit": 1,
        "beam_width": 1000,
        "max_time": 5,
        "targets": ["CCO"],
    },
)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url", default="http://127.0.0.1:8099")
    p.add_argument("--timeout", type=float, default=300.0)
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    with httpx.Client(timeout=args.timeout) as client:
        h = client.get(f"{args.url.rstrip('/')}/health")
        h.raise_for_status()
        for job in JOBS:
            body = {k: v for k, v in job.items() if k != "name"}
            r = client.post(f"{args.url.rstrip('/')}/xenonet", json=body)
            r.raise_for_status()
            data = r.json()
            if data.get("error") and not data.get("edges"):
                print(f"{job['name']}: {data['error']}", file=sys.stderr)
                return 1
            data["smiles"] = job["smiles"]
            data["depth_limit"] = job["depth_limit"]
            data["beam_width"] = job["beam_width"]
            if job.get("targets"):
                data["targets"] = job["targets"]
            path = OUT / job["name"]
            path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
            print(f"wrote {path} ({len(data.get('edges') or [])} edges)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
