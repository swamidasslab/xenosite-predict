#!/usr/bin/env python3
"""Hash-compare vendored NN and feature trees (sibling checkout or weights/legacy).

Writes a summary to stdout; used to refresh docs/vendored-diffs.md.
Does not copy the old tree into src/.
"""

from __future__ import annotations

import argparse
import hashlib
from collections import defaultdict
from pathlib import Path

NN = {
    "epoxidation": "epoxidation1/NN",
    "quinone": "quinone1/NN",
    "reactivity": "reactivity1/NN",
    "ugt": "ugt1/NN",
    "ndealk": "ndealk1/NN",
    "metabolism": "metabolism1/xenosite/NN",
}

FEAT = {
    "epoxidation": "epoxidation1/topological_descriptors",
    "quinone": "quinone1/topological_descriptors",
    "reactivity": "reactivity1/topological_descriptors",
    "ugt": "ugt1/xenosite/descriptor",
    "ndealk": "ndealk1/scripts",
    "metabolism": "metabolism1/xenosite/descriptor",
    "shared_topo": "topological_descriptors",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def py_files(root: Path) -> dict[str, str]:
    out = {}
    if not root.is_dir():
        return out
    for p in sorted(root.rglob("*.py")):
        if "__pycache__" in p.parts:
            continue
        out[str(p.relative_to(root))] = sha256(p)
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--root",
        type=Path,
        default=Path("../xenosite-legacy/src/libridass"),
    )
    args = p.parse_args(argv)
    root = args.root
    if not root.is_dir():
        alt = Path("weights/legacy/sibling_libridass")
        if alt.is_dir():
            root = alt
        else:
            print(f"no libridass root at {args.root} or {alt}")
            return 1

    print(f"# vendored diffs vs {root.resolve()}\n")
    print("## NN layers.py")
    hashes = {}
    for name, rel in NN.items():
        path = root / rel / "layers.py"
        h = sha256(path)[:16] if path.is_file() else "MISSING"
        hashes[name] = h
        print(f"- {name}: `{h}` `{rel}/layers.py`")
    groups = defaultdict(list)
    for n, h in hashes.items():
        groups[h].append(n)
    print("\nIdentical layers.py groups:")
    for h, names in groups.items():
        print(f"- {h}: {', '.join(names)}")

    print("\n## NN package file sets")
    sets = {n: tuple(sorted(py_files(root / rel).items())) for n, rel in NN.items()}
    print("- epoxidation == ugt == ndealk:", sets["epoxidation"] == sets["ugt"] == sets["ndealk"])
    print("- quinone vs reactivity layers.py:", hashes["quinone"] == hashes["reactivity"])

    print("\n## Feature files (sha256[:16])")
    for name, rel in FEAT.items():
        d = root / rel
        for fn in (
            "bond.py",
            "atom.py",
            "molecule.py",
            "atom_pair.py",
            "topo.py",
            "top.py",
            "mopac.py",
            "bond_desc.py",
            "Heuristic_desc.py",
        ):
            path = d / fn
            if path.is_file():
                print(f"- {name}/{fn}: `{sha256(path)[:16]}`")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
