#!/usr/bin/env python3
"""Convert legacy numpy-NN pickles to ONNX (via optional safetensors dump).

Not installed in the wheel. Convert-only deps: ``uv run --group convert``.

The numpy NN is a layered graph (WindowedInputLayer | FullyConnectedLayer |
LogisticLayer | … | CrossEntropyError). Conversion recovers topology, not only
tensors, and must run **inside the Python 2.7 image** for unpickling.

This host script:
1. Writes a py2 dump helper into the extract (or invokes docker).
2. Reads dumped JSON + tensors.
3. Builds one ONNX file per head under ``weights/onnx/<model>/``.
4. Writes committed feature-name JSON next to ``src/xenosite/predict/features/``
   when TSV headers are present.

Do not fake ONNX files when dump/convert fails. Phase1 TF1 molecularNN pickles
convert on the host (no TensorFlow): ``tools/convert_phase1.py``. Bioactivation
is a pipeline, not a single graph.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

MODELS = {
    "epoxidation": {
        "legacy": "epoxidation1",
        "heads": {
            "bond": "models/BOND.model",
            "mol": "models/MOLECULE.model",
        },
        "tsv": {
            "bond": "models/BOND_TRAINING.tsv",
            "mol": "models/MOLECULE_TRAINING.tsv",
        },
        "two_stage": True,
    },
    "quinone": {
        "legacy": "quinone1",
        "heads": {
            "atom": "models/atom_model.pyp",
            "pair": "models/atom_pair_model.pyp",
            "mol": "models/molecule_model.pyp",
        },
        "tsv": {
            "atom": "data/atom_training.tsv",
            "pair": "data/atom_pair_training.tsv",
            "mol": "data/molecule_training.tsv",
        },
        "two_stage": True,
    },
    "reactivity": {
        "legacy": "reactivity1",
        "heads": {
            "atom": "models/ATOM.model",
            "mol": "models/MOLECULE.model",
        },
        "tsv": {
            "atom": "models/ATOM_TRAINING.tsv",
            "mol": "models/MOLECULE_TRAINING.tsv",
        },
        "two_stage": True,
    },
    "ugt": {
        "legacy": "ugt1",
        "heads": {"atom": "models/ATOM.model"},
        "tsv": {"atom": "models/DESC.tsv"},
    },
    "ndealk": {
        "legacy": "ndealk1",
        "heads": {"bond": "models/BOND.model"},
        "tsv": {},
    },
}

# Public python:2.7-slim (linux/amd64) + numpy/scipy. Does not need the WashU registry.
DUMP_IMAGE = os.environ.get("XENOSITE_PY2_DUMP_IMAGE", "xenosite-predict-py2:dump")
DUMP_DOCKERFILE = Path(__file__).resolve().parent / "py2-dump" / "Dockerfile"
DUMP_HELPER = Path(__file__).resolve().parent / "py2-dump" / "dump_nn_py2.py"
SIBLING_LIBRIDASS = Path(__file__).resolve().parents[1] / "weights" / "legacy" / "sibling_libridass"
ALT_SIBLING = Path(__file__).resolve().parents[2] / "xenosite-legacy" / "src" / "libridass"


def tsv_columns(path: Path) -> list[str] | None:
    if not path.is_file():
        return None
    import csv

    with path.open(newline="") as f:
        row = next(csv.reader(f, delimiter="\t"))
    # ID + features + TARGET
    cols = [c for c in row if c not in ("ID", "TARGET", "weight")]
    return cols


def build_onnx(meta: dict, weights: bytes, dest: Path) -> None:
    import numpy as np
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    W = np.frombuffer(weights, dtype=np.float64)
    layers = meta["layers"]
    n_in = int(layers[0].get("n_out") or layers[0].get("n_in") or meta["I"])
    if layers[0]["class"] == "AbutLayer":
        n_in = int(layers[0].get("n_in") or meta["I"])
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [None, n_in])
    nodes: list = []
    inits: list = []

    def emit(layers_local: list, cursor: int, cur_name: str, prefix: str) -> tuple[str, int]:
        for i, layer in enumerate(layers_local):
            cls = layer["class"]
            tag = f"{prefix}{i}"
            if cls == "AbutLayer":
                children = layer.get("children") or []
                parts = []
                offset = 0
                for ci, child in enumerate(children):
                    n_in_c = int(child[0]["n_in"]) if child else 0
                    starts = numpy_helper.from_array(
                        np.array([0, offset], dtype=np.int64), name=f"{tag}s{ci}"
                    )
                    ends = numpy_helper.from_array(
                        np.array([np.iinfo(np.int64).max, offset + n_in_c], dtype=np.int64),
                        name=f"{tag}e{ci}",
                    )
                    axes = numpy_helper.from_array(
                        np.array([0, 1], dtype=np.int64), name=f"{tag}a{ci}"
                    )
                    inits.extend([starts, ends, axes])
                    piece = f"{tag}p{ci}"
                    nodes.append(
                        helper.make_node(
                            "Slice",
                            [cur_name, f"{tag}s{ci}", f"{tag}e{ci}", f"{tag}a{ci}"],
                            [piece],
                        )
                    )
                    piece, cursor = emit(child, cursor, piece, f"{tag}c{ci}_")
                    parts.append(piece)
                    offset += n_in_c
                out = f"{tag}cat"
                nodes.append(helper.make_node("Concat", parts, [out], axis=1))
                cur_name = out
            elif cls in ("WindowedInputLayer", "NormalizedInputLayer"):
                center = np.asarray(layer["center"], dtype=np.float32).reshape(1, -1)
                spread = np.asarray(layer["spread"], dtype=np.float32).reshape(1, -1)
                inits.append(numpy_helper.from_array(center, name=f"c{tag}"))
                inits.append(numpy_helper.from_array(spread, name=f"s{tag}"))
                sub = f"sub{tag}"
                nodes.append(helper.make_node("Sub", [cur_name, f"c{tag}"], [sub]))
                out = f"n{tag}"
                nodes.append(helper.make_node("Div", [sub, f"s{tag}"], [out]))
                cur_name = out
            elif cls == "FullyConnectedLayer":
                n_w = int(layer["n_w"])
                n_out, n_in_l = int(layer["n_out"]), int(layer["n_in"])
                w = W[cursor : cursor + n_w].astype(np.float32).reshape(n_out, n_in_l)
                cursor += n_w
                inits.append(numpy_helper.from_array(w, name=f"W{tag}"))
                out = f"n{tag}"
                nodes.append(
                    helper.make_node(
                        "Gemm", [cur_name, f"W{tag}"], [out], transB=1, alpha=1.0, beta=0.0
                    )
                )
                cur_name = out
            elif cls in ("LogisticLayer", "CrossEntropyError", "TanhLayer"):
                n_w = int(layer["n_w"])
                b = W[cursor : cursor + n_w].astype(np.float32).reshape(1, -1)
                cursor += n_w
                inits.append(numpy_helper.from_array(b, name=f"b{tag}"))
                add = f"add{tag}"
                nodes.append(helper.make_node("Add", [cur_name, f"b{tag}"], [add]))
                out = f"n{tag}"
                act = "Tanh" if cls == "TanhLayer" else "Sigmoid"
                nodes.append(helper.make_node(act, [add], [out]))
                cur_name = out
            elif cls == "GaussianError":
                std = np.asarray(layer["std"], dtype=np.float32).reshape(1, -1)
                ave = np.asarray(layer["ave"], dtype=np.float32).reshape(1, -1)
                inits.append(numpy_helper.from_array(std, name=f"std{tag}"))
                inits.append(numpy_helper.from_array(ave, name=f"ave{tag}"))
                mul = f"mul{tag}"
                nodes.append(helper.make_node("Mul", [cur_name, f"std{tag}"], [mul]))
                out = f"n{tag}"
                nodes.append(helper.make_node("Add", [mul, f"ave{tag}"], [out]))
                cur_name = out
            elif cls in ("OutputLayer", "EmptyLayer"):
                continue
            else:
                print(f"warning: skipping unknown layer {cls}", file=sys.stderr)
                continue
        return cur_name, cursor

    cur_name, cursor = emit(layers, 0, "X", "")
    if cursor != len(W):
        print(f"warning: consumed {cursor}/{len(W)} weights", file=sys.stderr)
    if cur_name != "Y":
        nodes.append(helper.make_node("Identity", [cur_name], ["Y"]))
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [None, None])
    graph = helper.make_graph(nodes, "xenosite_nn", [X], [Y], inits)
    model = helper.make_model(graph, producer_name="xenosite-predict", opset_imports=[helper.make_opsetid("", 13)])
    dest.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(dest))
    print("wrote", dest)


def libridass_root() -> Path | None:
    if SIBLING_LIBRIDASS.is_dir():
        return SIBLING_LIBRIDASS
    if ALT_SIBLING.is_dir():
        return ALT_SIBLING
    return None


def ensure_dump_image() -> bool:
    inspect = subprocess.run(
        ["docker", "image", "inspect", DUMP_IMAGE],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if inspect.returncode == 0:
        return True
    if not DUMP_DOCKERFILE.is_file():
        print(f"missing {DUMP_DOCKERFILE}", file=sys.stderr)
        return False
    print(f"building {DUMP_IMAGE} from {DUMP_DOCKERFILE}")
    r = subprocess.run(
        [
            "docker",
            "build",
            "--platform",
            "linux/amd64",
            "-t",
            DUMP_IMAGE,
            str(DUMP_DOCKERFILE.parent),
        ]
    )
    return r.returncode == 0


def dump_via_docker(
    pickle_path: Path,
    pkg_root: Path,
    dump_dir: Path,
    n_in: int | None = None,
) -> tuple[Path, Path] | None:
    if not ensure_dump_image():
        return None
    dump_dir.mkdir(parents=True, exist_ok=True)
    out_json = dump_dir / (pickle_path.stem + ".graph.json")
    out_w = dump_dir / (pickle_path.stem + ".weights.bin")
    x_json = dump_dir / (pickle_path.stem + ".x.json")
    y_json = dump_dir / (pickle_path.stem + ".y.json")
    # Random matrix for replay tests (patterns x features). Dim filled after dump
    # if n_in is known; otherwise dump first then a second predict pass.
    extra: list[str] = []
    if n_in:
        import numpy as np

        rng = np.random.default_rng(20260829)
        x = rng.normal(size=(8, n_in)).tolist()
        x_json.write_text(json.dumps(x))
        extra = [f"/dump/{x_json.name}", f"/dump/{y_json.name}"]
    cmd = [
        "docker",
        "run",
        "--rm",
        "--platform",
        "linux/amd64",
        "-v",
        f"{pickle_path.parent.resolve()}:/pkl:ro",
        "-v",
        f"{pkg_root.resolve()}:/pkg:ro",
        "-v",
        f"{dump_dir.resolve()}:/dump",
        "-v",
        f"{DUMP_HELPER.resolve()}:/work/dump_nn_py2.py:ro",
        DUMP_IMAGE,
        f"/pkl/{pickle_path.name}",
        "/pkg",
        f"/dump/{out_json.name}",
        f"/dump/{out_w.name}",
        *extra,
    ]
    try:
        subprocess.run(cmd, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        print(f"docker dump failed for {pickle_path}: {exc}", file=sys.stderr)
        return None
    if out_json.is_file() and out_w.is_file():
        if not extra:
            meta = json.loads(out_json.read_text())
            second = dump_via_docker(
                pickle_path, pkg_root, dump_dir, n_in=int(meta["I"])
            )
            return second or (out_json, out_w)
        return out_json, out_w
    return None


def write_feature_json(model: str, head: str, names: list[str], features_dir: Path) -> None:
    dest = features_dir / f"{model}_{head}_names.json"
    dest.write_text(json.dumps({"names": names}, indent=2) + "\n", encoding="utf-8")
    print("wrote", dest)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src", type=Path, default=Path("weights/legacy"))
    p.add_argument("--out", type=Path, default=Path("weights/onnx"))
    p.add_argument("--model", default=None)
    p.add_argument(
        "--features-dir",
        type=Path,
        default=Path("src/xenosite/predict/features"),
    )
    args = p.parse_args(argv)

    wanted = [args.model] if args.model else [*MODELS, "phase1"]
    any_ok = False
    tools_dir = Path(__file__).resolve().parent
    if str(tools_dir) not in sys.path:
        sys.path.insert(0, str(tools_dir))
    for name in wanted:
        if name == "bioactivation":
            print(
                "bioactivation: metabolite pipeline — not a single ONNX graph.",
                file=sys.stderr,
            )
            continue
        if name == "phase1":
            from convert_phase1 import convert as convert_phase1

            if convert_phase1(args.src, args.out):
                any_ok = True
            continue
        spec = MODELS.get(name)
        if not spec:
            print(f"unknown model {name}", file=sys.stderr)
            continue
        legacy = spec["legacy"]
        src_root = libridass_root()
        pkg_root = (src_root / legacy) if src_root else None
        if pkg_root is None or not pkg_root.is_dir():
            print(f"no vendored NN package for {legacy}", file=sys.stderr)
            continue
        # search extract
        roots = [p for p in args.src.rglob(legacy) if p.is_dir()]
        # Prefer a tree that actually contains the pickled nets (tarball), not
        # sibling sources that omit models.
        def score(root: Path) -> int:
            return sum(1 for rel in spec["heads"].values() if (root / rel).is_file())

        roots.sort(key=score, reverse=True)
        if not roots or score(roots[0]) == 0:
            print(f"no pickles for {legacy} under {args.src}", file=sys.stderr)
            continue
        root = roots[0]
        print(f"using extract {root} for {name} (NN from {pkg_root})")
        dump_dir = args.out / "_dump" / name
        for head, rel in spec["heads"].items():
            pkl = root / rel
            if not pkl.is_file():
                print(f"missing pickle {pkl}", file=sys.stderr)
                continue
            dumped = dump_via_docker(pkl, pkg_root, dump_dir)
            if dumped is None:
                continue
            meta = json.loads(dumped[0].read_text())
            weights = dumped[1].read_bytes()
            dest = args.out / name / f"{head}.onnx"
            try:
                build_onnx(meta, weights, dest)
                dest.with_suffix(".meta.json").write_text(
                    json.dumps(meta, indent=2) + "\n", encoding="utf-8"
                )
                any_ok = True
            except Exception as exc:
                print(f"ONNX build failed for {name}/{head}: {exc}", file=sys.stderr)
                continue
            y_src = dump_dir / (pkl.stem + ".y.json")
            x_src = dump_dir / (pkl.stem + ".x.json")
            if x_src.is_file() and y_src.is_file():
                fixtures = Path("tests/fixtures/random_vectors.json")
                data = {}
                if fixtures.is_file() and fixtures.stat().st_size:
                    try:
                        data = json.loads(fixtures.read_text())
                    except json.JSONDecodeError:
                        data = {}
                data[f"{name}_{head}"] = {
                    "seed": 20260829,
                    "x": json.loads(x_src.read_text()),
                    "y": json.loads(y_src.read_text())["y"],
                    "I": meta["I"],
                    "O": meta["O"],
                }
                fixtures.parent.mkdir(parents=True, exist_ok=True)
                fixtures.write_text(json.dumps(data, indent=2) + "\n")
                print("updated", fixtures, f"{name}_{head}")
        for head, rel in spec.get("tsv", {}).items():
            cols = tsv_columns(root / rel)
            meta_path = args.out / name / f"{head}.meta.json"
            if cols and meta_path.is_file():
                n_in = int(json.loads(meta_path.read_text())["I"])
                cols = cols[:n_in]
            if cols:
                write_feature_json(name, head, cols, args.features_dir)

    if not any_ok:
        print(
            "No ONNX files written. Weights stay gitignored; tests skip. "
            "Do not commit placeholder ONNX.",
            file=sys.stderr,
        )
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
