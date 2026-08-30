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

Do not fake ONNX files when dump/convert fails. Phase1/bioactivation are TF1
molecularNN — stop and document if convert fails; do not add TF at runtime.
"""

from __future__ import annotations

import argparse
import json
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

IMAGE = "dockerreg01.accounts.ad.wustl.edu/swamidass/xenosite-legacy:api"

DUMP_PY2 = r'''
# Python 2.7 dump helper — executed inside xenosite-legacy:api
from __future__ import print_function
import json, os, sys
sys.path.insert(0, os.environ.get("LIBRIDASS_ROOT", "/usr/local/lib/python2.7/site-packages"))

def walk_layers(layer):
    out = []
    cur = layer
    while cur is not None:
        name = cur.__class__.__name__
        rec = {"class": name, "n_in": int(getattr(cur, "n_in", 0) or 0),
               "n_out": int(getattr(cur, "n_out", 0) or 0), "n_w": int(cur.this_len())}
        if name in ("WindowedInputLayer", "NormalizedInputLayer"):
            rec["center"] = [float(x) for x in cur.center.flatten()]
            rec["spread"] = [float(x) for x in cur.spread.flatten()]
        if name == "GaussianError":
            rec["ave"] = [float(x) for x in cur.ave.flatten()]
            rec["std"] = [float(x) for x in cur.std.flatten()]
        out.append(rec)
        cur = getattr(cur, "ABOVE", None)
    return out

def dump_model(path, out_json, out_w):
    from libridass.utils import RebasingUnpickler
    pkg = os.environ["REBASE_PKG"]
    m = RebasingUnpickler(pkg, open(path, "rb")).load()
    W = m.R.xf
    graph = walk_layers(m.model)
    meta = {"layers": graph, "I": int(m.I), "H": int(getattr(m, "H", 0) or 0),
            "O": int(m.O), "n_weights": int(len(W))}
    json.dump(meta, open(out_json, "w"))
    open(out_w, "wb").write(W.astype("float64").tobytes())
    print("dumped", path, "->", out_json)

if __name__ == "__main__":
    dump_model(sys.argv[1], sys.argv[2], sys.argv[3])
'''


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
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [None, n_in])
    nodes = []
    inits = []
    cursor = 0
    cur_name = "X"
    for i, layer in enumerate(layers):
        cls = layer["class"]
        if cls in ("WindowedInputLayer", "NormalizedInputLayer"):
            center = np.asarray(layer["center"], dtype=np.float32).reshape(1, -1)
            spread = np.asarray(layer["spread"], dtype=np.float32).reshape(1, -1)
            inits.append(numpy_helper.from_array(center, name=f"c{i}"))
            inits.append(numpy_helper.from_array(spread, name=f"s{i}"))
            sub = f"sub{i}"
            nodes.append(helper.make_node("Sub", [cur_name, f"c{i}"], [sub]))
            out = f"n{i}"
            nodes.append(helper.make_node("Div", [sub, f"s{i}"], [out]))
            cur_name = out
        elif cls == "FullyConnectedLayer":
            n_w = int(layer["n_w"])
            n_out, n_in_l = int(layer["n_out"]), int(layer["n_in"])
            w = W[cursor : cursor + n_w].astype(np.float32).reshape(n_out, n_in_l)
            cursor += n_w
            # ONNX Gemm: Y = X * W^T  (X is N×in, W is out×in)
            inits.append(numpy_helper.from_array(w, name=f"W{i}"))
            out = f"n{i}"
            nodes.append(
                helper.make_node("Gemm", [cur_name, f"W{i}"], [out], transB=1, alpha=1.0, beta=0.0)
            )
            cur_name = out
        elif cls in ("LogisticLayer", "CrossEntropyError"):
            n_w = int(layer["n_w"])
            b = W[cursor : cursor + n_w].astype(np.float32).reshape(1, -1)
            cursor += n_w
            inits.append(numpy_helper.from_array(b, name=f"b{i}"))
            add = f"add{i}"
            nodes.append(helper.make_node("Add", [cur_name, f"b{i}"], [add]))
            out = f"n{i}"
            nodes.append(helper.make_node("Sigmoid", [add], [out]))
            cur_name = out
        elif cls == "GaussianError":
            std = np.asarray(layer["std"], dtype=np.float32).reshape(1, -1)
            ave = np.asarray(layer["ave"], dtype=np.float32).reshape(1, -1)
            inits.append(numpy_helper.from_array(std, name=f"std{i}"))
            inits.append(numpy_helper.from_array(ave, name=f"ave{i}"))
            mul = f"mul{i}"
            nodes.append(helper.make_node("Mul", [cur_name, f"std{i}"], [mul]))
            out = f"n{i}"
            nodes.append(helper.make_node("Add", [mul, f"ave{i}"], [out]))
            cur_name = out
        elif cls in ("OutputLayer", "EmptyLayer"):
            continue
        else:
            # identity / unknown — keep going
            continue
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [None, None])
    graph = helper.make_graph(nodes, "xenosite_nn", [X], [Y], inits)
    model = helper.make_model(graph, producer_name="xenosite-predict")
    dest.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(dest))
    print("wrote", dest)


def dump_via_docker(pickle_path: Path, rebase_pkg: str, dump_dir: Path) -> tuple[Path, Path] | None:
    dump_dir.mkdir(parents=True, exist_ok=True)
    helper = dump_dir / "dump_nn_py2.py"
    helper.write_text(DUMP_PY2)
    out_json = dump_dir / (pickle_path.stem + ".graph.json")
    out_w = dump_dir / (pickle_path.stem + ".weights.bin")
    try:
        subprocess.run(
            [
                "docker",
                "run",
                "--rm",
                "-v",
                f"{pickle_path.parent.resolve()}:/pkl:ro",
                "-v",
                f"{dump_dir.resolve()}:/dump",
                "-e",
                f"REBASE_PKG={rebase_pkg}",
                IMAGE,
                "python",
                "/dump/dump_nn_py2.py",
                f"/pkl/{pickle_path.name}",
                f"/dump/{out_json.name}",
                f"/dump/{out_w.name}",
            ],
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        print(f"docker dump failed for {pickle_path}: {exc}", file=sys.stderr)
        return None
    if out_json.is_file() and out_w.is_file():
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

    wanted = [args.model] if args.model else list(MODELS)
    any_ok = False
    for name in wanted:
        spec = MODELS.get(name)
        if not spec:
            print(f"unknown model {name}", file=sys.stderr)
            continue
        if name in ("phase1", "bioactivation"):
            print(
                f"{name}: TF molecularNN / pipeline — stop if convert fails; "
                "not converting on the host without a dump.",
                file=sys.stderr,
            )
            continue
        legacy = spec["legacy"]
        rebase = f"libridass.{legacy}"
        # search extract
        roots = list(args.src.rglob(legacy))
        if not roots:
            print(f"no extract for {legacy} under {args.src}", file=sys.stderr)
            continue
        root = roots[0]
        dump_dir = args.out / "_dump" / name
        for head, rel in spec["heads"].items():
            pkl = root / rel
            if not pkl.is_file():
                print(f"missing pickle {pkl}", file=sys.stderr)
                continue
            dumped = dump_via_docker(pkl, rebase, dump_dir)
            if dumped is None:
                continue
            meta = json.loads(dumped[0].read_text())
            weights = dumped[1].read_bytes()
            dest = args.out / name / f"{head}.onnx"
            try:
                build_onnx(meta, weights, dest)
                any_ok = True
            except Exception as exc:
                print(f"ONNX build failed for {name}/{head}: {exc}", file=sys.stderr)
        for head, rel in spec.get("tsv", {}).items():
            cols = tsv_columns(root / rel)
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
