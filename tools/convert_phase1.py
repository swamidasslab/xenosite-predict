#!/usr/bin/env python3
"""Convert phase1 TensorFlow molecularNN pickles to ONNX (no TF at runtime).

The ``.model`` files are Python-2 pickles of ``molecularNN.model.Model``:
architecture kwargs plus numpy ``parameter_values`` / ``trainable_values``.
Host unpickle uses latin-1; TensorFlow is not imported.

Site (``HLMGraphMultipleLayer``): Window → Dense(ReLU)×2 → FC → Sigmoid.
Mol (``HLMMoleculeGraph``): Window → FC → Tanh → FC → Sigmoid.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import types
from pathlib import Path

import numpy as np

SITE_PICKLE = "BOND_FullyConnected_HB25_2Layers_0.3_L2.model"
MOL_PICKLE = "MOL_FullyConnected_HM5_2Layers_0.3_L2.model"
SITE_HEAD = "site"
MOL_HEAD = "mol"
CLASS_HEADS = (
    "stable_oxygenation",
    "unstable_oxygenation",
    "dehydrogenation",
    "reduction",
    "hydrolysis",
)


def _jsonable(v):
    if isinstance(v, (np.integer, int)) and not isinstance(v, bool):
        return int(v)
    if isinstance(v, (np.floating, float)):
        return float(v)
    return v


class _Dummy:
    def __new__(cls, *args, **kwargs):
        return object.__new__(cls)

    def __setstate__(self, state):
        if isinstance(state, dict):
            self.__dict__.update(state)
        else:
            self._state = state


def load_tf_pickle(path: Path) -> tuple[str, dict, list[np.ndarray], list[np.ndarray]]:
    """Return ``(g_class, kwargs, parameter_values, trainable_values)``."""
    g_class = [""]

    class Unpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module.startswith("numpy") or module in (
                "copy_reg",
                "_codecs",
                "__builtin__",
            ):
                if module == "__builtin__":
                    import builtins

                    return getattr(builtins, name)
                __import__(module)
                return getattr(sys.modules[module], name)
            if module.startswith("molecularNN") or module.startswith("scripts."):
                g_class[0] = f"{module}.{name}"
                return _Dummy
            mod = sys.modules.get(module)
            if mod is None:
                mod = types.ModuleType(module)
                sys.modules[module] = mod
            cls = getattr(mod, name, None)
            if cls is None:
                cls = type(str(name), (_Dummy,), {"__module__": module})
                setattr(mod, name, cls)
            return cls

    with path.open("rb") as fh:
        obj = Unpickler(fh, encoding="latin1").load()
    kwargs = dict(getattr(obj, "kwargs", None) or {})
    params = [np.asarray(x, dtype=np.float32) for x in obj.parameter_values]
    trains = [np.asarray(x, dtype=np.float32) for x in obj.trainable_values]
    return g_class[0], kwargs, params, trains


def window(x: np.ndarray, mn: np.ndarray, mx: np.ndarray) -> np.ndarray:
    mn = np.asarray(mn, dtype=np.float32).reshape(1, -1)
    mx = np.asarray(mx, dtype=np.float32).reshape(1, -1)
    denom = np.where(mx == mn, np.float32(1.0), mx - mn)
    return np.float32(2.0) * (x - mn) / denom - np.float32(1.0)


def numpy_forward_site(x: np.ndarray, spec: dict) -> np.ndarray:
    h = window(x, spec["mn"], spec["mx"])
    w0, w1, w2 = spec["weights"]
    b0, b1, b2 = spec["biases"]
    h = np.maximum(h @ w0 + b0, 0.0)
    h = np.maximum(h @ w1 + b1, 0.0)
    return 1.0 / (1.0 + np.exp(-(h @ w2 + b2)))


def numpy_forward_mol(x: np.ndarray, spec: dict) -> np.ndarray:
    h = window(x, spec["mn"], spec["mx"])
    w0, w1 = spec["weights"]
    b0, b1 = spec["biases"]
    h = np.tanh(h @ w0 + b0)
    return 1.0 / (1.0 + np.exp(-(h @ w1 + b1)))


def site_spec(params: list[np.ndarray], trains: list[np.ndarray]) -> dict:
    if len(params) < 2 or len(trains) != 6:
        raise ValueError(f"unexpected site tensors params={len(params)} trains={len(trains)}")
    return {
        "mn": params[0],
        "mx": params[1],
        "weights": trains[:3],
        "biases": trains[3:],
        "I": int(trains[0].shape[0]),
        "O": int(trains[2].shape[1]),
    }


def mol_spec(params: list[np.ndarray], trains: list[np.ndarray]) -> dict:
    if len(params) < 2 or len(trains) != 4:
        raise ValueError(f"unexpected mol tensors params={len(params)} trains={len(trains)}")
    # Duplicate WindowLayer in HLMMoleculeGraph; both copies match.
    return {
        "mn": params[0],
        "mx": params[1],
        "weights": trains[:2],
        "biases": trains[2:],
        "I": int(trains[0].shape[0]),
        "O": int(trains[1].shape[1]),
    }


def _const(arr: np.ndarray, name: str):
    from onnx import numpy_helper

    return numpy_helper.from_array(np.asarray(arr, dtype=np.float32), name=name)


def build_windowed_mlp(
    dest: Path,
    *,
    n_in: int,
    n_out: int,
    mn: np.ndarray,
    mx: np.ndarray,
    layers: list[tuple[np.ndarray, np.ndarray, str]],
) -> None:
    """``layers`` is ``(W[I,O], b[O], act)`` with act in Relu/Tanh/Sigmoid."""
    from onnx import TensorProto, helper

    mn = np.asarray(mn, dtype=np.float32).reshape(1, -1)
    mx = np.asarray(mx, dtype=np.float32).reshape(1, -1)
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [None, n_in])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [None, n_out])
    inits = [
        _const(mn, "mn"),
        _const(mx, "mx"),
        _const(np.ones_like(mn), "ones"),
        _const(np.array(2.0, dtype=np.float32), "two"),
        _const(np.array(1.0, dtype=np.float32), "one"),
    ]
    nodes = [
        helper.make_node("Sub", ["X", "mn"], ["shift"]),
        helper.make_node("Sub", ["mx", "mn"], ["span"]),
        helper.make_node("Equal", ["mx", "mn"], ["tied"]),
        helper.make_node("Where", ["tied", "ones", "span"], ["denom"]),
        helper.make_node("Div", ["shift", "denom"], ["unit"]),
        helper.make_node("Mul", ["unit", "two"], ["twice"]),
        helper.make_node("Sub", ["twice", "one"], ["win"]),
    ]
    cur = "win"
    for i, (w, b, act) in enumerate(layers):
        w = np.asarray(w, dtype=np.float32)
        b = np.asarray(b, dtype=np.float32).reshape(1, -1)
        inits.append(_const(w, f"W{i}"))
        inits.append(_const(b, f"b{i}"))
        gemm = f"g{i}"
        nodes.append(
            helper.make_node("Gemm", [cur, f"W{i}", f"b{i}"], [gemm], alpha=1.0, beta=1.0)
        )
        out = f"a{i}"
        nodes.append(helper.make_node(act, [gemm], [out]))
        cur = out
    if cur != "Y":
        nodes.append(helper.make_node("Identity", [cur], ["Y"]))
    graph = helper.make_graph(nodes, "phase1_mlp", [X], [Y], inits)
    model = helper.make_model(
        graph,
        producer_name="xenosite-predict",
        opset_imports=[helper.make_opsetid("", 13)],
    )
    dest.parent.mkdir(parents=True, exist_ok=True)
    import onnx

    onnx.save(model, str(dest))
    print("wrote", dest)


def find_pickle(src: Path, name: str) -> Path | None:
    hits = [p for p in src.rglob(name) if p.is_file()]
    return hits[0] if hits else None


def convert(src: Path, out: Path) -> bool:
    site_pkl = find_pickle(src, SITE_PICKLE)
    mol_pkl = find_pickle(src, MOL_PICKLE)
    if site_pkl is None or mol_pkl is None:
        print(f"no phase1 pickles under {src}", file=sys.stderr)
        return False

    _g, kw, params, trains = load_tf_pickle(site_pkl)
    site = site_spec(params, trains)
    dest = out / "phase1" / f"{SITE_HEAD}.onnx"
    build_windowed_mlp(
        dest,
        n_in=site["I"],
        n_out=site["O"],
        mn=site["mn"],
        mx=site["mx"],
        layers=[
            (site["weights"][0], site["biases"][0], "Relu"),
            (site["weights"][1], site["biases"][1], "Relu"),
            (site["weights"][2], site["biases"][2], "Sigmoid"),
        ],
    )
    meta = {
        "I": site["I"],
        "O": site["O"],
        "kind": "tf_mlp",
        "g_class": "scripts.hlm_model_top5.HLMGraphMultipleLayer",
        "kwargs": {k: _jsonable(v) for k, v in kw.items()},
        "heads": list(CLASS_HEADS),
    }
    dest.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    _g, kw, params, trains = load_tf_pickle(mol_pkl)
    mol = mol_spec(params, trains)
    dest = out / "phase1" / f"{MOL_HEAD}.onnx"
    build_windowed_mlp(
        dest,
        n_in=mol["I"],
        n_out=mol["O"],
        mn=mol["mn"],
        mx=mol["mx"],
        layers=[
            (mol["weights"][0], mol["biases"][0], "Tanh"),
            (mol["weights"][1], mol["biases"][1], "Sigmoid"),
        ],
    )
    meta = {
        "I": mol["I"],
        "O": mol["O"],
        "kind": "tf_mlp",
        "g_class": "scripts.hlm_model_top5_molecule.HLMMoleculeGraph",
        "kwargs": {k: _jsonable(v) for k, v in kw.items()},
        "heads": list(CLASS_HEADS),
    }
    dest.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    rng = np.random.default_rng(20260830)
    xs = rng.standard_normal((8, site["I"])).astype(np.float32)
    xm = rng.standard_normal((8, mol["I"])).astype(np.float32)
    np.testing.assert_allclose(
        numpy_forward_site(xs, site).astype(np.float32),
        _ort_run(out / "phase1" / f"{SITE_HEAD}.onnx", xs),
        atol=1e-5,
        rtol=0,
    )
    np.testing.assert_allclose(
        numpy_forward_mol(xm, mol).astype(np.float32),
        _ort_run(out / "phase1" / f"{MOL_HEAD}.onnx", xm),
        atol=1e-5,
        rtol=0,
    )
    print("phase1 ONNX matches numpy reconstruction (atol 1e-5)")
    return True


def _ort_run(path: Path, x: np.ndarray) -> np.ndarray:
    import onnxruntime as ort

    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    return np.asarray(sess.run(None, {name: np.asarray(x, dtype=np.float32)})[0])


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src", type=Path, default=Path("weights/legacy"))
    p.add_argument("--out", type=Path, default=Path("weights/onnx/v0"))
    args = p.parse_args(argv)
    return 0 if convert(args.src, args.out) else 1


if __name__ == "__main__":
    raise SystemExit(main())
