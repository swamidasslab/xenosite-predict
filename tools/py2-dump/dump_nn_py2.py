# coding: utf-8
# Python 2.7 dump helper: unpickle a numpy NN without libridass.utils
# (that module imports OpenBabel). Put the model package root on sys.path so
# pickle GLOBALS like code.NNmodel and NN.prior resolve.
from __future__ import print_function

import json
import os
import pickle
import sys
import types

import numpy as np


def _stub(modname, **attrs):
    mod = types.ModuleType(modname)
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules[modname] = mod
    return mod


class OpenOptResult(object):
    pass


class EmptyClass(object):
    pass


_stub("result", OpenOptResult=OpenOptResult)
_stub("nonOptMisc", EmptyClass=EmptyClass)
# NN.train does ``from openopt import *`` at import; we only need the pickled
# OpenOptResult instance (module ``result``), not the solver.
_stub("openopt")


class PermissiveUnpickler(pickle.Unpickler):
    """Resolve real classes when present; stub missing OpenOpt/etc. modules."""

    def find_class(self, module, name):
        try:
            __import__(module)
            return getattr(sys.modules[module], name)
        except (ImportError, AttributeError):
            mod = sys.modules.get(module)
            if mod is None:
                mod = types.ModuleType(module)
                sys.modules[module] = mod
            cls = getattr(mod, name, None)
            if cls is None:
                cls = type(str(name), (object,), {"__module__": module})
                setattr(mod, name, cls)
            return cls


def walk_layers(layer):
    out = []
    cur = layer
    while cur is not None:
        name = cur.__class__.__name__
        rec = {
            "class": name,
            "n_in": int(getattr(cur, "n_in", 0) or 0),
            "n_out": int(getattr(cur, "n_out", 0) or 0),
            "n_w": int(cur.this_len()),
        }
        if name in ("WindowedInputLayer", "NormalizedInputLayer"):
            rec["center"] = [float(x) for x in np.asarray(cur.center).flatten()]
            rec["spread"] = [float(x) for x in np.asarray(cur.spread).flatten()]
        if name == "GaussianError":
            rec["ave"] = [float(x) for x in np.asarray(cur.ave).flatten()]
            rec["std"] = [float(x) for x in np.asarray(cur.std).flatten()]
        if name == "AbutLayer":
            rec["children"] = [walk_layers(L) for L in cur.LAYERS]
        out.append(rec)
        cur = getattr(cur, "ABOVE", None)
    return out


def dump_model(path, pkg_root, out_json, out_w, x_json=None, y_json=None):
    pkg_root = os.path.abspath(pkg_root)
    sys.path.insert(0, pkg_root)
    m = PermissiveUnpickler(open(path, "rb")).load()
    W = np.asarray(m.R.xf).reshape(-1)
    graph = walk_layers(m.model)
    meta = {
        "layers": graph,
        "I": int(m.I),
        "H": int(getattr(m, "H", 0) or 0),
        "O": int(m.O),
        "n_weights": int(len(W)),
        "class": m.__class__.__name__,
    }
    json.dump(meta, open(out_json, "w"))
    open(out_w, "wb").write(W.astype("float64").tobytes())
    print("dumped", path, "I", m.I, "H", m.H, "O", m.O, "layers",
          [L["class"] for L in graph])
    if x_json and y_json and os.path.isfile(x_json):
        X = np.asarray(json.load(open(x_json)), dtype=float)
        # Legacy net wants features x patterns.
        if X.shape[1] == int(m.I) and X.shape[0] != int(m.I):
            Xin = np.mat(X.T)
        else:
            Xin = np.mat(X)
        Y = np.asarray(m.model.output(Xin, np.asarray(W).reshape(-1, 1)))
        y = np.asarray(Y).T
        json.dump({"y": y.tolist(), "shape": list(y.shape)}, open(y_json, "w"))
        print("predicted", y.shape)


if __name__ == "__main__":
    # argv: pickle pkg_root out_json out_w [x_json y_json]
    dump_model(*sys.argv[1:])
