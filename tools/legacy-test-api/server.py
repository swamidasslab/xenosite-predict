# -*- coding: utf-8 -*-
"""Tiny test-only HTTP API. Not production Flask.

FROM xenosite-legacy:api. Endpoints:

- GET  /health
- POST /predict/<model>   JSON {smiles}
- POST /nn/<model>/<head> JSON {x: [[...], ...]}  (patterns x features)
- POST /features/<model>  JSON {smiles}  OpenBabel feature dump for RDKit compare
"""

# This file is copied into the py2 image; keep it 2/3 compatible where possible.
import json
import os
import sys
import types


def _stub_confargparse():
    sys.modules.setdefault("confargparse", types.ModuleType("confargparse"))


def _stub_module(name, **attrs):
    mod = types.ModuleType(name)
    for key, val in attrs.items():
        setattr(mod, key, val)
    sys.modules[name] = mod
    return mod


def _stub_openopt():
    """Stub openopt only when not installed (production image has the real package)."""
    try:
        __import__("openopt")
        return
    except ImportError:
        pass

    class OpenOptResult(object):
        pass

    class EmptyClass(object):
        pass

    _stub_module("result", OpenOptResult=OpenOptResult)
    _stub_module("nonOptMisc", EmptyClass=EmptyClass)
    mod = types.ModuleType("openopt")

    class _Result(object):
        xf = []

    class NLP(object):
        def __init__(self, *args, **kwargs):
            pass

        def solve(self, *args, **kwargs):
            return _Result()

    mod.NLP = NLP
    sys.modules["openopt"] = mod


_stub_openopt()
_stub_confargparse()

try:
    from http.server import BaseHTTPRequestHandler, HTTPServer  # py3
except ImportError:
    from BaseHTTPServer import BaseHTTPRequestHandler, HTTPServer  # py2


def _json(handler, code, obj):
    body = json.dumps(obj)
    if not isinstance(body, bytes):
        body = body.encode("utf-8")
    handler.send_response(code)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    def do_GET(self):
        if self.path.startswith("/health"):
            _json(self, 200, {"ok": True, "pid": os.getpid()})
            return
        _json(self, 404, {"error": "not found"})

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length else ""
        if isinstance(raw, bytes):
            text = raw.decode("utf-8") or "{}"
        else:
            text = raw or "{}"
        try:
            payload = json.loads(text)
        except ValueError:
            _json(self, 400, {"error": "invalid json"})
            return
        parts = [p for p in self.path.split("/") if p]
        try:
            if len(parts) >= 2 and parts[0] == "predict":
                _json(self, 200, predict(parts[1], payload.get("smiles")))
                return
            if len(parts) >= 3 and parts[0] == "nn":
                _json(self, 200, nn(parts[1], parts[2], payload.get("x")))
                return
            if len(parts) >= 2 and parts[0] == "features":
                _json(self, 200, features(parts[1], payload.get("smiles")))
                return
            if len(parts) >= 2 and parts[0] == "moldesc":
                _json(self, 200, moldesc(parts[1], payload.get("smiles")))
                return
        except Exception as exc:
            _json(self, 500, {"error": str(exc)})
            return
        _json(self, 404, {"error": "not found"})


def _load_predictor_impl(model):
    """Load pickled nets once per process (warmup); duplicate requests cached by nginx."""
    if model == "epoxidation":
        from libridass.epoxidation1 import PyMolPredictor

        return PyMolPredictor()
    if model == "quinone":
        from libridass.quinone1 import PyMolPredictor

        return PyMolPredictor()
    if model == "reactivity":
        from libridass.reactivity1 import PyMolPredictor

        return PyMolPredictor()
    if model == "ugt":
        from libridass.ugt1 import PyMolPredictor

        return PyMolPredictor()
    if model in ("ndealk", "isozyme"):
        from libridass.ndealk1 import PyMolPredictor

        return PyMolPredictor()
    if model == "bioactivation":
        from libridass.bioactivation1 import PyMolPredictor

        return PyMolPredictor()
    if model == "phase1":
        from libridass.bioactivation1 import PyMolPredictor

        return PyMolPredictor().BPD.APMP
    raise KeyError("unknown model %s" % model)


_LOADED_PREDICTORS = {}


def _load_predictor(model):
    if model not in _LOADED_PREDICTORS:
        _LOADED_PREDICTORS[model] = _load_predictor_impl(model)
    return _LOADED_PREDICTORS[model]


def _serialize(obj):
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if isinstance(k, frozenset):
                k = "-".join(str(x) for x in sorted(k))
            else:
                k = str(k)
            out[k] = _serialize(v)
        return out
    if isinstance(obj, (list, tuple)):
        return [_serialize(x) for x in obj]
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:
            pass
    if isinstance(obj, (int, float, str, bool)) or obj is None:
        return obj
    return str(obj)


def _rdkit_site_dict(site):
    """Convert 1-based topological group ids in quinone site maps to 0-based keys.

    Quinone ``site`` frozensets use topologically equivalent **group** numbers
    (middle field of ``mol.group.atom``), not raw OpenBabel ``GetIdx()``.
    """
    if not isinstance(site, dict):
        return site
    out = {}
    for k, v in site.items():
        if isinstance(k, frozenset):
            out[frozenset(int(x) - 1 for x in k)] = v
        elif isinstance(k, str) and "-" in k:
            # Some legacy predictors emit string keys (still 1-based OB).
            parts = k.split("-", 1)
            if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                a, b = int(parts[0]) - 1, int(parts[1]) - 1
                out["%d-%d" % (a, b)] = v
            else:
                out[k] = v
        else:
            out[k] = v
    return out


def _normalize_smiles(smiles):
    if smiles is None:
        raise ValueError("missing smiles")
    try:
        unicode_type = unicode  # py2
    except NameError:
        unicode_type = str
    if isinstance(smiles, unicode_type) and unicode_type is not str:
        smiles = smiles.encode("utf-8")
    return smiles


def predict(model, smiles):
    from libridass.base import XvalSub

    smiles = _normalize_smiles(smiles)
    P = _load_predictor(model)
    if model in ("ndealk", "isozyme"):
        pymol = XvalSub.cast_pymol(smiles)
        raw = P.predict(pymol, indexed_zero=True)
    elif model == "epoxidation":
        raw = P.predict(smiles)
    elif model in ("quinone", "reactivity"):
        pymol = XvalSub.cast_pymol(smiles)
        raw = P.predict(pymol, xval_sub=False)
    elif hasattr(P, "predict"):
        pymol = XvalSub.cast_pymol(smiles)
        raw = P.predict(pymol)
    else:
        pymol = XvalSub.cast_pymol(smiles)
        raw = P(pymol)
    if isinstance(raw, dict):
        for key in ("site", "bond"):
            if key in raw:
                raw[key] = _rdkit_site_dict(raw[key])
    return _serialize(raw)


def nn(model, head, matrix):
    """Run the pickled net on a numeric matrix (list of rows)."""
    import numpy as np
    from numpy import mat

    P = _load_predictor(model)
    net = _head_model(P, model, head)
    X = mat(np.asarray(matrix, dtype=float)).T  # features x patterns
    ids = [str(i) for i in range(X.shape[1])]
    pred = net.predict((ids, X))
    _id, Y = pred
    return {"y": Y.T.A.tolist()}


def _head_model(P, model, head):
    if model == "epoxidation":
        return P.bond_model if head in ("bond", "site") else P.mol_model
    if model == "quinone":
        return {
            "atom": P.atom_model,
            "pair": P.atom_pair_model,
            "mol": P.mol_model,
        }[head]
    if model == "reactivity":
        return P.atom_model if head.startswith("atom") else P.mol_model
    if model == "ugt":
        return P.model
    if model in ("ndealk", "isozyme"):
        return P.bond_model
    raise KeyError("no nn head %s/%s" % (model, head))


def moldesc(model, smiles):
    """Dump two-stage mol-head input row (legacy ``MolDesc`` / ``MolData``)."""
    from libridass.base import XvalSub

    smiles = _normalize_smiles(smiles)
    pymol = XvalSub.cast_pymol(smiles)
    if model == "reactivity":
        from libridass.reactivity1 import PyMolPredictor
        from libridass.reactivity1.code.moldesc import MolDesc

        P = PyMolPredictor()
        atom_data = P.atom_descriptors(pymol)
        atom_pred = P.apply_model(atom_data, P.atom_model)
        mol_data = MolDesc().run(
            atom_data, atom_pred, P.mol_header, "Training__MolDesc__AtomTop5"
        )
        row = mol_data.iloc[0]
        return {
            "columns": list(row.index),
            "row": [float(row[c]) if row[c] == row[c] else 0.0 for c in row.index],
        }
    raise KeyError("no moldesc dump for %s" % model)


def features(model, smiles):
    """Dump OpenBabel feature rows for RDKit-vs-OB tests."""
    from libridass.base import XvalSub

    smiles = _normalize_smiles(smiles)
    P = _load_predictor(model)
    pymol = XvalSub.cast_pymol(smiles)
    if model == "epoxidation":
        df = P.bond_descriptors(pymol, original_atom_ordering=True)
        return {"columns": list(df.columns), "rows": df.fillna(0).values.tolist(),
                "index": list(df.index)}
    if model == "ugt":
        df = P.atom_descriptors(pymol)
        return {"columns": list(df.columns), "rows": df.fillna(0).values.tolist(),
                "index": list(df.index)}
    if model == "quinone":
        df = P.atom_descriptors(pymol)
        return {"columns": list(df.columns), "rows": df.fillna(0).values.tolist(),
                "index": list(df.index)}
    if model == "reactivity":
        df = P.atom_descriptors(pymol)
        return {"columns": list(df.columns), "rows": df.fillna(0).values.tolist(),
                "index": list(df.index)}
    if model in ("ndealk", "isozyme"):
        df = P.bond_descriptors(pymol)
        return {"columns": list(df.columns), "rows": df.fillna(0).values.tolist(),
                "index": list(df.index)}
    raise KeyError("no feature dump for %s" % model)


def main():
    port = int(os.environ.get("PORT", "8099"))
    httpd = HTTPServer(("0.0.0.0", port), Handler)
    sys.stderr.write("legacy-test-api listening on %s (pid=%s)\n" % (port, os.getpid()))
    httpd.serve_forever()


if __name__ == "__main__":
    main()
