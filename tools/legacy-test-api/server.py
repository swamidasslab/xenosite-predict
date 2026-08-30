"""Tiny test-only HTTP API. Not production Flask.

FROM xenosite-legacy:api. Endpoints:

- GET  /health
- POST /predict/<model>   JSON {smiles}
- POST /nn/<model>/<head> JSON {x: [[...], ...]}  (patterns × features)
- POST /features/<model>  JSON {smiles}  OpenBabel feature dump for RDKit compare
"""

from __future__ import annotations

# This file is copied into the py2 image; keep it 2/3 compatible where possible.
import json
import os
import sys

try:
    from http.server import BaseHTTPRequestHandler, HTTPServer  # py3
except ImportError:
    from BaseHTTPServer import BaseHTTPRequestHandler, HTTPServer  # py2


def _json(handler, code, obj):
    body = json.dumps(obj).encode("utf-8")
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
            _json(self, 200, {"ok": True})
            return
        _json(self, 404, {"error": "not found"})

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length else b"{}"
        try:
            payload = json.loads(raw.decode("utf-8") or "{}")
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
        except Exception as exc:
            _json(self, 500, {"error": str(exc)})
            return
        _json(self, 404, {"error": "not found"})


def _load_predictor(model):
    from libridass import (
        bioactivation1,
        epoxidation1,
        ndealk1,
        quinone1,
        reactivity1,
        ugt1,
    )

    mapping = {
        "epoxidation": epoxidation1.PyMolPredictor,
        "quinone": quinone1.PyMolPredictor,
        "reactivity": reactivity1.PyMolPredictor,
        "ugt": ugt1.PyMolPredictor,
        "ndealk": ndealk1.PyMolPredictor,
        "isozyme": ndealk1.PyMolPredictor,
        "bioactivation": bioactivation1.PyMolPredictor,
    }
    if model == "phase1":
        from libridass import bioactivation1 as b

        return b.PyMolPredictor().BPD.APMP
    if model not in mapping:
        raise KeyError("unknown model %s" % model)
    return mapping[model]()


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


def predict(model, smiles):
    P = _load_predictor(model)
    raw = P.predict(smiles) if hasattr(P, "predict") else P(smiles)
    return _serialize(raw)


def nn(model, head, matrix):
    """Run the pickled net on a numeric matrix (list of rows)."""
    import numpy as np
    from numpy import mat

    P = _load_predictor(model)
    net = _head_model(P, model, head)
    X = mat(np.asarray(matrix, dtype=float)).T  # features × patterns
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


def features(model, smiles):
    """Dump OpenBabel feature rows for RDKit-vs-OB tests."""
    P = _load_predictor(model)
    from libridass.base import XvalSub

    mol = XvalSub.cast_pymol(smiles)
    if model == "epoxidation":
        df = P.bond_descriptors(mol, original_atom_ordering=True)
        return {"columns": list(df.columns), "rows": df.fillna(0).values.tolist(),
                "index": list(df.index)}
    if model == "ugt":
        df = P.atom_descriptors(mol)
        return {"columns": list(df.columns), "rows": df.fillna(0).values.tolist(),
                "index": list(df.index)}
    if model == "reactivity":
        df = P.atom_descriptors(mol)
        return {"columns": list(df.columns), "rows": df.fillna(0).values.tolist(),
                "index": list(df.index)}
    if model in ("ndealk", "isozyme"):
        df = P.bond_descriptors(mol)
        return {"columns": list(df.columns), "rows": df.fillna(0).values.tolist(),
                "index": list(df.index)}
    raise KeyError("no feature dump for %s" % model)


def main():
    port = int(os.environ.get("PORT", "8099"))
    httpd = HTTPServer(("0.0.0.0", port), Handler)
    sys.stderr.write("legacy-test-api listening on %s\n" % port)
    httpd.serve_forever()


if __name__ == "__main__":
    main()
