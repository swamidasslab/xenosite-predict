"""N-dealkylation and isozyme (same ndealk1 ONNX; isozyme exposes all 10 heads).

Production Flask wires ``metabolism1`` to ``ndealk1.PyMolPredictor``. The unused
``metabolism1.predictor.PyMolPredictor`` path uses MOPAC+SmartCYP and is **not**
this model. See docs/vendored-diffs.md.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..backends.adapters import append_bond, reorder_by_bond
from ..backends.onnx import OnnxBackend
from ..errors import WeightsNotFound
from ..features import bond_rows, load_names, matrix_from_rows
from ..registry import register_model
from ..types import Molecule
from ._base import BaseRunner

ISOZYMES = ("3a4", "2d6", "2c8", "2c9", "1a2", "2b6", "2e1", "2a6", "2c19", "hlm")
# ndealk1 output column order from predictor.py
_OUT_ORDER = "1A2 2A6 2B6 2C8 2C9 2C19 2D6 2E1 3A4 HLM".split()
_OUT_TO_API = {
    "1A2": "1a2",
    "2A6": "2a6",
    "2B6": "2b6",
    "2C8": "2c8",
    "2C9": "2c9",
    "2C19": "2c19",
    "2D6": "2d6",
    "2E1": "2e1",
    "3A4": "3a4",
    "HLM": "hlm",
}


class NdealkFamily(BaseRunner):
    version = "0"
    onnx_heads = ("bond",)
    isozyme_mode: bool = False

    def from_onnx(self, molecule: Molecule, backend: OnnxBackend) -> None:
        if not backend.has_head("ndealk", "bond"):
            raise WeightsNotFound(
                "ndealk/isozyme ONNX missing bond. Run make convert-onnx MODEL=ndealk"
            )
        mol = self.rdkit_mol(molecule)
        rows = bond_rows(mol, original_atom_ordering=True)
        names = load_names("ndealk", "bond")
        if not names:
            raise WeightsNotFound(
                "ndealk feature-name JSON is missing (no training TSV in the tarball). "
                "Cannot align RDKit columns to the 386-D ONNX input."
            )
        x, _ = matrix_from_rows(rows, names)
        y = backend.run_head("ndealk", "bond", x)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        keys = [frozenset(r["_atoms"]) for r in rows]  # type: ignore[arg-type]
        self._append_from_matrix(molecule, y, keys)

    def from_legacy(self, molecule: Molecule, native: Any) -> None:
        # native: {isozyme: {frozenset or "i-j": score}}
        if self.isozyme_mode:
            for out in ISOZYMES:
                site = native.get(out) or native.get(out.upper()) or {}
                self._append_bond_map(molecule, f"isozyme.{out}", site)
        else:
            site = native.get("hlm") or native.get("HLM") or native.get("site") or {}
            self._append_bond_map(molecule, "ndealk", site)

    def _append_from_matrix(self, molecule: Molecule, y: np.ndarray, keys) -> None:
        col_index = {name: i for i, name in enumerate(_OUT_ORDER)}
        if self.isozyme_mode:
            for api_name in ISOZYMES:
                src = {v: k for k, v in _OUT_TO_API.items()}[api_name]
                ci = col_index.get(src, 0)
                ci = min(ci, y.shape[1] - 1)
                pred = [float(v) for v in y[:, ci]]
                bond_pred = reorder_by_bond(pred, keys, molecule.bonds.idx, fill=0.0)
                append_bond(molecule, model=f"isozyme.{api_name}", version=self.version, bond=bond_pred)
        else:
            ci = col_index.get("HLM", y.shape[1] - 1)
            ci = min(ci, y.shape[1] - 1)
            pred = [float(v) for v in y[:, ci]]
            bond_pred = reorder_by_bond(pred, keys, molecule.bonds.idx, fill=0.0)
            append_bond(molecule, model="ndealk", version=self.version, bond=bond_pred)

    def _append_bond_map(self, molecule: Molecule, model: str, site: dict) -> None:
        current = []
        pred = []
        for key, val in (site.items() if isinstance(site, dict) else []):
            if isinstance(key, str) and "-" in key:
                a, b = key.split("-", 1)
                current.append((int(a), int(b)))
            elif isinstance(key, (list, tuple)):
                current.append((int(key[0]), int(key[1])))
            else:
                continue
            pred.append(float(val or 0.0))
        bond_pred = reorder_by_bond(pred, current, molecule.bonds.idx, fill=0.0)
        append_bond(molecule, model=model, version=self.version, bond=bond_pred)


class NdealkRunner(NdealkFamily):
    name = "ndealk"
    isozyme_mode = False


class IsozymeRunner(NdealkFamily):
    name = "isozyme"
    isozyme_mode = True


register_model("ndealk", "0", factory=lambda: NdealkRunner(), heads=("hlm",))
register_model("isozyme", "0", factory=lambda: IsozymeRunner(), heads=ISOZYMES)
