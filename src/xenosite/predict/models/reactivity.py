"""Reactivity: atom heads (gsh/protein/cyanide/dna) then mol heads (two-stage)."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..backends.adapters import append_mol_atom
from ..backends.onnx import OnnxBackend
from ..errors import WeightsNotFound
from ..features import load_names, matrix_from_rows, reactivity_atom_rows, topn_site_features
from ..registry import register_model
from ..types import Molecule
from ._base import BaseRunner

HEADS = ("gsh", "protein", "cyanide", "dna")


class ReactivityRunner(BaseRunner):
    name = "reactivity"
    version = "0"
    onnx_heads = tuple(f"atom_{h}" for h in HEADS) + tuple(f"mol_{h}" for h in HEADS)

    def from_onnx(self, molecule: Molecule, backend: OnnxBackend) -> None:
        missing = [h for h in ("atom", "mol") if not backend.has_head(self.name, h)]
        # Prefer one multi-output atom.onnx; fall back to per-head files
        mol = self.rdkit_mol(molecule)
        rows = reactivity_atom_rows(mol)
        names = load_names("reactivity", "atom")
        x, _ = matrix_from_rows(rows, names)
        atom_idx = [int(r["_atom"]) for r in rows]

        if backend.has_head(self.name, "atom"):
            y = backend.run_head(self.name, "atom", x)
            if y.ndim == 1:
                y = y.reshape(-1, 1)
            if y.shape[1] == 1 and y.shape[0] == len(HEADS) * max(len(rows), 1):
                y = y.reshape(len(rows), len(HEADS))
        else:
            cols = []
            for h in HEADS:
                if not backend.has_head(self.name, f"atom_{h}"):
                    raise WeightsNotFound(
                        f"reactivity ONNX missing atom_{h}. Run make convert-onnx MODEL=reactivity"
                    )
                cols.append(backend.run_head(self.name, f"atom_{h}", x).reshape(-1))
            y = np.stack(cols, axis=1)

        mol_names = load_names("reactivity", "mol")
        for hi, h in enumerate(HEADS):
            col = y[:, hi] if y.shape[1] > hi else y[:, 0]
            atom_pred = [0.0] * molecule.atoms.num
            for idx, s in zip(atom_idx, col):
                if 0 <= idx < len(atom_pred):
                    atom_pred[idx] = float(s)
            mol_score = float(np.max(col)) if len(col) else 0.0
            head_name = f"mol_{h}" if backend.has_head(self.name, f"mol_{h}") else "mol"
            if mol_names and backend.has_head(self.name, head_name):
                mx = topn_site_features(col, rows, mol_names)
                # multi-output mol head: take column hi
                my = backend.run_head(self.name, head_name, mx)
                mol_score = float(my.reshape(-1)[min(hi, my.size - 1)])
            append_mol_atom(
                molecule,
                model=f"reactivity.{h}",
                version=self.version,
                mol=mol_score,
                atom=atom_pred,
            )

    def from_legacy(self, molecule: Molecule, native: Any) -> None:
        atom = native.get("atom") or native.get("site") or {}
        mols = native.get("mol") or {}
        for h in HEADS:
            label = h.capitalize() if h != "gsh" else "GSH"
            # native keys may be GSH/Protein/Cyanide/DNA
            key = {"gsh": "GSH", "protein": "Protein", "cyanide": "Cyanide", "dna": "DNA"}[h]
            site_map = atom.get(key) or atom.get(h) or {}
            atom_pred = [0.0] * molecule.atoms.num
            if isinstance(site_map, dict):
                for k, v in site_map.items():
                    i = int(k)
                    if 0 <= i < len(atom_pred):
                        atom_pred[i] = float(v)
            mol_score = float(mols.get(key, mols.get(h, 0.0)))
            append_mol_atom(
                molecule,
                model=f"reactivity.{h}",
                version=self.version,
                mol=mol_score,
                atom=atom_pred,
            )


register_model(
    "reactivity",
    "0",
    factory=lambda: ReactivityRunner(),
    two_stage=True,
    heads=HEADS,
)
