"""Reactivity: atom heads (gsh/protein/cyanide/dna) then mol heads (two-stage)."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..backends.adapters import append_mol_atom, legacy_atom_vector
from ..backends.onnx import OnnxBackend
from ..errors import WeightsNotFound
from ..features import load_names, matrix_from_rows, reactivity_atom_rows
from ..features.reactivity_mol import (
    reactivity_mol_features,
    reactivity_mol_output_index,
    reactivity_onnx_rows,
)
from ..registry import register_model
from ..types import Molecule
from ._base import BaseRunner

HEADS = ("gsh", "protein", "cyanide", "dna")
# Legacy atom model OUT1..4 order (see ``reactivity1`` ``self.targets``).
_ATOM_OUT_ORDER = ("cyanide", "dna", "gsh", "protein")


class ReactivityRunner(BaseRunner):
    name = "reactivity"
    version = "0"
    onnx_heads = tuple(f"atom_{h}" for h in HEADS) + tuple(f"mol_{h}" for h in HEADS)

    def from_onnx(self, molecule: Molecule, backend: OnnxBackend) -> None:
        missing = [h for h in ("atom", "mol") if not backend.has_head(self.name, h)]
        mol = self.rdkit_mol(molecule)
        rows = reactivity_atom_rows(mol)
        onnx_rows = reactivity_onnx_rows(rows)
        names = load_names("reactivity", "atom")
        x, _ = matrix_from_rows(onnx_rows, names)
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
        scores_by_head: dict[str, list[float]] = {}
        atom_preds_by_head: dict[str, list[float]] = {}

        for hi, h in enumerate(_ATOM_OUT_ORDER):
            col = y[:, hi] if y.shape[1] > hi else y[:, 0]
            scores_by_head[h] = [float(v) for v in col]
            atom_pred = [0.0] * molecule.atoms.num
            for idx, s in zip(atom_idx, col):
                if 0 <= idx < len(atom_pred):
                    atom_pred[idx] = float(s)
            atom_preds_by_head[h] = atom_pred

        mol_mx = (
            reactivity_mol_features(rows, scores_by_head, mol_names)
            if mol_names
            else None
        )
        mol_out = None
        if mol_mx is not None and backend.has_head(self.name, "mol"):
            mol_out = backend.run_head(self.name, "mol", mol_mx).reshape(-1)

        for h in HEADS:
            mol_score = 0.0
            if mol_out is not None:
                mol_score = float(mol_out[reactivity_mol_output_index(h)])
            append_mol_atom(
                molecule,
                model=f"reactivity.{h}",
                version=self.version,
                mol=mol_score,
                atom=atom_preds_by_head[h],
            )

    def from_legacy(self, molecule: Molecule, native: Any) -> None:
        atom = native.get("atom") or native.get("site") or {}
        mols = native.get("mol") or {}
        for h in HEADS:
            label = h.capitalize() if h != "gsh" else "GSH"
            key = {"gsh": "GSH", "protein": "Protein", "cyanide": "Cyanide", "dna": "DNA"}[h]
            site_map = atom.get(key) or atom.get(h) or {}
            atom_pred = legacy_atom_vector(site_map, molecule.atoms.num, one_based=True)
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
