"""UGT: atom-level topological descriptors + one ONNX head. No MOPAC/SmartCYP."""

from __future__ import annotations

from typing import Any

from ..backends.adapters import append_atom
from ..backends.onnx import OnnxBackend
from ..errors import WeightsNotFound
from ..features import load_names, matrix_from_rows, ugt_atom_rows
from ..registry import register_model
from ..types import Molecule
from ._base import BaseRunner


class UgtRunner(BaseRunner):
    name = "ugt"
    version = "0"
    onnx_heads = ("atom",)

    def from_onnx(self, molecule: Molecule, backend: OnnxBackend) -> None:
        if not backend.has_head(self.name, "atom"):
            raise WeightsNotFound("ugt ONNX missing atom. Run make convert-onnx MODEL=ugt")
        mol = self.rdkit_mol(molecule)
        rows = ugt_atom_rows(mol)
        names = load_names("ugt", "atom")
        x, _ = matrix_from_rows(rows, names)
        y = backend.run_head(self.name, "atom", x).reshape(-1)
        atom_pred = [0.0] * molecule.atoms.num
        for row, s in zip(rows, y):
            i = int(row["_atom"])
            if 0 <= i < len(atom_pred):
                atom_pred[i] = float(s)
        append_atom(molecule, model=self.name, version=self.version, atom=atom_pred)

    def from_legacy(self, molecule: Molecule, native: Any) -> None:
        atom_map = native.get("atom") or native.get("xenosite") or native
        atom_pred = [0.0] * molecule.atoms.num
        if isinstance(atom_map, dict):
            for k, v in atom_map.items():
                try:
                    i = int(k)
                except (TypeError, ValueError):
                    continue
                if 0 <= i < len(atom_pred):
                    atom_pred[i] = float(v) if not isinstance(v, dict) else float(next(iter(v.values()), 0.0))
        append_atom(molecule, model=self.name, version=self.version, atom=atom_pred)


register_model("ugt", "0", factory=lambda: UgtRunner(), heads=("atom",))
