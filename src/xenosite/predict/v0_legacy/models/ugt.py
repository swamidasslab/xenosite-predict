"""UGT: atom-level topological descriptors + one ONNX head. No MOPAC/SmartCYP."""

from __future__ import annotations

from typing import Any

from xenosite.predict.backends.adapters import append_atom, legacy_atom_vector
from xenosite.predict.backends.onnx import OnnxBackend
from xenosite.predict.errors import WeightsNotFound
from ..features import load_names, matrix_from_rows
from ..features.ugt import ugt_inference_rows
from xenosite.predict.registry import register_model
from xenosite.predict.types import Molecule
from ._base import BaseRunner


class UgtRunner(BaseRunner):
    name = "ugt"
    version = "0"
    onnx_heads = ("atom",)

    def from_onnx(self, molecule: Molecule, backend: OnnxBackend) -> None:
        if not backend.has_head(self.name, "atom"):
            raise WeightsNotFound("ugt ONNX missing atom. Run make convert-onnx MODEL=ugt")
        mol = self.rdkit_mol(molecule)
        rows = ugt_inference_rows(mol)
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
        atom_map = (
            native.get("XenoSite")
            or native.get("xenosite")
            or native.get("atom")
            or native
        )
        if not isinstance(atom_map, dict):
            atom_map = {}
        atom_pred = legacy_atom_vector(atom_map, molecule.atoms.num, one_based=True)
        append_atom(molecule, model=self.name, version=self.version, atom=atom_pred)


register_model("ugt", "0", factory=lambda: UgtRunner(), heads=("atom",))
