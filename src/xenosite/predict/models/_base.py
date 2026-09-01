"""Shared model-runner helpers."""

from __future__ import annotations

from typing import Any

from ..backends.onnx import OnnxBackend
from ..errors import ModelNotAvailable, WeightsNotFound
from ..molecule import parse_smiles
from ..registry import ModelRunner
from ..types import Molecule as Mol


class BaseRunner(ModelRunner):
    name: str = ""
    version: str = "0"
    onnx_heads: tuple[str, ...] = ()
    blocked_reason: str | None = None

    def available(self, backend) -> bool:
        if self.blocked_reason:
            return False
        if getattr(backend, "name", None) == "onnx":
            if self.name not in ("bioactivation",):
                from ..features import _ob

                if not _ob.installed():
                    return False
            key = "ndealk" if self.name in ("ndealk", "isozyme") else self.name
            return all(backend.has_head(key, h) for h in self.onnx_heads)
        specs = set(backend.available_models())
        return (self.name, self.version) in specs

    def predict_molecule(self, molecule: Mol, backend) -> None:
        if self.blocked_reason:
            raise ModelNotAvailable(self.blocked_reason)
        bname = getattr(backend, "name", "")
        if bname == "http":
            self._from_http(molecule, backend)
            return
        if bname == "legacy":
            native = backend.predict_native(molecule.smiles, self.name, self.version)
            self.from_legacy(molecule, native)
            return
        if bname == "onnx" or isinstance(backend, OnnxBackend):
            self.from_onnx(molecule, backend)
            return
        # Unknown backend: try native then HTTP-shaped
        native = backend.predict_native(molecule.smiles, self.name, self.version)
        if isinstance(native, dict) and "results" in native:
            self._merge_molecule_payload(molecule, native)
        else:
            self.from_legacy(molecule, native)

    def from_onnx(self, molecule: Mol, backend: OnnxBackend) -> None:
        raise WeightsNotFound(
            f"{self.name} ONNX path is not implemented or weights are missing"
        )

    def from_legacy(self, molecule: Mol, native: Any) -> None:
        raise NotImplementedError(f"{self.name}: no legacy adapter")

    def _from_http(self, molecule: Mol, backend) -> None:
        payload = backend.predict_native(molecule.smiles, self.name, self.version)
        self._merge_molecule_payload(molecule, payload)

    def _merge_molecule_payload(self, molecule: Mol, payload: dict) -> None:
        incoming = Mol.model_validate(payload)
        molecule.results.extend(incoming.results)

    def rdkit_mol(self, molecule: Mol):
        mol, _ = parse_smiles(molecule.smiles)
        return mol
