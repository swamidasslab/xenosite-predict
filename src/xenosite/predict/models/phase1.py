"""Phase I (molecularNN / TensorFlow). Converted to ONNX; no TensorFlow at runtime.

Site and mol heads are windowed MLPs dumped from the TF1 pickles. SMILES
inference still needs Bond_and_LonePair descriptors (not ported yet).
HTTP/legacy backends still work.
"""

from __future__ import annotations

from typing import Any

from ..backends.adapters import append_atom_bond
from ..backends.onnx import OnnxBackend
from ..errors import WeightsNotFound
from ..registry import register_model
from ..types import Molecule
from ._base import BaseRunner

PHASE1_HEADS = (
    "stable_oxygenation",
    "unstable_oxygenation",
    "dehydrogenation",
    "hydrolysis",
    "reduction",
)
_LEGACY_HEADS = (
    "StableOxygenation",
    "UnstableOxygenation",
    "Dehydrogenation",
    "Hydrolysis",
    "Reduction",
)


def _snake(s: str) -> str:
    try:
        from camel_converter import to_snake as _ts

        return _ts(s)
    except ImportError:
        out = []
        for c in s:
            if c.isupper() and out:
                out.append("_")
            out.append(c.lower())
        return "".join(out)


class Phase1Runner(BaseRunner):
    name = "phase1"
    version = "0"
    onnx_heads = ("site", "mol")

    def from_onnx(self, molecule: Molecule, backend: OnnxBackend) -> None:
        if not backend.has_head(self.name, "site") or not backend.has_head(self.name, "mol"):
            raise WeightsNotFound(
                "phase1 ONNX missing (site + mol). Run `make convert-onnx MODEL=phase1`. "
                "TF is not a runtime dep."
            )
        raise WeightsNotFound(
            "phase1 ONNX site/mol heads exist, but Bond_and_LonePair descriptors "
            "are not ported yet, so SMILES inference cannot run."
        )

    def from_legacy(self, molecule: Molecule, native: Any) -> None:
        data = native.get("data") or native
        idx = data.get("index") or []
        n_atom = molecule.atoms.num
        n_bond = len(molecule.bonds.idx)
        bond2idx = {frozenset(x): i for i, x in enumerate(molecule.bonds.idx)}
        for h in _LEGACY_HEADS:
            atom = [0.0] * n_atom
            bond = [0.0] * n_bond
            scores = data.get(h) or []
            for i, r in zip(idx, scores):
                atoms = [int(x) - 1 for x in i if x != "h"] if isinstance(i, (list, tuple)) else []
                if len(atoms) == 1:
                    atom[atoms[0]] = max(float(r), atom[atoms[0]])
                elif len(atoms) >= 2:
                    bi = bond2idx.get(frozenset(atoms[:2]))
                    if bi is not None:
                        bond[bi] = max(float(r), bond[bi])
            append_atom_bond(
                molecule,
                model=f"phase1.{_snake(h)}",
                version=self.version,
                atom=atom,
                bond=bond,
            )


register_model(
    "phase1",
    "0",
    factory=lambda: Phase1Runner(),
    heads=PHASE1_HEADS,
    two_stage=True,
)
