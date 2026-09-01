"""Bioactivation last: metabolite enumeration pipeline, not a single ONNX.

Calls other models (phase1/APMP, reactivity, …) then path/mol heads.
Golden fixtures do not exist in xenosite-api today; add them here when gathered.
"""

from __future__ import annotations

from typing import Any

from xenosite.predict.backends.adapters import append_mol_atom, or_combine
from xenosite.predict.backends.onnx import OnnxBackend
from xenosite.predict.errors import ModelNotAvailable
from xenosite.predict.registry import register_model
from xenosite.predict.types import Metabolite, Molecule
from ._base import BaseRunner

_BLOCKED = (
    "bioactivation is a metabolite-enumeration pipeline (not one ONNX). "
    "Port remaining models first; ONNX mol/path heads are not sufficient alone."
)


class BioactivationRunner(BaseRunner):
    name = "bioactivation"
    version = "0"
    onnx_heads = ("mol", "path")
    blocked_reason = None  # HTTP/legacy allowed; ONNX raises until pipeline lands

    def from_onnx(self, molecule: Molecule, backend: OnnxBackend) -> None:
        raise ModelNotAvailable(_BLOCKED)

    def from_legacy(self, molecule: Molecule, native: Any) -> None:
        data = native.get("data") or native
        mol_score = float(data.get("MBS") or native.get("MBS") or 0.0)
        pbs = data.get("PBS") or native.get("PBS") or {}
        metabolites: list[Metabolite] = []
        atom_lists: list[list[float]] = [[] for _ in range(molecule.atoms.num)]
        items = pbs.items() if isinstance(pbs, dict) else []
        for key, value in sorted(items, key=lambda x: x[1], reverse=True):
            parts = str(key).split("_")
            if len(parts) < 3:
                continue
            idx_s, pathway, smi = parts[0], parts[1], "_".join(parts[2:])
            idx = [int(i) - 1 for i in idx_s.split(".")][1:]
            metabolites.append(
                Metabolite(
                    atom=[i + 1 for i in idx] if idx else None,
                    pathway=pathway,
                    smiles=smi,
                    score=float(value),
                )
            )
            for i in idx:
                if 0 <= i < len(atom_lists):
                    atom_lists[i].append(float(value))
        atom = [or_combine(x) for x in atom_lists]
        append_mol_atom(
            molecule,
            model=self.name,
            version=self.version,
            mol=mol_score,
            atom=atom,
            metabolite=metabolites,
        )


register_model(
    "bioactivation",
    "0",
    factory=lambda: BioactivationRunner(),
    pipeline=True,
    heads=("mol", "path"),
)
