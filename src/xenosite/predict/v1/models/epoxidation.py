"""Epoxidation: two-stage bond site head then mol head; average two atom orderings."""

from __future__ import annotations

from typing import Any

import numpy as np

from xenosite.predict.backends.adapters import append_mol_bond, reorder_by_bond
from xenosite.predict.backends.onnx import OnnxBackend
from xenosite.predict.errors import WeightsNotFound
from ..features import bond_rows, load_names, matrix_from_rows, topn_site_features
from xenosite.predict.registry import register_model
from xenosite.predict.types import Molecule
from ._base import BaseRunner


class EpoxidationRunner(BaseRunner):
    name = "epoxidation"
    version = "0"
    onnx_heads = ("bond", "mol")

    def from_onnx(self, molecule: Molecule, backend: OnnxBackend) -> None:
        if not backend.has_head(self.name, "bond") or not backend.has_head(self.name, "mol"):
            raise WeightsNotFound(
                "epoxidation ONNX heads missing (bond + mol). Run make convert-onnx MODEL=epoxidation"
            )
        mol = self.rdkit_mol(molecule)
        bond_names = load_names("epoxidation", "bond")
        mol_names = load_names("epoxidation", "mol")

        scores_by_bond: dict[frozenset[int], list[float]] = {}
        mol_scores: list[float] = []
        for ordering in (True, False):
            rows = bond_rows(
                mol,
                original_atom_ordering=ordering,
                bond_nrings_mode=self.bond_nrings_mode(molecule),
            )
            x, _used = matrix_from_rows(rows, bond_names)
            if x.size == 0:
                continue
            y = backend.run_head(self.name, "bond", x).reshape(-1)
            for row, s in zip(rows, y):
                key = frozenset(row["_atoms"])  # type: ignore[arg-type]
                scores_by_bond.setdefault(key, []).append(float(s))
            if mol_names:
                mx = topn_site_features(
                    y,
                    rows,
                    mol_names,
                    score_suffix="__AtomScore",
                )
                mol_scores.append(
                    float(backend.run_head(self.name, "mol", mx).reshape(-1)[0])
                )

        averaged = {k: float(np.mean(v)) for k, v in scores_by_bond.items()}
        keys = list(averaged)
        pred = [averaged[k] for k in keys]
        bond_pred = self.symmetrize_bond_scores(
            molecule,
            reorder_by_bond(pred, keys, molecule.bonds.idx, fill=0.0),
        )

        mol_score = float(np.mean(mol_scores)) if mol_scores else 0.0
        if not mol_scores and averaged:
            mol_score = float(max(averaged.values()))

        append_mol_bond(
            molecule, model=self.name, version=self.version, mol=mol_score, bond=bond_pred
        )

    def from_legacy(self, molecule: Molecule, native: Any) -> None:
        """Legacy test-API: ``{"mol": float, "site": {"i-j": score, ...}}`` (0-based)."""
        mol_score = float(native.get("mol", 0.0))
        site = native.get("site") or native.get("bond") or {}
        current = []
        pred = []
        for key, val in site.items():
            if isinstance(key, str) and "-" in key:
                a, b = key.split("-", 1)
                current.append((int(a), int(b)))
            elif isinstance(key, (list, tuple)):
                current.append((int(key[0]), int(key[1])))
            else:
                continue
            pred.append(float(val))
        bond_pred = reorder_by_bond(pred, current, molecule.bonds.idx, fill=0.0)
        append_mol_bond(
            molecule, model=self.name, version=self.version, mol=mol_score, bond=bond_pred
        )


def _factory() -> EpoxidationRunner:
    return EpoxidationRunner()


register_model("epoxidation", "1", factory=_factory, two_stage=True, heads=("bond", "mol"))
