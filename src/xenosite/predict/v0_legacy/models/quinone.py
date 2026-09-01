"""Quinone: atom head → eligible pairs → pair head → mol head (two-stage mol)."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from xenosite.predict.backends.adapters import append_atom_pair, or_combine
from xenosite.predict.backends.onnx import OnnxBackend
from xenosite.predict.errors import WeightsNotFound
from ..features import load_names, matrix_from_rows, quinone_atom_rows
from ..features.quinone import eligible_atom_rows, quinone_mol_features, quinone_pair_rows
from xenosite.predict.registry import register_model
from xenosite.predict.types import Molecule
from ._base import BaseRunner


def _tf1_atom_scores(y: np.ndarray) -> np.ndarray:
    """Match TF1.15 float32 outputs ORT flushes to exact zero (quinone pair logit path)."""
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    return np.where(y == 0.0, 3.022989607e-08, y)


def _atom_scores_from_pairs(
    pair_idx: Sequence[tuple[int, int]],
    pair: Sequence[float],
    n_atoms: int,
) -> list[float]:
    """Per-atom scores from pair logits (legacy ``or_combine`` / ONNX path)."""
    collect: list[list[float]] = [[] for _ in range(n_atoms)]
    for (a, b), s in zip(pair_idx, pair):
        if 0 <= a < n_atoms:
            collect[a].append(float(s))
        if 0 <= b < n_atoms:
            collect[b].append(float(s))
    return [or_combine(p) for p in collect]


class QuinoneRunner(BaseRunner):
    name = "quinone"
    version = "0"
    onnx_heads = ("atom", "pair", "mol")

    def _quinone_omp_mode(self, molecule: Molecule) -> str:
        mode = molecule._parameter.get("quinone_omp_mode", "principled")
        return mode if mode in ("legacy", "principled") else "principled"

    def from_onnx(self, molecule: Molecule, backend: OnnxBackend) -> None:
        if not all(backend.has_head(self.name, h) for h in self.onnx_heads):
            raise WeightsNotFound(
                "quinone ONNX heads missing (atom, pair, mol). Run make convert-onnx MODEL=quinone"
            )
        mol = self.rdkit_mol(molecule)
        atom_rows = quinone_atom_rows(mol, omp_mode=self._quinone_omp_mode(molecule))
        eligible = eligible_atom_rows(atom_rows)
        if not eligible:
            append_atom_pair(
                molecule,
                model=self.name,
                version=self.version,
                mol=0.0,
                atom=[0.0] * molecule.atoms.num,
                pair=[],
                pair_idx=[],
            )
            return
        atom_names = load_names("quinone", "atom")
        x, _ = matrix_from_rows(eligible, atom_names)
        atom_scores = _tf1_atom_scores(backend.run_head(self.name, "atom", x).reshape(-1))

        ob_to_rdkit = {int(str(r["_index"]).split(".")[-1]): int(r["_atom"]) for r in atom_rows}
        atom_scores_by_ob = {
            int(str(r["_index"]).split(".")[-1]): float(s)
            for r, s in zip(eligible, atom_scores)
        }

        pair_rows = quinone_pair_rows(mol, atom_rows, atom_scores_by_ob)
        pair_names = load_names("quinone", "pair")
        pair_scores = np.zeros(len(pair_rows))
        if pair_rows:
            px, _ = matrix_from_rows(pair_rows, pair_names)
            pair_scores = backend.run_head(self.name, "pair", px).reshape(-1)

        pair_keys = [tuple(r["_atoms"]) for r in pair_rows]
        atom_pred = _atom_scores_from_pairs(pair_keys, pair_scores, molecule.atoms.num)

        mol_names = load_names("quinone", "mol")
        mol_score = 0.0
        if mol_names and len(pair_scores):
            mx = quinone_mol_features(atom_rows, pair_rows, pair_scores, mol_names)
            mol_score = float(backend.run_head(self.name, "mol", mx).reshape(-1)[0])

        append_atom_pair(
            molecule,
            model=self.name,
            version=self.version,
            mol=mol_score,
            atom=atom_pred,
            pair=[float(s) for s in pair_scores],
            pair_idx=pair_keys,
        )

    def from_legacy(self, molecule: Molecule, native: Any) -> None:
        from xenosite.predict.numbering import (
            build_group_to_rdkit_from_rows,
            build_ob_to_rdkit_from_rows,
            legacy_ob_order_from_rows,
            map_legacy_quinone_site_pair_to_rdkit,
        )

        raw_mol = native.get("mol", 0.0)
        if isinstance(raw_mol, dict) or raw_mol == {}:
            mol_score = 0.0
        else:
            mol_score = float(raw_mol or 0.0)
        site = native.get("site") or native.get("pair") or {}
        mol = self.rdkit_mol(molecule)
        rows = quinone_atom_rows(mol)
        group_to_rd = build_group_to_rdkit_from_rows(rows)
        ob_to_rd = build_ob_to_rdkit_from_rows(rows)
        row_ob_order = legacy_ob_order_from_rows(rows)
        n = molecule.atoms.num

        legacy_pair_scores: dict[tuple[int, int], float] = {}
        for key, val in (site.items() if isinstance(site, dict) else []):
            if isinstance(key, str) and "-" in key:
                ia, ib = int(key.split("-", 1)[0]), int(key.split("-", 1)[1])
            elif isinstance(key, (list, tuple)):
                ia, ib = int(key[0]), int(key[1])
            else:
                continue
            score = 0.0 if val == {} else float(val)
            rd = map_legacy_quinone_site_pair_to_rdkit(
                ia,
                ib,
                group_to_rd,
                zero_based_keys=True,
                ob_to_rd=ob_to_rd,
                legacy_ob_order=row_ob_order,
                n_heavy=n,
            )
            legacy_pair_scores[rd] = score

        atom_site = native.get("atom") or {}
        atom_scores_by_ob = {
            int(k): float(v if v != {} else 0.0)
            for k, v in (atom_site.items() if isinstance(atom_site, dict) else [])
        }

        pair_rows = quinone_pair_rows(mol, rows, atom_scores_by_ob)
        pair_idx = [tuple(r["_atoms"]) for r in pair_rows]
        pair = [legacy_pair_scores.get(idx, 0.0) for idx in pair_idx]
        atom_pred = _atom_scores_from_pairs(pair_idx, pair, n)

        append_atom_pair(
            molecule,
            model=self.name,
            version=self.version,
            mol=mol_score,
            atom=atom_pred,
            pair=pair,
            pair_idx=pair_idx,
        )


register_model(
    "quinone", "0", factory=lambda: QuinoneRunner(), two_stage=True, heads=("atom", "pair", "mol")
)
