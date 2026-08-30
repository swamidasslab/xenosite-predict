"""Quinone: atom head → eligible pairs → pair head → mol head (two-stage mol)."""

from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np
from rdkit import Chem

from ..backends.adapters import append_atom_pair, or_combine
from ..backends.onnx import OnnxBackend
from ..errors import WeightsNotFound
from ..features import load_names, matrix_from_rows, reactivity_atom_rows, topn_site_features
from ..registry import register_model
from ..types import Molecule
from ._base import BaseRunner


class QuinoneRunner(BaseRunner):
    name = "quinone"
    version = "0"
    onnx_heads = ("atom", "pair", "mol")

    def from_onnx(self, molecule: Molecule, backend: OnnxBackend) -> None:
        if not all(backend.has_head(self.name, h) for h in self.onnx_heads):
            raise WeightsNotFound(
                "quinone ONNX heads missing (atom, pair, mol). Run make convert-onnx MODEL=quinone"
            )
        mol = self.rdkit_mol(molecule)
        atom_rows = reactivity_atom_rows(mol)
        atom_names = load_names("quinone", "atom")
        x, _ = matrix_from_rows(atom_rows, atom_names)
        atom_scores = backend.run_head(self.name, "atom", x).reshape(-1)
        atom_idx = [int(r["_atom"]) for r in atom_rows]

        pair_names = load_names("quinone", "pair")
        pair_rows = []
        pair_keys: list[tuple[int, int]] = []
        dist = Chem.GetDistanceMatrix(mol)
        for i, j in combinations(range(len(atom_idx)), 2):
            a, b = atom_idx[i], atom_idx[j]
            d = float(dist[a, b])
            row = {
                "Atom1_Pred": float(atom_scores[i]),
                "Atom2_Pred": float(atom_scores[j]),
                "AtomPair__Distance": d,
                "AtomPair__Distance_Is_Odd": float(int(d) % 2),
            }
            pair_rows.append(row)
            pair_keys.append(tuple(sorted((a, b))))

        pair_scores = np.zeros(len(pair_rows))
        if pair_rows:
            px, _ = matrix_from_rows(pair_rows, pair_names)
            pair_scores = backend.run_head(self.name, "pair", px).reshape(-1)

        collect = [[] for _ in range(molecule.atoms.num)]
        for (a, b), s in zip(pair_keys, pair_scores):
            collect[a].append(float(s))
            collect[b].append(float(s))
        atom_pred = [or_combine(p) for p in collect]

        mol_names = load_names("quinone", "mol")
        mol_score = 0.0
        if mol_names and len(pair_scores):
            mx = topn_site_features(pair_scores, pair_rows, mol_names)
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
        mol_score = float(native.get("mol", 0.0))
        site = native.get("site") or native.get("pair") or {}
        pair_idx = []
        pair = []
        for key, val in (site.items() if isinstance(site, dict) else []):
            if isinstance(key, str) and "-" in key:
                a, b = key.split("-", 1)
                pair_idx.append((int(a), int(b)))
            elif isinstance(key, (list, tuple)):
                pair_idx.append((int(key[0]), int(key[1])))
            else:
                continue
            pair.append(0.0 if val == {} else float(val))
        collect = [[] for _ in range(molecule.atoms.num)]
        for (a, b), s in zip(pair_idx, pair):
            if isinstance(a, int) and 0 <= a < len(collect):
                collect[a].append(s)
            if isinstance(b, int) and 0 <= b < len(collect):
                collect[b].append(s)
        atom_pred = [or_combine(p) for p in collect]
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
