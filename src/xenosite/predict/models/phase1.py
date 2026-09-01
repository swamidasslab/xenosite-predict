"""Phase I (molecularNN / TensorFlow). Converted to ONNX; no TensorFlow at runtime.

Site and mol heads are windowed MLPs dumped from the TF1 pickles. SMILES
inference runs Bond_and_LonePair rows (404 site.onnx inputs) then topology-group
max pooling, matching ``xenosite.api.v0.adapters.phase1`` index handling.
Possible_Sites SMARTS masks are not multiplied into class scores here (legacy
``model1`` applies them after the site net for ReactionType only).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..backends.adapters import append_atom_bond
from ..backends.onnx import OnnxBackend
from ..errors import WeightsNotFound
from ..features import load_names, matrix_from_rows, phase1_rows
from ..features.bond_lonepair import phase1_pymol
from ..features.phase1_mol import phase1_mol_features
from ..registry import register_model
from ..types import Molecule
from ._base import BaseRunner

PHASE1_HEADS = (
    "stable_oxygenation",
    "unstable_oxygenation",
    "dehydrogenation",
    "reduction",
    "hydrolysis",
)
_LEGACY_HEADS = (
    "StableOxygenation",
    "UnstableOxygenation",
    "Dehydrogenation",
    "Reduction",
    "Hydrolysis",
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


def _hydrogen_ids(pymol) -> set[str]:
    return {str(a.idx) for a in pymol.atoms if a.OBAtom.IsHydrogen()}


def _topology_pool(scores: np.ndarray, groups: list[str]) -> np.ndarray:
    """Legacy ``Class.groupby('TopologyGroup').max()`` broadcast back to rows."""
    out = np.array(scores, copy=True)
    for col in range(out.shape[1]):
        best: dict[str, float] = {}
        for g, val in zip(groups, out[:, col]):
            best[g] = max(best.get(g, 0.0), float(val))
        for i, g in enumerate(groups):
            out[i, col] = best[g]
    return out


def _site_targets(index: str, hydrogen_ids: set[str]) -> tuple[str, int] | tuple[str, frozenset[int]]:
    """Map a Bond_and_LonePair row index to RDKit atom or bond (0-based).

    Mirrors ``ApplyPyMolPredictors._process_phase1_preds`` + ``adapters.phase1``.
    """
    a1, a2 = index.split(".")[-2:]
    parts = ["h" if p in hydrogen_ids else p for p in (a1, a2)]
    heavy = [int(x) - 1 for x in parts if x != "h"]
    if len(set(heavy)) == 1:
        return ("atom", heavy[0])
    return ("bond", frozenset(heavy))


def _assign_scores(
    molecule: Molecule,
    rows: list[dict],
    site_scores: np.ndarray,
    hydrogen_ids: set[str],
) -> dict[str, tuple[list[float], list[float]]]:
    n_atom = molecule.atoms.num
    n_bond = len(molecule.bonds.idx)
    bond2idx = {frozenset(x): i for i, x in enumerate(molecule.bonds.idx)}
    out: dict[str, tuple[list[float], list[float]]] = {}
    for hi, head in enumerate(_LEGACY_HEADS):
        atom = [0.0] * n_atom
        bond = [0.0] * n_bond
        for row, score in zip(rows, site_scores[:, hi]):
            kind, key = _site_targets(str(row["_index"]), hydrogen_ids)
            s = float(score)
            if kind == "atom":
                if 0 <= key < n_atom:
                    atom[key] = max(s, atom[key])
            else:
                bi = bond2idx.get(key)
                if bi is not None:
                    bond[bi] = max(s, bond[bi])
        out[head] = (atom, bond)
    return out


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
        mol = self.rdkit_mol(molecule)
        rows = phase1_rows(mol)
        site_names = load_names("phase1", "site")
        if not site_names:
            from ..features.phase1_mol import phase1_site_column_names

            site_names = phase1_site_column_names(rows[0])
        x, _ = matrix_from_rows(rows, site_names)
        if x.size == 0:
            raise ValueError("phase1 produced no Bond_and_LonePair rows")

        site_scores = np.asarray(backend.run_head(self.name, "site", x), dtype=np.float64)
        if site_scores.ndim == 1:
            site_scores = site_scores.reshape(-1, len(_LEGACY_HEADS))
        groups = [str(r["_index"]).split(".")[1] for r in rows]
        site_scores = _topology_pool(site_scores, groups)

        pymol = phase1_pymol(mol)
        per_head = _assign_scores(molecule, rows, site_scores, _hydrogen_ids(pymol))

        mol_names = load_names("phase1", "mol")
        if mol_names:
            mx = phase1_mol_features(site_scores, rows, mol_names)
            backend.run_head(self.name, "mol", mx)

        for h in _LEGACY_HEADS:
            atom, bond = per_head[h]
            append_atom_bond(
                molecule,
                model=f"phase1.{_snake(h)}",
                version=self.version,
                atom=atom,
                bond=bond,
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
