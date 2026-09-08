"""Per-row phase1 Class lookup (XenoNet ``get_prob_for_one_site``).

v0 clones the Python-2 dataframe walk, including the in-place site-swap on
missed bond rows. Missing sites yield ``None`` (callers treat that as weight 0).
``EpoxideOpening`` is hard-coded to 1.0.

v1 looks up the same reaction on **max-pooled** atom/bond vectors instead of
raw Bond_and_LonePair rows — the pooling ``Phase1Runner`` already applies for
``predict()``. Do not use :func:`xenosite.predict.forest.site_score` for v0.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from ._unvalidated import warn_unvalidated

warn_unvalidated()

import numpy as np

from xenosite.predict.backends.onnx import OnnxBackend
from xenosite.predict.v0_legacy.features import (
    load_names,
    matrix_from_rows,
    phase1_rows,
)
from xenosite.predict.v0_legacy.features.phase1_mol import phase1_site_column_names
from xenosite.predict.v0_legacy.models.phase1 import (
    _LEGACY_HEADS,
    _hydrogen_ids,
    _topology_pool,
)
from xenosite.predict.v0_legacy.features.bond_lonepair import phase1_pymol
from xenosite.predict.v0_legacy.symmetry import collapse_opposite_direction_rows

# Reaction rule name → phase1 Class column (XenoNet ``rxn_dict``).
RXN_DICT: dict[str, str] = {
    "Dephosphorylation": "Hydrolysis",
    "EpoxideOpening": "EpoxideOpening",
    "Hydrolysis": "Hydrolysis",
    "Dehydrogenation": "Dehydrogenation",
    "NitroaromaticReduction": "Reduction",
    "NitrogenReduction": "Reduction",
    "Dehydration": "Reduction",
    "SulfurReduction": "Reduction",
    "OxygenReduction": "Reduction",
    "ReductiveDehalogenation": "Reduction",
    "Hydrogenation": "Reduction",
    "Epoxidation": "StableOxygenation",
    "Hydroxylation": "StableOxygenation",
    "SulfurOxidation": "StableOxygenation",
    "NitrogenOxidation": "StableOxygenation",
    "Dealkylation": "UnstableOxygenation",
    "OxidativeDehalogenation": "UnstableOxygenation",
    "QuinoneFormation": "QuinoneFormation",
}

# Rule name → ``phase1_sites_on`` (Metabolic Forest Phase I table).
PHASE1_SITES_ON: dict[str, str] = {
    "Dealkylation": "bonds",
    "Dehydration": "bonds",
    "Dehydrogenation": "atom_hydrogen",
    "Dephosphorylation": "bonds",
    "Epoxidation": "bonds",
    "EpoxideOpening": "bonds",
    "Hydrogenation": "atoms",
    "Hydrolysis": "bonds",
    "Hydroxylation": "atom_hydrogen",
    "NitrogenOxidation": "atoms",
    "NitrogenReduction": "bonds",
    "OxidativeDehalogenation": "bonds",
    "OxygenReduction": "bonds",
    "ReductiveDehalogenation": "bonds",
    "SulfurOxidation": "atoms",
    "SulfurReduction": "bonds",
}


@dataclass
class Phase1SiteTable:
    """Unpooled Class columns plus optional pooled atom/bond maps for v1."""

    index: list[str]
    scores: dict[str, list[float]]
    n_heavy: int
    atom: dict[str, list[float]] = field(default_factory=dict)
    bond: dict[str, dict[frozenset[int], float]] = field(default_factory=dict)


def phase1_site_strings(rule_name: str, site_rdkit_zero: frozenset[int]) -> frozenset[str]:
    """``convert_site_to_phase1_format``: 0-based RDKit → 1-based ``N.h`` / ``a.b``."""
    kind = PHASE1_SITES_ON.get(rule_name.split("_", 1)[0], "bonds")
    ones = [i + 1 for i in site_rdkit_zero]
    if kind == "atom_hydrogen":
        return frozenset(f"{s}.h" for s in ones)
    if kind == "atoms":
        return frozenset(f"{s}.{s}" for s in ones)
    if len(ones) == 2:
        a, b = ones[0], ones[1]
        return frozenset([f"{a}.{b}"])
    return frozenset(str(s) for s in ones)


def get_prob_for_one_site(
    table: Phase1SiteTable,
    rxn_site: str,
    rxn_type: str,
    num_heavy_atoms: Optional[int] = None,
) -> Optional[float]:
    """Clone of XenoNet ``BioActNetwork.get_prob_for_one_site`` (phase1 branch)."""
    if rxn_type == "EpoxideOpening":
        return 1.0
    if rxn_type == "QuinoneFormation":
        return None

    col = table.scores.get(rxn_type)
    if col is None:
        return None
    n_heavy = num_heavy_atoms if num_heavy_atoms is not None else table.n_heavy
    split_rxn_site = rxn_site.split(".")
    if len(split_rxn_site) < 2:
        return None

    if split_rxn_site[1] == "h":
        for count, idx in enumerate(table.index):
            parts = idx.split(".")
            if len(parts) < 4:
                continue
            if split_rxn_site[0] == parts[2] and int(parts[3]) > n_heavy:
                return float(col[count])
        return None

    for count, idx in enumerate(table.index):
        parts = idx.split(".")
        if len(parts) < 4:
            continue
        if parts[2] == split_rxn_site[0] and parts[3] == split_rxn_site[1]:
            return float(col[count])
        split_rxn_site[0], split_rxn_site[1] = split_rxn_site[1], split_rxn_site[0]
        if parts[2] == split_rxn_site[0] and parts[3] == split_rxn_site[1]:
            return float(col[count])
    return None


def edge_weight_v0(
    table: Phase1SiteTable,
    site_strings: frozenset[str] | list[str],
    rxn_type: str,
) -> float:
    """Product of per-site scores. Missing site → 0. Stored weight is positive."""
    rxn_prob = -1.0
    for rxn in site_strings:
        try:
            p = get_prob_for_one_site(table, str(rxn), rxn_type, table.n_heavy)
            if p is None:
                return 0.0
            rxn_prob *= float(p)
        except (TypeError, ValueError, IndexError):
            return 0.0
    return -rxn_prob


def edge_weight_v1(
    table: Phase1SiteTable,
    site_rdkit_zero: frozenset[int],
    rxn_type: str,
) -> float:
    """Pooled atom/bond lookup (principled mapping of the same ONNX rows)."""
    if rxn_type == "EpoxideOpening":
        return 1.0
    if rxn_type == "QuinoneFormation":
        return 0.0
    atoms = sorted(site_rdkit_zero)
    atom_col = table.atom.get(rxn_type)
    bond_col = table.bond.get(rxn_type) or {}
    if len(atoms) == 2:
        w = bond_col.get(frozenset(atoms))
        if w is not None:
            return float(w)
        if atom_col:
            vals = [atom_col[a] for a in atoms if 0 <= a < len(atom_col)]
            return float(max(vals)) if vals else 0.0
        return 0.0
    if atom_col and atoms:
        vals = [atom_col[a] for a in atoms if 0 <= a < len(atom_col)]
        if not vals:
            return 0.0
        rxn_prob = -1.0
        for v in vals:
            rxn_prob *= float(v)
        return -rxn_prob
    return 0.0


def edge_weight(
    table: Phase1SiteTable,
    *,
    site_strings: frozenset[str] | list[str],
    site_rdkit_zero: frozenset[int],
    rxn_type: str,
    scoring: str = "0",
) -> float:
    if scoring == "1":
        return edge_weight_v1(table, site_rdkit_zero, rxn_type)
    return edge_weight_v0(table, site_strings, rxn_type)


def _pooled_atom_bond(
    rows: list[dict],
    site_scores: np.ndarray,
    n_heavy: int,
    hydrogen_ids: set[str],
) -> tuple[dict[str, list[float]], dict[str, dict[frozenset[int], float]]]:
    atom: dict[str, list[float]] = {}
    bond: dict[str, dict[frozenset[int], float]] = {}
    for hi, head in enumerate(_LEGACY_HEADS):
        atom_vec = [0.0] * n_heavy
        bond_map: dict[frozenset[int], float] = {}
        for row, score in zip(rows, site_scores[:, hi]):
            idx = str(row["_index"])
            a1, a2 = idx.split(".")[-2:]
            parts = ["h" if p in hydrogen_ids else p for p in (a1, a2)]
            heavy = [int(x) - 1 for x in parts if x != "h"]
            s = float(score)
            if len(set(heavy)) == 1:
                k = heavy[0]
                if 0 <= k < n_heavy:
                    atom_vec[k] = max(s, atom_vec[k])
            elif len(heavy) >= 2:
                key = frozenset(heavy[:2])
                bond_map[key] = max(s, bond_map.get(key, 0.0))
        atom[head] = atom_vec
        bond[head] = bond_map
    return atom, bond


def phase1_site_table(rdmol, backend: OnnxBackend) -> Phase1SiteTable:
    """Run phase1 site ONNX; keep topology-pooled **rows**, not atom/bond vectors."""
    n_heavy = rdmol.GetNumHeavyAtoms() if hasattr(rdmol, "GetNumHeavyAtoms") else rdmol.GetNumAtoms()
    if not backend.has_head("phase1", "site"):
        from xenosite.predict.errors import WeightsNotFound

        raise WeightsNotFound("phase1 ONNX missing site head")

    rows = phase1_rows(rdmol)
    names = load_names("phase1", "site")
    if not names and rows:
        names = phase1_site_column_names(rows[0])
    x, _ = matrix_from_rows(rows, names)
    if x.size == 0:
        return Phase1SiteTable(index=[], scores={h: [] for h in _LEGACY_HEADS}, n_heavy=n_heavy)

    site_scores = np.asarray(backend.run_head("phase1", "site", x), dtype=np.float64)
    if site_scores.ndim == 1:
        site_scores = site_scores.reshape(-1, len(_LEGACY_HEADS))
    rows, site_scores = collapse_opposite_direction_rows(rows, site_scores)
    groups = [str(r["_index"]).split(".")[1] for r in rows]
    site_scores = _topology_pool(site_scores, groups)

    index = [str(r["_index"]) for r in rows]
    scores = {
        head: [float(site_scores[i, hi]) for i in range(len(rows))]
        for hi, head in enumerate(_LEGACY_HEADS)
    }
    pymol = phase1_pymol(rdmol)
    hydrogen_ids = _hydrogen_ids(pymol)
    atom, bond = _pooled_atom_bond(rows, site_scores, n_heavy, hydrogen_ids)
    return Phase1SiteTable(
        index=index,
        scores=scores,
        n_heavy=n_heavy,
        atom=atom,
        bond=bond,
    )


def reaction_type(rule_name: str) -> str:
    base = rule_name.split("_", 1)[0]
    return RXN_DICT.get(base, base)
