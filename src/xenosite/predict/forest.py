"""Thin adapter to ``xenosite.forest`` for SOM → metabolite structure inference.

Site-of-metabolism *scores* come from :func:`predict`; structure enumeration is
delegated to `Metabolic Forest <https://github.com/swamidasslab/xenosite-forest>`_.

``xenosite.forest`` is still v0. Two indexing conventions are detected separately:

- **Site frozensets** (:func:`forest_site_indexing`) — probe ``CC`` + ``SO``.
- **AtomTracker parent maps** (:func:`forest_map_indexing`) — probe ``CC`` +
  ``SO`` product ``react_atom_idx`` vs ``old_mapno``.

Sites are normalized to 0-based RDKit on the parent; ``Metabolite.map_idx`` uses
1-based parent atom numbers (0 = new atom), per forest AtomTracker conventions.
"""

from __future__ import annotations

import os
from enum import Enum
from functools import lru_cache
from typing import Iterator, Optional

from rdkit import Chem
from xenosite.forest import load_ruleset
from xenosite.forest.base import can_smi

from .molecule import parse_smiles
from .types import (
    AtomBondResult,
    BondResult,
    Metabolite,
    MolAtomPairResult,
    MolBondResult,
    ModelResult,
    Molecule,
)

# ``result.model`` → ``load_ruleset(...)`` specifier (see xenosite-forest docs/rulesets.md).
_MODEL_RULESETS: dict[str, str] = {
    "epoxidation": "SO.Epoxidation",
    "ndealk": "UO.Dealkylation",
    "quinone": "QF.QuinoneFormation",
    "phase1.stable_oxygenation": "SO",
    "phase1.unstable_oxygenation": "UO",
    "phase1.dehydrogenation": "DH",
    "phase1.reduction": "RD",
    "phase1.hydrolysis": "HD",
}

_INDEX_PROBE_SMILES = "CC"
_INDEX_PROBE_RULESET = "SO"
_ENV_SITE_INDEXING = "XENOSITE_FOREST_SITE_INDEXING"
_ENV_MAP_INDEXING = "XENOSITE_FOREST_MAP_INDEXING"
_REACT_ATOM_IDX = "react_atom_idx"
_OLD_MAPNO = "old_mapno"


class ForestSiteIndexing(str, Enum):
    """How ``xenosite.forest`` labels sites in ``RuleSet.metabolites`` frozensets."""

    RDKIT_ZERO = "rdkit_zero"
    ATOM_NUMBER_ONE = "atom_number_one"


class ForestMapIndexing(str, Enum):
    """How AtomTracker ``react_atom_idx`` relates to 1-based parent atom numbers."""

    RDKIT_ZERO = "rdkit_zero"
    ATOM_NUMBER_ONE = "atom_number_one"


def ruleset_for_model(model: str) -> Optional[str]:
    """Return a forest ruleset name for a predict result model, if supported."""
    if model in _MODEL_RULESETS:
        return _MODEL_RULESETS[model]
    if model.startswith("isozyme."):
        return "UO.Dealkylation"
    return None


def pathway_name(rule: str) -> str:
    """Normalize forest rule labels (``Hydroxylation_Smarts...`` → ``Hydroxylation``)."""
    return rule.split("_", 1)[0]


def site_rdkit_indices(site: frozenset[int]) -> list[int]:
    """Sorted 0-based RDKit indices for a site (after :func:`forest_site_to_rdkit`)."""
    return sorted(site)


def _site_indexing_from_env() -> Optional[ForestSiteIndexing]:
    raw = os.environ.get(_ENV_SITE_INDEXING, "").strip().lower()
    if raw in ("rdkit_zero", "rdkit", "0", "zero"):
        return ForestSiteIndexing.RDKIT_ZERO
    if raw in ("atom_number_one", "one", "1", "phase1"):
        return ForestSiteIndexing.ATOM_NUMBER_ONE
    return None


def _map_indexing_from_env() -> Optional[ForestMapIndexing]:
    raw = os.environ.get(_ENV_MAP_INDEXING, "").strip().lower()
    if raw in ("rdkit_zero", "rdkit", "0", "zero"):
        return ForestMapIndexing.RDKIT_ZERO
    if raw in ("atom_number_one", "one", "1", "phase1"):
        return ForestMapIndexing.ATOM_NUMBER_ONE
    return None


def _probe_forest_site_indexing() -> ForestSiteIndexing:
    """Detect forest site numbering using ``CC`` + stable oxygenation (``SO``)."""
    forced = _site_indexing_from_env()
    if forced is not None:
        return forced

    mol = Chem.MolFromSmiles(_INDEX_PROBE_SMILES)
    if mol is None:
        return ForestSiteIndexing.RDKIT_ZERO
    n_heavy = mol.GetNumAtoms()

    rs = load_ruleset(_INDEX_PROBE_RULESET)
    saw_zero = False
    saw_one_based_high = False
    for (_rule, site), _mols in rs.metabolites(mol, unique=True):
        for idx in site:
            if idx == 0:
                saw_zero = True
            if idx >= n_heavy:
                saw_one_based_high = True

    if saw_zero:
        return ForestSiteIndexing.RDKIT_ZERO
    if saw_one_based_high:
        return ForestSiteIndexing.ATOM_NUMBER_ONE
    return ForestSiteIndexing.RDKIT_ZERO


def _probe_forest_map_indexing() -> ForestMapIndexing:
    """Detect AtomTracker parent indexing from a ``CC`` + ``SO`` hydroxylation product."""
    forced = _map_indexing_from_env()
    if forced is not None:
        return forced

    mol = Chem.MolFromSmiles(_INDEX_PROBE_SMILES)
    if mol is None:
        return ForestMapIndexing.RDKIT_ZERO

    rs = load_ruleset(_INDEX_PROBE_RULESET)
    for (_rule, _site), mols in rs.metabolites(mol, unique=True):
        product = mols[-1]
        for atom in product.GetAtoms():
            if atom.GetAtomicNum() == 1:
                continue
            if not atom.HasProp(_REACT_ATOM_IDX):
                continue
            react_idx = int(atom.GetProp(_REACT_ATOM_IDX))
            if react_idx == 0:
                return ForestMapIndexing.RDKIT_ZERO
            if atom.HasProp(_OLD_MAPNO):
                old_mapno = int(atom.GetProp(_OLD_MAPNO))
                if old_mapno == react_idx:
                    return ForestMapIndexing.ATOM_NUMBER_ONE
                if old_mapno == react_idx + 1:
                    return ForestMapIndexing.RDKIT_ZERO
        break

    return ForestMapIndexing.RDKIT_ZERO


@lru_cache(maxsize=8)
def forest_site_indexing_for_version(version: str) -> ForestSiteIndexing:
    return _probe_forest_site_indexing()


@lru_cache(maxsize=8)
def forest_map_indexing_for_version(version: str) -> ForestMapIndexing:
    return _probe_forest_map_indexing()


def forest_site_indexing() -> ForestSiteIndexing:
    """Cached site-index convention for the installed ``xenosite.forest`` version."""
    import xenosite.forest as xf

    version = getattr(xf, "__version__", "unknown")
    return forest_site_indexing_for_version(version)


def forest_map_indexing() -> ForestMapIndexing:
    """Cached AtomTracker parent-map convention for the installed forest version."""
    import xenosite.forest as xf

    version = getattr(xf, "__version__", "unknown")
    return forest_map_indexing_for_version(version)


def forest_site_to_rdkit(
    site: frozenset[int],
    n_atoms: int,
    mode: Optional[ForestSiteIndexing] = None,
) -> frozenset[int]:
    """Convert a forest site frozenset to 0-based RDKit atom indices."""
    if mode is None:
        mode = forest_site_indexing()
    if mode == ForestSiteIndexing.ATOM_NUMBER_ONE:
        out = frozenset(i - 1 for i in site)
    else:
        out = site
    for idx in out:
        if idx < 0 or idx >= n_atoms:
            raise ValueError(
                f"forest site {sorted(site)} ({mode.value}) maps to invalid "
                f"RDKit index {idx} for {n_atoms} heavy atoms"
            )
    return out


def _parent_map_number(atom: Chem.Atom, mode: Optional[ForestMapIndexing] = None) -> int:
    """1-based parent atom number for a product heavy atom, or 0 if newly added."""
    if mode is None:
        mode = forest_map_indexing()
    if not atom.HasProp(_REACT_ATOM_IDX):
        return 0
    react_idx = int(atom.GetProp(_REACT_ATOM_IDX))
    if mode == ForestMapIndexing.RDKIT_ZERO:
        return react_idx + 1
    return react_idx


def metabolite_atom_maps(
    product: Chem.Mol,
    *,
    mode: Optional[ForestMapIndexing] = None,
    mapped_smiles: bool = False,
) -> tuple[list[int], Optional[str]]:
    """Return ``(map_idx, mapped_smiles)`` from a forest product mol.

    ``map_idx`` is always computed (1-based parent atom numbers per heavy atom in
    canonical ``smiles`` order; 0 = new atom). ``mapped_smiles`` is included only
    when requested — SMILES with ``:N`` map numbers tracing to the parent.
    """
    if mode is None:
        mode = forest_map_indexing()

    tagged = Chem.Mol(product)
    for atom in tagged.GetAtoms():
        if atom.GetAtomicNum() == 1:
            continue
        parent = _parent_map_number(atom, mode)
        if parent > 0:
            atom.SetAtomMapNum(parent)

    mapped = can_smi(rdmol=tagged)[0] if mapped_smiles else None
    parsed = Chem.MolFromSmiles(mapped or can_smi(rdmol=tagged)[0])
    if parsed is None:
        return [], mapped

    map_idx = [
        int(atom.GetAtomMapNum())
        for atom in parsed.GetAtoms()
        if atom.GetAtomicNum() > 1
    ]
    return map_idx, mapped


def metabolite_map_indices(
    product: Chem.Mol,
    *,
    mode: Optional[ForestMapIndexing] = None,
) -> list[int]:
    """Per heavy atom in canonical ``smiles`` order: 1-based parent map, 0 if new."""
    map_idx, _ = metabolite_atom_maps(product, mode=mode, mapped_smiles=False)
    return map_idx


def _bond_index(molecule: Molecule, a: int, b: int) -> Optional[int]:
    key = frozenset({a, b})
    for bi, pair in enumerate(molecule.bonds.idx):
        if frozenset(pair) == key:
            return bi
    return None


def site_score(molecule: Molecule, result: ModelResult, site: frozenset[int]) -> float:
    """Look up the predictor score for a forest site (0-based RDKit atom indices)."""
    atoms = sorted(site)

    if isinstance(result, (MolBondResult, BondResult)):
        bonds = result.bond
        if len(atoms) == 2:
            bi = _bond_index(molecule, atoms[0], atoms[1])
            if bi is not None and bi < len(bonds):
                return float(bonds[bi])
        return 0.0

    if isinstance(result, AtomBondResult):
        if len(atoms) == 1:
            ai = atoms[0]
            if 0 <= ai < len(result.atom):
                return float(result.atom[ai])
            return 0.0
        if len(atoms) == 2:
            bi = _bond_index(molecule, atoms[0], atoms[1])
            if bi is not None and bi < len(result.bond):
                return float(result.bond[bi])
            return max(float(result.atom[a]) for a in atoms if 0 <= a < len(result.atom))
        return max(float(result.atom[a]) for a in atoms if 0 <= a < len(result.atom))

    if isinstance(result, MolAtomPairResult):
        for (a, b), score in zip(result.pair_idx, result.pair):
            if frozenset({a, b}) == site:
                return float(score)
        return 0.0

    return 0.0


def enumerate_metabolites(
    rdmol: Chem.Mol,
    ruleset_spec: str,
) -> Iterator[tuple[str, frozenset[int], str, Chem.Mol]]:
    """Yield ``(pathway, rdkit_site, canonical_smiles, product_mol)`` for each metabolite.

    One call to ``RuleSet.metabolites`` enumerates all products for the substrate.
    """
    mode = forest_site_indexing()
    n_atoms = rdmol.GetNumAtoms()
    rs = load_ruleset(ruleset_spec)
    for (rule, site), mols in rs.metabolites(rdmol, unique=True):
        if not mols:
            continue
        product = mols[-1]
        rdkit_site = forest_site_to_rdkit(site, n_atoms, mode)
        yield pathway_name(rule), rdkit_site, can_smi(rdmol=product)[0], product


def attach_metabolites(
    molecule: Molecule,
    *,
    min_score: Optional[float] = None,
    mapped_smiles: bool = False,
    rdmol: Chem.Mol | None = None,
) -> Molecule:
    """Attach forest-inferred metabolite structures to SOM results (in place).

    For each supported model result, forest is called **once** per ruleset to
    enumerate every metabolite the rules allow on the substrate. Predictor site
    scores are looked up for each product, then the list is sorted by score
    (descending). Each metabolite always includes ``map_idx`` (1-based parent
    atom numbers via AtomTracker). When ``mapped_smiles`` is ``True``, also set
    ``mapped_smiles`` with ``:N`` atom-map labels in the SMILES string.
    """
    if rdmol is None:
        rdmol, _ = parse_smiles(molecule.smiles)

    map_mode = forest_map_indexing()
    enumerated: dict[str, list[tuple[str, frozenset[int], str, Chem.Mol]]] = {}

    for result in molecule.results:
        spec = ruleset_for_model(result.model)
        if spec is None or result.metabolite:
            continue
        if spec not in enumerated:
            try:
                enumerated[spec] = list(enumerate_metabolites(rdmol, spec))
            except RuntimeError:
                enumerated[spec] = []
        _attach_from_enumeration(
            molecule,
            result,
            enumerated[spec],
            min_score=min_score,
            mapped_smiles=mapped_smiles,
            map_mode=map_mode,
        )
    return molecule


def _attach_from_enumeration(
    molecule: Molecule,
    result: ModelResult,
    products: list[tuple[str, frozenset[int], str, Chem.Mol]],
    *,
    min_score: Optional[float],
    mapped_smiles: bool,
    map_mode: ForestMapIndexing,
) -> None:
    metabolites: list[Metabolite] = []
    seen: set[tuple[str, str, tuple[int, ...]]] = set()

    for pathway, site, smiles, product in products:
        key = (pathway, smiles, tuple(site_rdkit_indices(site)))
        if key in seen:
            continue
        seen.add(key)
        score = site_score(molecule, result, site)
        if min_score is not None and score < min_score:
            continue
        maps, mapped = metabolite_atom_maps(
            product, mode=map_mode, mapped_smiles=mapped_smiles
        )
        metabolites.append(
            Metabolite(
                smiles=smiles,
                atom=site_rdkit_indices(site),
                map_idx=maps,
                mapped_smiles=mapped,
                pathway=pathway,
                score=score,
            )
        )

    if metabolites:
        metabolites.sort(key=lambda m: (-float(m.score or 0.0), m.pathway or "", m.smiles))
        result.metabolite = metabolites
