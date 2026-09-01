"""Thin adapter to ``xenosite.forest`` for SOM → metabolite structure inference.

Site-of-metabolism *scores* come from :func:`predict`; structure enumeration is
delegated to `Metabolic Forest <https://github.com/swamidasslab/xenosite-forest>`_.

``xenosite.forest`` is still v0; site frozensets may be 0-based RDKit indices or
1-based atom numbers. :func:`forest_site_indexing` probes ethane (``CC``) with the
``SO`` (stable oxygenation) ruleset once per installed forest version and converts
sites to 0-based RDKit before joining predictor scores.
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
_ENV_INDEXING = "XENOSITE_FOREST_SITE_INDEXING"


class ForestSiteIndexing(str, Enum):
    """How ``xenosite.forest`` labels sites in ``RuleSet.metabolites`` frozensets."""

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


def _indexing_from_env() -> Optional[ForestSiteIndexing]:
    raw = os.environ.get(_ENV_INDEXING, "").strip().lower()
    if raw in ("rdkit_zero", "rdkit", "0", "zero"):
        return ForestSiteIndexing.RDKIT_ZERO
    if raw in ("atom_number_one", "one", "1", "phase1"):
        return ForestSiteIndexing.ATOM_NUMBER_ONE
    return None


def _probe_forest_site_indexing() -> ForestSiteIndexing:
    """Detect forest site numbering using ``CC`` + stable oxygenation (``SO``).

    - RDKit 0-based: hydroxylation sites include atom index ``0``.
    - 1-based atom numbers: sites use ``1..N`` only; on ethane (``N=2``) index ``2``
      appears and index ``0`` does not.
    """
    forced = _indexing_from_env()
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


@lru_cache(maxsize=8)
def forest_site_indexing_for_version(version: str) -> ForestSiteIndexing:
    return _probe_forest_site_indexing()


def forest_site_indexing() -> ForestSiteIndexing:
    """Cached site-index convention for the installed ``xenosite.forest`` version."""
    import xenosite.forest as xf

    version = getattr(xf, "__version__", "unknown")
    return forest_site_indexing_for_version(version)


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
) -> Iterator[tuple[str, frozenset[int], str]]:
    """Yield ``(pathway, rdkit_site, canonical_smiles)`` for every forest metabolite.

    One call to ``RuleSet.metabolites`` enumerates all products for the substrate;
    site scores are joined afterward (see :func:`attach_metabolites`).
    """
    mode = forest_site_indexing()
    n_atoms = rdmol.GetNumAtoms()
    rs = load_ruleset(ruleset_spec)
    for (rule, site), mols in rs.metabolites(rdmol, unique=True):
        if not mols:
            continue
        rdkit_site = forest_site_to_rdkit(site, n_atoms, mode)
        yield pathway_name(rule), rdkit_site, can_smi(rdmol=mols[-1])[0]


def attach_metabolites(
    molecule: Molecule,
    *,
    min_score: Optional[float] = None,
    rdmol: Chem.Mol | None = None,
) -> Molecule:
    """Attach forest-inferred metabolite structures to SOM results (in place).

    For each supported model result, forest is called **once** per ruleset to
    enumerate every metabolite the rules allow on the substrate. Predictor site
    scores are looked up for each product, then the list is sorted by score
    (descending). Existing ``metabolite`` lists are left unchanged (e.g.
    bioactivation from a legacy backend).
    """
    if rdmol is None:
        rdmol, _ = parse_smiles(molecule.smiles)

    enumerated: dict[str, list[tuple[str, frozenset[int], str]]] = {}

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
            molecule, result, enumerated[spec], min_score=min_score
        )
    return molecule


def _attach_from_enumeration(
    molecule: Molecule,
    result: ModelResult,
    products: list[tuple[str, frozenset[int], str]],
    *,
    min_score: Optional[float],
) -> None:
    metabolites: list[Metabolite] = []
    seen: set[tuple[str, str, tuple[int, ...]]] = set()

    for pathway, site, smiles in products:
        key = (pathway, smiles, tuple(site_rdkit_indices(site)))
        if key in seen:
            continue
        seen.add(key)
        score = site_score(molecule, result, site)
        if min_score is not None and score < min_score:
            continue
        metabolites.append(
            Metabolite(
                smiles=smiles,
                atom=site_rdkit_indices(site),
                pathway=pathway,
                score=score,
            )
        )

    if metabolites:
        metabolites.sort(key=lambda m: (-float(m.score or 0.0), m.pathway or "", m.smiles))
        result.metabolite = metabolites
