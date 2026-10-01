"""Thin adapter to ``xenosite.forest`` for SOM → metabolite structure inference.

Site-of-metabolism *scores* come from :func:`predict`; structure enumeration is
delegated to `Metabolic Forest <https://github.com/swamidasslab/xenosite-forest>`_
via ``resolve("xf:…")`` + ``RuleSet.metabolize(ForestMol(...))``.

Forest 0.10.2+ keeps the input heavy-atom order when building ``ForestMol`` from
an RDKit mol, so ``emit.site_atoms`` are already in that index frame. Product
CSMI spelling may differ from RDKit canonical form; this adapter
re-canonicalizes products with RDKit before exposing them.

``bioactivation`` is the exception: forest 0.10+ has no ``xf:Bioactivation``
catalog entry, so that model still enumerates through the frozen
``xenosite.forest.legacy`` ``BA`` ruleset.
"""

from __future__ import annotations

import warnings
from functools import lru_cache
from typing import Collection, Iterator, Optional

from rdkit import Chem
from xenosite.forest import ForestMol

from .conjugates import (
    HEADS as _CONJUGATE_HEADS,
    bare_smiles,
    head_for_model,
    is_star_conjugate,
    load_conjugate_ruleset,
)
from .molecule import parse_smiles
from .types import (
    AtomBondResult,
    AtomResult,
    BondResult,
    Metabolite,
    MolAtomPairResult,
    MolAtomResult,
    MolBondResult,
    ModelResult,
    Molecule,
)

# ``result.model`` → ``resolve(...)`` CURIE (see xenosite-forest catalog).
# Conjugation heads live in :mod:`xenosite.predict.conjugates`.
# ``bioactivation`` uses the legacy ``BA`` archive (no xf: catalog entry yet).
_LEGACY_BIOACTIVATION = "BA"
_MODEL_RULESETS: dict[str, str] = {
    "epoxidation": "xf:Epoxidation",
    "ndealk": "xf:NDealkylation",
    "quinone": "xf:QuinoneFormation",
    "bioactivation": _LEGACY_BIOACTIVATION,
    "phase1.stable_oxygenation": "xf:StableOxygenation",
    "phase1.unstable_oxygenation": "xf:UnstableOxygenation",
    "phase1.dehydrogenation": "xf:Dehydrogenation",
    "phase1.reduction": "xf:PhaseOne/Reduction",
    "phase1.hydrolysis": "xf:PhaseOne/Hydrolysis",
}

# Legacy bioactivation PBS pathway labels → forest ``BA`` names.
_LEGACY_BIOACTIVATION_PATHWAY: dict[str, str] = {
    "NitrogenReduction": "NitroaromaticReduction",
    "SulfurOxidation": "ThiopheneSulfurOxidation",
}

_REACT_ATOM_IDX = "react_atom_idx"
_INVALID_METABOLITE_WARNING = "Dropping RDKit-invalid metabolite"


def ruleset_for_model(model: str) -> Optional[str]:
    """Return a forest ruleset CURIE (or legacy ``BA``) for a model, if supported."""
    head = head_for_model(model)
    if head is not None:
        return head.ruleset
    if model in _MODEL_RULESETS:
        return _MODEL_RULESETS[model]
    if model.startswith("isozyme."):
        return "xf:NDealkylation"
    return None


def metabolite_supported(model: str) -> bool:
    """True when forest can attach metabolite structures for ``result.model``."""
    return ruleset_for_model(model) is not None


def supported_metabolite_models() -> frozenset[str]:
    """``result.model`` names with explicit forest ruleset mappings."""
    return frozenset(_MODEL_RULESETS) | frozenset(_CONJUGATE_HEADS)


def pathway_name(rule_path: list[str] | str, pattern_name: str = "") -> str:
    """Leaf rule name from a metabolize ``rule_path`` (leaf-first) or legacy rule."""
    if isinstance(rule_path, str):
        return rule_path.split("_", 1)[0]
    if rule_path:
        return rule_path[0]
    return pattern_name


def bioactivation_pathway_name(pathway: str) -> str:
    """Map legacy bioactivation PBS pathway labels onto forest ``BA`` names."""
    return _LEGACY_BIOACTIVATION_PATHWAY.get(pathway, pathway)


def _is_legacy_ruleset(spec: str) -> bool:
    return spec == _LEGACY_BIOACTIVATION or not spec.startswith("xf:")


def site_rdkit_indices(site: frozenset[int] | list[int] | tuple[int, ...]) -> list[int]:
    """Sorted 0-based RDKit indices for a site (input-mol frame)."""
    return sorted(site)


def _rdkit_product_smiles(csmi: str, *, star: bool) -> tuple[Optional[str], Optional[Chem.Mol]]:
    """Parse a forest product CSMI and re-canonicalize with RDKit.

    Conjugation products keep CX ``atomLabel`` blocks; others return bare
    non-isomeric canonical SMILES.
    """
    mol = Chem.MolFromSmiles(csmi)
    if mol is None:
        return None, None
    try:
        if star or "|" in csmi:
            written = Chem.MolToCXSmiles(mol, canonical=True, isomericSmiles=False)
        else:
            written = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
    except Exception:
        return None, None
    return written, mol


def metabolite_atom_maps(
    product: Chem.Mol,
    origins: list[Optional[int]],
    *,
    mapped_smiles: bool = False,
) -> tuple[list[int], Optional[str]]:
    """Return ``(map_idx, mapped_smiles)`` from chematic ``stamp_origins``.

    ``origins`` are 0-based parent indexes (``None`` = new atom), parallel to
    chematic heavy atoms. ``map_idx`` is always computed as 1-based parent atom
    numbers in canonical ``smiles`` order (0 = new atom).
    """
    tagged = Chem.Mol(product)
    heavy = [a for a in tagged.GetAtoms() if a.GetAtomicNum() != 1]
    if len(heavy) != len(origins):
        map_idx = [0] * len(heavy)
        if not mapped_smiles:
            return map_idx, None
        try:
            return map_idx, Chem.MolToSmiles(tagged, canonical=True, isomericSmiles=False)
        except Exception:
            return map_idx, None

    for atom, origin in zip(heavy, origins):
        atom.SetAtomMapNum(0 if origin is None else int(origin) + 1)

    has_star_label = any(
        atom.GetAtomicNum() == 0 and atom.HasProp("atomLabel") for atom in tagged.GetAtoms()
    )
    try:
        if has_star_label:
            written = Chem.MolToCXSmiles(tagged, canonical=True, isomericSmiles=False)
        else:
            written = Chem.MolToSmiles(tagged, canonical=True, isomericSmiles=False)
    except Exception:
        written = None
    mapped = written if mapped_smiles else None
    parsed = Chem.MolFromSmiles(written or "")
    if parsed is None:
        return [], mapped

    map_idx = [
        int(atom.GetAtomMapNum())
        for atom in parsed.GetAtoms()
        if atom.GetAtomicNum() != 1
    ]
    return map_idx, mapped


def metabolite_map_indices(
    product: Chem.Mol,
    origins: list[Optional[int]],
) -> list[int]:
    """Per heavy atom in canonical ``smiles`` order: 1-based parent map, 0 if new."""
    map_idx, _ = metabolite_atom_maps(product, origins, mapped_smiles=False)
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

    if isinstance(result, (AtomResult, MolAtomResult)):
        vals = [float(result.atom[a]) for a in atoms if 0 <= a < len(result.atom)]
        return max(vals) if vals else 0.0

    return 0.0


@lru_cache(maxsize=32)
def _ruleset(spec: str):
    return load_conjugate_ruleset(spec)


def _origins_from_legacy_product(product: Chem.Mol) -> list[Optional[int]]:
    """0-based parent indexes from AtomTracker ``react_atom_idx`` (forest 0.6+)."""
    origins: list[Optional[int]] = []
    for atom in product.GetAtoms():
        if atom.GetAtomicNum() == 1:
            continue
        if not atom.HasProp(_REACT_ATOM_IDX):
            origins.append(None)
            continue
        origins.append(int(atom.GetProp(_REACT_ATOM_IDX)))
    return origins


def _enumerate_legacy(
    rdmol: Chem.Mol,
    ruleset_spec: str,
) -> Iterator[tuple[str, frozenset[int], str, Chem.Mol, list[Optional[int]]]]:
    """Enumerate via frozen ``xenosite.forest.legacy`` (bioactivation ``BA`` only)."""
    from xenosite.forest.legacy import load_ruleset
    from xenosite.forest.legacy.base import can_smi

    n_atoms = rdmol.GetNumAtoms()
    topo_ranks = list(
        Chem.CanonicalRankAtoms(rdmol, includeChirality=False, breakTies=False)
    )
    seen: set[tuple[str, str, tuple[int, ...]]] = set()
    # Legacy metabolize kekulizes the reactant in place — copy first.
    reactant = Chem.Mol(rdmol)
    rs = load_ruleset(ruleset_spec)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=_INVALID_METABOLITE_WARNING,
            category=UserWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message=r"xenosite\.forest\.legacy is a frozen",
            category=DeprecationWarning,
        )
        rows = list(rs.metabolites(reactant, unique=True))

    for (rule, site), mols in rows:
        if not mols:
            continue
        rdkit_site = frozenset(int(i) for i in site)
        for idx in rdkit_site:
            if idx < 0 or idx >= n_atoms:
                raise ValueError(
                    f"legacy forest site {sorted(site)} out of range for "
                    f"{n_atoms} heavy atoms"
                )
        pathway = pathway_name(rule)
        rank_key = tuple(sorted(topo_ranks[i] for i in rdkit_site))
        for product in mols:
            if product is None or product.GetNumAtoms() == 0:
                continue
            smiles_list = can_smi(rdmol=product)
            if not smiles_list:
                continue
            smiles, product_mol = _rdkit_product_smiles(smiles_list[0], star=False)
            if not smiles or product_mol is None:
                continue
            identity = (pathway, bare_smiles(smiles), rank_key)
            if identity in seen:
                continue
            seen.add(identity)
            origins = _origins_from_legacy_product(product)
            yield pathway, rdkit_site, smiles, product, origins


def enumerate_metabolites(
    rdmol: Chem.Mol,
    ruleset_spec: str,
) -> Iterator[tuple[str, frozenset[int], str, Chem.Mol, list[Optional[int]]]]:
    """Yield ``(pathway, site, smiles, product_mol, origins)`` for each metabolite.

    ``site`` is a frozenset of 0-based indexes in the **input** RDKit mol frame.
    Cleavage rules yield one row per fragment. Product SMILES are RDKit-canonical
    (CX preserved for star conjugates).

    Rows are collapsed with the XenoSite UI identity key (pathway + product SMILES +
    sorted topological ranks of the site).

    ``BA`` / other non-``xf:`` specs use the legacy archive path.
    """
    if _is_legacy_ruleset(ruleset_spec):
        yield from _enumerate_legacy(rdmol, ruleset_spec)
        return

    n_atoms = rdmol.GetNumAtoms()
    star = is_star_conjugate(ruleset_spec)
    topo_ranks = list(
        Chem.CanonicalRankAtoms(rdmol, includeChirality=False, breakTies=False)
    )
    seen: set[tuple[str, str, tuple[int, ...]]] = set()
    rs = _ruleset(ruleset_spec)
    reactant = ForestMol(rdmol)

    for emit in rs.metabolize(reactant):
        site_atoms = list(emit.site_atoms)
        if any(i < 0 or i >= n_atoms for i in site_atoms):
            raise ValueError(
                f"forest site {site_atoms} out of range for {n_atoms} heavy atoms"
            )
        rdkit_site = frozenset(site_atoms)
        path = list(emit.rule_path())
        pathway = pathway_name(path, emit.pattern_name)
        rank_key = tuple(sorted(topo_ranks[i] for i in rdkit_site))
        products = emit.products()
        csmis = emit.product_csmis()
        for product_fm, csmi in zip(products, csmis):
            if not csmi:
                continue
            smiles, product_mol = _rdkit_product_smiles(csmi, star=star)
            if not smiles or product_mol is None:
                continue
            identity = (pathway, bare_smiles(smiles), rank_key)
            if identity in seen:
                continue
            seen.add(identity)
            origins = list(product_fm.stamp_origins())
            yield pathway, rdkit_site, smiles, product_mol, origins


def attach_metabolites(
    molecule: Molecule,
    *,
    models: Optional[Collection[str]] = None,
    min_score: Optional[float] = None,
    mapped_smiles: bool = False,
    rdmol: Chem.Mol | None = None,
    rdkit: bool = False,
) -> Molecule:
    """Attach forest-inferred metabolite structures to SOM results (in place).

    For each supported model result, forest is called **once** per ruleset to
    enumerate every metabolite the rules allow on the substrate. Predictor site
    scores are looked up for each product, then the list is sorted by score
    (descending). Each metabolite always includes ``map_idx`` (1-based parent
    atom numbers via chematic stamp origins). When ``mapped_smiles`` is ``True``,
    also set ``mapped_smiles`` with ``:N`` atom-map labels in the SMILES string.

    Conjugation heads (``ugt``, ``reactivity.*``) write **CXSMILES** on
    ``Metabolite.smiles`` (dummy ``*`` + ``atomLabel``). The SMILES token
    before ``|`` is valid on its own and depicts as ``*`` if CX is ignored.

    Parameters
    ----------
    models:
        When set, only results whose ``model`` is in this collection are
        considered. ``None`` (default) considers every forest-supported result.
    rdkit:
        When ``True``, keep the parent RDKit mol on ``molecule.rdkit`` and each
        forest product on ``Metabolite.rdkit`` (no extra parse). Default ``False``.
    """
    if rdmol is None:
        rdmol = molecule.rdkit
    if rdmol is None:
        rdmol, _ = parse_smiles(molecule.smiles)
    if rdkit and molecule.rdkit is None:
        molecule.rdkit = rdmol

    enumerated: dict[
        str, list[tuple[str, frozenset[int], str, Chem.Mol, list[Optional[int]]]]
    ] = {}
    allowed = None if models is None else set(models)

    for result in molecule.results:
        if allowed is not None and result.model not in allowed:
            continue
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
            rdkit=rdkit,
        )
    return molecule


def _attach_from_enumeration(
    molecule: Molecule,
    result: ModelResult,
    products: list[tuple[str, frozenset[int], str, Chem.Mol, list[Optional[int]]]],
    *,
    min_score: Optional[float],
    mapped_smiles: bool,
    rdkit: bool,
) -> None:
    metabolites: list[Metabolite] = []
    seen: set[tuple[str, str, tuple[int, ...]]] = set()

    head = head_for_model(result.model)
    for pathway, site, smiles, product, origins in products:
        if head is not None and head.pathway:
            pathway = head.pathway
        out_smiles = smiles
        if not out_smiles:
            continue
        key = (pathway, bare_smiles(out_smiles), tuple(site_rdkit_indices(site)))
        if key in seen:
            continue
        seen.add(key)
        score = site_score(molecule, result, site)
        if min_score is not None and score < min_score:
            continue
        maps, mapped = metabolite_atom_maps(
            product, origins, mapped_smiles=mapped_smiles
        )
        metabolites.append(
            Metabolite(
                smiles=out_smiles,
                atom=site_rdkit_indices(site),
                map_idx=maps,
                mapped_smiles=mapped,
                pathway=pathway,
                score=score,
                rdkit=product if rdkit else None,
            )
        )

    if metabolites:
        metabolites.sort(key=lambda m: (-float(m.score or 0.0), m.pathway or "", m.smiles))
        result.metabolite = metabolites
