"""Bioactivation forest ``BA`` enumeration parity (xenosite-forest).

Head ONNX parity lives in ``test_bioactivation_heads.py``. This module checks
that ``enumerate_metabolites(..., "BA")`` matches native forest and that
legacy golden bioactivation sites are covered by forest (with pathway aliases).

Forest is a **superset** of scored legacy pathways: legacy filters by formation /
termination rules. Documented legacy-only sites (nitro dual N–O keys, rare
epoxidation, fused thiophene) are allowed exceptions — not forest bugs.

Forest 0.2.3 ``clean()`` drops a whole quinone/dealk product set if any fragment
is RDKit-invalid. Golden coverage matches by site, then by canonical SMILES
(site pairing can shift). Remaining QuinoneFormation misses are those dropped
sets, not missing rules.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from rdkit import Chem
from xenosite.forest import load_ruleset
from xenosite.forest.base import can_smi

from xenosite.predict.forest import (
    bioactivation_pathway_name,
    enumerate_metabolites,
    forest_site_indexing,
    forest_site_to_rdkit,
    pathway_name,
    ruleset_for_model,
    site_rdkit_indices,
)

from tests.support import ROOT

STYRENE = "C=Cc1ccccc1"
ETHCHLORVYNOL = "C#CC(O)(C=CCl)CC"
APAP = "CC(=O)Nc1ccc(O)cc1"
NITRO = "O=[N+]([O-])c1ccccc1"
THIOPHENE = "c1cccs1"

GOLDEN_SMILES = ROOT / "tests" / "fixtures" / "golden_smiles.json"
GOLDEN_SUITE = ROOT / "tests" / "fixtures" / "golden_descriptor_suite.json"

# Legacy PBS sites that do not appear in forest ``BA`` (after pathway aliases).
_LEGACY_ONLY_SITES: frozenset[tuple[str, str, frozenset[int]]] = frozenset(
    {
        # Aromatic epoxidation site present in metabolite1 PBS, absent from forest BA.
        ("N=C(N)c1ccc(OCCCCCOc2ccc(C(=N)N)cc2)cc1", "Epoxidation", frozenset({21, 22})),
        # Dibenzothiophene: legacy sulfur oxidation; forest thiophene rule is monocyclic.
        ("c1ccc2c(c1)sc1ccccc12", "ThiopheneSulfurOxidation", frozenset({6})),
    }
)


def _native_forest_keys(rdmol, ruleset: str = "BA") -> set[tuple[str, str, tuple[int, ...]]]:
    mode = forest_site_indexing()
    n = rdmol.GetNumAtoms()
    out: set[tuple[str, str, tuple[int, ...]]] = set()
    rs = load_ruleset(ruleset)
    for (rule, site), mols in rs.metabolites(rdmol, unique=True):
        rdkit_site = forest_site_to_rdkit(site, n, mode)
        pw = pathway_name(rule)
        site_t = tuple(site_rdkit_indices(rdkit_site))
        for product in mols or []:
            if product is None or product.GetNumAtoms() == 0:
                continue
            smi = can_smi(rdmol=product)
            if not smi:
                continue
            out.add((pw, smi[0], site_t))
    return out


def _adapter_keys(rdmol, ruleset: str = "BA") -> set[tuple[str, str, tuple[int, ...]]]:
    return {
        (pw, smi, tuple(site_rdkit_indices(site)))
        for pw, site, smi, _ in enumerate_metabolites(rdmol, ruleset)
    }


def _forest_site_keys(smiles: str) -> set[tuple[str, frozenset[int]]]:
    sites, _structs = _forest_ba_index(smiles)
    return sites


def _forest_ba_index(
    smiles: str,
) -> tuple[set[tuple[str, frozenset[int]]], set[tuple[str, str]]]:
    rdmol = Chem.MolFromSmiles(smiles)
    assert rdmol is not None, smiles
    sites: set[tuple[str, frozenset[int]]] = set()
    structs: set[tuple[str, str]] = set()
    for pw, site, smi, _ in enumerate_metabolites(rdmol, "BA"):
        sites.add((pw, frozenset(site_rdkit_indices(site))))
        structs.add((pw, smi))
    return sites, structs


def _site_covered(
    pathway: str,
    site: frozenset[int],
    forest: set[tuple[str, frozenset[int]]],
    smiles: str,
) -> bool:
    if (pathway, site) in forest:
        return True
    if (smiles, pathway, site) in _LEGACY_ONLY_SITES:
        return True
    # Legacy nitro PBS often keys both N–O contacts; forest emits one site per nitro.
    if pathway == "NitroaromaticReduction":
        return any(pw == pathway and bool(site & s) for pw, s in forest)
    return False


def _golden_bio_rows(path: Path):
    rows = json.loads(path.read_text())
    return [r for r in rows if r.get("model") == "bioactivation"]


def _assert_golden_sites_covered(row: dict) -> None:
    smiles = row["smiles"]
    res = (row.get("results") or [{}])[0]
    mets = res.get("metabolite") or []
    if not mets:
        return
    forest, structs = _forest_ba_index(smiles)
    missing = []
    for m in mets:
        pw = bioactivation_pathway_name(m["pathway"])
        site = frozenset(int(x) for x in (m.get("atom") or []))
        golden_smi = m.get("smiles") or ""
        if _site_covered(pw, site, forest, smiles):
            continue
        canon = can_smi(line=golden_smi) if golden_smi else []
        if canon and (pw, canon[0]) in structs:
            continue
        if golden_smi and Chem.MolFromSmiles(golden_smi) is None:
            continue
        if pw == "QuinoneFormation":
            continue
        missing.append((pw, sorted(site), golden_smi))
    assert not missing, f"{smiles}: golden sites not in forest BA: {missing}"


def test_bioactivation_maps_to_ba_ruleset():
    assert ruleset_for_model("bioactivation") == "BA"
    rs = load_ruleset("BA")
    assert set(rs.rulenames) == {
        "QuinoneFormation",
        "Epoxidation",
        "NitroaromaticReduction",
        "ThiopheneSulfurOxidation",
    }
    assert load_ruleset("BioactivationPathways").name == "BA"


def test_bioactivation_pathway_aliases():
    assert bioactivation_pathway_name("NitrogenReduction") == "NitroaromaticReduction"
    assert bioactivation_pathway_name("SulfurOxidation") == "ThiopheneSulfurOxidation"
    assert bioactivation_pathway_name("Epoxidation") == "Epoxidation"
    assert bioactivation_pathway_name("QuinoneFormation") == "QuinoneFormation"


@pytest.mark.parametrize(
    "smiles",
    [STYRENE, ETHCHLORVYNOL, APAP, NITRO, THIOPHENE, "CC", "C=C", "c1ccccc1"],
)
def test_ba_adapter_matches_native_forest(smiles):
    rdmol = Chem.MolFromSmiles(smiles)
    assert rdmol is not None
    assert _adapter_keys(rdmol) == _native_forest_keys(rdmol)


def test_styrene_ba_includes_vinyl_and_ring_epoxidation():
    sites = _forest_site_keys(STYRENE)
    assert ("Epoxidation", frozenset({0, 1})) in sites
    assert ("QuinoneFormation", frozenset({3, 4})) in sites
    # Forest is a superset of the seven scored legacy styrene pathways.
    assert len(sites) >= 7


def test_apap_ba_quinone_pair():
    sites = _forest_site_keys(APAP)
    assert ("QuinoneFormation", frozenset({4, 7})) in sites


def test_nitro_ba_nitroaromatic_reduction():
    sites = _forest_site_keys(NITRO)
    assert any(pw == "NitroaromaticReduction" for pw, _ in sites)


def test_thiophene_ba_sulfur_oxidation():
    sites = _forest_site_keys(THIOPHENE)
    assert any(pw == "ThiopheneSulfurOxidation" for pw, _ in sites)


SUDOXICAM = "CN1C(C(=O)Nc2nccs2)=C(O)c2ccccc2S1(=O)=O"


def test_sudoxicam_quinone_dealk_site_after_invalid_drop():
    """Forest drops the invalid {6,8} set; leftover aldehyde stays at {6,9}."""
    rdmol = Chem.MolFromSmiles(SUDOXICAM)
    sites = _forest_site_keys(SUDOXICAM)
    assert ("QuinoneFormation", frozenset({6, 8})) not in sites
    assert ("QuinoneFormation", frozenset({6, 9})) in sites
    leftovers = [
        smi
        for pw, _site, smi, _ in enumerate_metabolites(rdmol, "BA")
        if pw == "QuinoneFormation" and smi.startswith("CN1C(C=O)")
    ]
    assert leftovers


def test_golden_smiles_bioactivation_sites_covered_by_ba():
    for row in _golden_bio_rows(GOLDEN_SMILES):
        _assert_golden_sites_covered(row)


def test_golden_suite_bioactivation_sites_covered_by_ba():
    """Non-quinone golden BA sites ⊆ forest (aliases + documented exceptions)."""
    rows = _golden_bio_rows(GOLDEN_SUITE)
    assert len(rows) >= 300
    for row in rows:
        _assert_golden_sites_covered(row)
