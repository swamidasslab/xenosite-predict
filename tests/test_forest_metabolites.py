"""Tests for xenosite.forest metabolite adapter — index alignment and full enumeration."""

from __future__ import annotations

import pytest
from rdkit import Chem

from xenosite.predict import predict
from xenosite.predict.backends.onnx import OnnxBackend
from xenosite.predict.features import _ob
from xenosite.predict.forest import (
    ForestSiteIndexing,
    attach_metabolites,
    enumerate_metabolites,
    forest_site_indexing,
    forest_site_indexing_for_version,
    forest_site_to_rdkit,
    site_rdkit_indices,
    site_score,
)
from xenosite.predict.molecule import parse_smiles
from xenosite.predict.types import Atoms, Bonds, MolBondResult, Molecule

from tests.support import ROOT, onnx_weights_present

GAP_SMILES = "COc1cc2nc(SCc3ccccc3C)[nH]c2cc1OC"
ETHYLENE = "C=C"
PROPANE = "CCC"


def _assert_rdkit_indices(molecule: Molecule) -> None:
    n = molecule.atoms.num
    for result in molecule.results:
        for met in result.metabolite or []:
            assert met.atom is not None
            for ai in met.atom:
                assert 0 <= ai < n, f"atom index {ai} out of range for n={n}"


def _assert_sorted_by_score_desc(metabolites) -> None:
    scores = [float(m.score or 0.0) for m in metabolites]
    assert scores == sorted(scores, reverse=True)


def test_ethylene_epoxidation_rdkit_indices():
    mol = Molecule(
        smiles=ETHYLENE,
        atoms=Atoms(num=2),
        bonds=Bonds(idx=[(0, 1)]),
        results=[MolBondResult(model="epoxidation", version="0", mol=0.9, bond=[0.85])],
    )
    attach_metabolites(mol)
    assert len(mol.results[0].metabolite) == 1
    met = mol.results[0].metabolite[0]
    assert met.atom == [0, 1]
    assert met.smiles == "C1CO1"
    assert met.pathway == "Epoxidation"
    assert met.score == pytest.approx(0.85)


def test_propane_includes_all_forest_metabolites_sorted():
    rdmol, mol = parse_smiles(PROPANE)
    mol.results = [
        __import__("xenosite.predict.types", fromlist=["AtomBondResult"]).AtomBondResult(
            model="phase1.stable_oxygenation",
            version="0",
            atom=[0.0, 0.72, 0.0],
            bond=[0.0, 0.0],
        )
    ]
    attach_metabolites(mol, rdmol=rdmol)
    mets = mol.results[0].metabolite
    assert mets
    forest_count = sum(1 for _ in enumerate_metabolites(rdmol, "SO"))
    assert len(mets) == forest_count
    _assert_sorted_by_score_desc(mets)
    assert mets[0].score >= mets[-1].score
    top = [m for m in mets if abs(float(m.score or 0.0) - 0.72) < 1e-9]
    assert top
    assert all(m.atom == [1] for m in top)


def test_site_score_bond_and_atom():
    _, mol = parse_smiles(PROPANE)
    result = __import__("xenosite.predict.types", fromlist=["AtomBondResult"]).AtomBondResult(
        model="phase1.stable_oxygenation",
        version="0",
        atom=[0.0, 0.1, 0.72],
        bond=[0.55, 0.0],
    )
    assert site_score(mol, result, frozenset({2})) == pytest.approx(0.72)
    assert site_score(mol, result, frozenset({0, 1})) == pytest.approx(0.55)
    assert site_score(mol, result, frozenset({1, 2})) == pytest.approx(0.0)


def test_zero_score_sites_still_included():
    mol = Molecule(
        smiles=ETHYLENE,
        atoms=Atoms(num=2),
        bonds=Bonds(idx=[(0, 1)]),
        results=[MolBondResult(model="epoxidation", version="0", mol=0.9, bond=[0.0])],
    )
    attach_metabolites(mol)
    assert mol.results[0].metabolite
    assert mol.results[0].metabolite[0].score == 0.0


def test_min_score_filters_after_enumeration():
    mol = Molecule(
        smiles=ETHYLENE,
        atoms=Atoms(num=2),
        bonds=Bonds(idx=[(0, 1)]),
        results=[MolBondResult(model="epoxidation", version="0", mol=0.9, bond=[0.85])],
    )
    attach_metabolites(mol, min_score=0.9)
    assert mol.results[0].metabolite is None


@pytest.mark.skipif(not _ob.installed(), reason="OpenBabel not installed")
@pytest.mark.skipif(not onnx_weights_present("epoxidation"), reason="no epoxidation ONNX")
def test_predict_ethylene_metabolites_end_to_end():
    be = OnnxBackend(ROOT / "weights" / "onnx")
    mol = predict(ETHYLENE, model="epoxidation", backend=be, metabolites=True)
    result = next(r for r in mol.results if r.model == "epoxidation")
    assert result.metabolite
    _assert_rdkit_indices(mol)
    _assert_sorted_by_score_desc(result.metabolite)
    epox = next(m for m in result.metabolite if m.pathway == "Epoxidation")
    assert epox.atom == [0, 1]
    assert epox.score == pytest.approx(result.bond[0])


@pytest.mark.skipif(not _ob.installed(), reason="OpenBabel not installed")
@pytest.mark.skipif(not onnx_weights_present("reactivity"), reason="no reactivity ONNX")
def test_gap_smiles_nh_reactivity_indices():
    """Legacy [nH] gap molecule: metabolite atoms must stay in RDKit index range."""
    be = OnnxBackend(ROOT / "weights" / "onnx")
    mol = predict(GAP_SMILES, model="reactivity", backend=be, metabolites=True)
    _assert_rdkit_indices(mol)
    rdmol, _ = parse_smiles(mol.smiles)
    assert rdmol.GetNumAtoms() == mol.atoms.num


@pytest.mark.skipif(not _ob.installed(), reason="OpenBabel not installed")
@pytest.mark.skipif(not onnx_weights_present("phase1"), reason="no phase1 ONNX")
def test_phase1_metabolites_match_site_scores():
    be = OnnxBackend(ROOT / "weights" / "onnx")
    mol = predict(PROPANE, models=["phase1"], backend=be, metabolites=True)
    for result in mol.results:
        if not result.model.startswith("phase1."):
            continue
        _assert_rdkit_indices(mol)
        if not result.metabolite:
            continue
        _assert_sorted_by_score_desc(result.metabolite)
        for met in result.metabolite:
            site = frozenset(met.atom or [])
            assert site_score(mol, result, site) == pytest.approx(float(met.score or 0.0))


def test_forest_site_matches_rdkit_mol():
    """Forest enumeration uses the same canonical RDKit mol as predict."""
    rdmol, mol = parse_smiles(ETHYLENE)
    sites = {site for _, site, _ in enumerate_metabolites(rdmol, "SO.Epoxidation")}
    assert frozenset({0, 1}) in sites
    for idx in site_rdkit_indices(frozenset({0, 1})):
        assert rdmol.GetAtomWithIdx(idx).GetAtomicNum() > 1 or idx in (0, 1)


def test_cc_stable_oxygenation_probe_is_rdkit_zero():
    """Ethane + SO: hydroxylation site index 0 ⇒ forest uses RDKit 0-based (v0.1.0)."""
    import xenosite.forest as xf

    forest_site_indexing_for_version.cache_clear()
    mode = forest_site_indexing()
    assert mode == ForestSiteIndexing.RDKIT_ZERO

    rdmol = Chem.MolFromSmiles("CC")
    hydroxy = [
        site
        for _p, site, _s in enumerate_metabolites(rdmol, "SO")
        if _p == "Hydroxylation"
    ]
    assert hydroxy
    assert any(0 in s for s in hydroxy)
    assert all(max(s) < rdmol.GetNumAtoms() for s in hydroxy)


def test_forest_site_to_rdkit_one_based_shift():
    assert forest_site_to_rdkit(
        frozenset({1, 2}), 2, ForestSiteIndexing.ATOM_NUMBER_ONE
    ) == frozenset({0, 1})
    assert forest_site_to_rdkit(
        frozenset({0, 1}), 2, ForestSiteIndexing.RDKIT_ZERO
    ) == frozenset({0, 1})


def test_one_based_indexing_via_env(monkeypatch):
    monkeypatch.setenv("XENOSITE_FOREST_SITE_INDEXING", "atom_number_one")
    forest_site_indexing_for_version.cache_clear()
    assert forest_site_indexing() == ForestSiteIndexing.ATOM_NUMBER_ONE
    forest_site_indexing_for_version.cache_clear()
    monkeypatch.delenv("XENOSITE_FOREST_SITE_INDEXING", raising=False)
