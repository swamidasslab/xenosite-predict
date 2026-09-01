"""Tests for xenosite.forest metabolite adapter — index alignment and full enumeration."""

from __future__ import annotations

import pytest
from rdkit import Chem

from xenosite.predict import predict
from xenosite.predict.backends.onnx import OnnxBackend
from xenosite.predict.features import _ob
from xenosite.predict.forest import (
    ForestMapIndexing,
    ForestSiteIndexing,
    attach_metabolites,
    enumerate_metabolites,
    forest_map_indexing,
    forest_map_indexing_for_version,
    forest_site_indexing,
    forest_site_indexing_for_version,
    forest_site_to_rdkit,
    metabolite_atom_maps,
    metabolite_map_indices,
    site_rdkit_indices,
    site_score,
)
from xenosite.predict.molecule import parse_smiles
from xenosite.predict.types import (
    AtomBondResult,
    Atoms,
    BondResult,
    Bonds,
    MolAtomPairResult,
    MolBondResult,
    Molecule,
)

from tests.support import ROOT, onnx_weights_present, onnx_root

GAP_SMILES = "COc1cc2nc(SCc3ccccc3C)[nH]c2cc1OC"
ETHYLENE = "C=C"
PROPANE = "CCC"
NDEALK = "CN(C)Cc1ccccc1"
PHENOL = "Oc1ccccc1"
BENZENE = "c1ccccc1"


def _forest_metabolite_keys(rdmol, ruleset: str) -> set[tuple[str, str, tuple[int, ...]]]:
    return {
        (pathway, smiles, tuple(site_rdkit_indices(site)))
        for pathway, site, smiles, _ in enumerate_metabolites(rdmol, ruleset)
    }


def _attached_metabolite_keys(metabolites) -> set[tuple[str, str, tuple[int, ...]]]:
    return {
        (m.pathway or "", m.smiles, tuple(m.atom or []))
        for m in metabolites
    }


def _unique_forest_count(rdmol, ruleset: str) -> int:
    keys = {
        (pathway, smiles, tuple(site_rdkit_indices(site)))
        for pathway, site, smiles, _ in enumerate_metabolites(rdmol, ruleset)
    }
    return len(keys)


def _bond_scores(mol: Molecule, site: tuple[int, int], score: float) -> list[float]:
    key = frozenset(site)
    out = [0.0] * len(mol.bonds.idx)
    for bi, pair in enumerate(mol.bonds.idx):
        if frozenset(pair) == key:
            out[bi] = score
    return out


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


def _assert_ethylene_epoxidation_map_idx(maps: list[int]) -> None:
    """Invariants checked by ``test_ethylene_epoxidation_map_idx``."""
    assert maps
    assert set(maps) <= {0, 1, 2}
    assert 1 in maps and 2 in maps


def _opposite_site_indexing(mode: ForestSiteIndexing) -> ForestSiteIndexing:
    return (
        ForestSiteIndexing.ATOM_NUMBER_ONE
        if mode == ForestSiteIndexing.RDKIT_ZERO
        else ForestSiteIndexing.RDKIT_ZERO
    )


def _opposite_map_indexing(mode: ForestMapIndexing) -> ForestMapIndexing:
    return (
        ForestMapIndexing.ATOM_NUMBER_ONE
        if mode == ForestMapIndexing.RDKIT_ZERO
        else ForestMapIndexing.RDKIT_ZERO
    )


def _clear_forest_indexing_caches() -> None:
    forest_site_indexing_for_version.cache_clear()
    forest_map_indexing_for_version.cache_clear()


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
    assert met.map_idx
    assert met.mapped_smiles is None
    assert met.pathway == "Epoxidation"
    assert met.score == pytest.approx(0.85)


def test_propane_includes_all_forest_metabolites_sorted():
    rdmol, mol = parse_smiles(PROPANE)
    mol.results = [
        AtomBondResult(
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
    result = AtomBondResult(
        model="phase1.stable_oxygenation",
        version="0",
        atom=[0.0, 0.1, 0.72],
        bond=[0.55, 0.0],
    )
    assert site_score(mol, result, frozenset({2})) == pytest.approx(0.72)
    assert site_score(mol, result, frozenset({0, 1})) == pytest.approx(0.55)
    assert site_score(mol, result, frozenset({1, 2})) == pytest.approx(0.0)


def test_site_score_bond_result():
    """``BondResult`` (ndealk / isozyme): 2-atom sites use ``molecule.bonds`` lookup."""
    _, mol = parse_smiles(NDEALK)
    result = BondResult(
        model="ndealk",
        version="0",
        bond=_bond_scores(mol, (0, 1), 0.88),
    )
    assert site_score(mol, result, frozenset({0, 1})) == pytest.approx(0.88)
    assert site_score(mol, result, frozenset({1, 3})) == pytest.approx(0.0)
    assert site_score(mol, result, frozenset({1})) == pytest.approx(0.0)


def test_site_score_mol_atom_pair_result():
    """``MolAtomPairResult`` (quinone): scores keyed by ``pair_idx``."""
    _, mol = parse_smiles(PHENOL)
    result = MolAtomPairResult(
        model="quinone",
        version="0",
        mol=0.5,
        atom=[0.0] * mol.atoms.num,
        pair=[0.77, 0.33],
        pair_idx=[(1, 2), (1, 4)],
    )
    assert site_score(mol, result, frozenset({1, 2})) == pytest.approx(0.77)
    assert site_score(mol, result, frozenset({2, 1})) == pytest.approx(0.77)
    assert site_score(mol, result, frozenset({1, 4})) == pytest.approx(0.33)
    assert site_score(mol, result, frozenset({0, 1})) == pytest.approx(0.0)


def test_attach_metabolites_bond_result():
    rdmol, mol = parse_smiles(NDEALK)
    mol.results = [
        BondResult(
            model="ndealk",
            version="0",
            bond=_bond_scores(mol, (0, 1), 0.88),
        )
    ]
    attach_metabolites(mol, rdmol=rdmol)
    mets = mol.results[0].metabolite
    assert mets
    assert len(mets) == _unique_forest_count(rdmol, "UO.Dealkylation")
    _assert_sorted_by_score_desc(mets)
    scored = [m for m in mets if m.atom == [0, 1]]
    assert scored
    assert scored[0].score == pytest.approx(0.88)
    assert scored[0].map_idx


def test_attach_metabolites_mol_atom_pair_result():
    rdmol, mol = parse_smiles(PHENOL)
    result = MolAtomPairResult(
        model="quinone",
        version="0",
        mol=0.5,
        atom=[0.0] * mol.atoms.num,
        pair=[0.77, 0.33],
        pair_idx=[(1, 2), (1, 4)],
    )
    mol.results = [result]
    attach_metabolites(mol, rdmol=rdmol)
    mets = mol.results[0].metabolite
    assert len(mets) == _unique_forest_count(rdmol, "QF.QuinoneFormation")
    _assert_rdkit_indices(mol)
    for met in mets:
        assert met.map_idx
        assert site_score(mol, result, frozenset(met.atom or [])) == pytest.approx(
            float(met.score or 0.0)
        )


def test_topologically_equivalent_soms_both_emitted():
    """RDKit-symmetric sites with the same product SMILES stay separate rows."""
    from tests.v0_legacy.rdkit_equiv import bond_symmetry_groups

    rdmol, mol = parse_smiles(BENZENE)
    groups = bond_symmetry_groups(rdmol)
    sym_class = next(members for members in groups.values() if 0 in members and 5 in members)

    mol.results = [
        MolBondResult(
            model="epoxidation",
            version="0",
            mol=0.5,
            bond=[0.0] * len(mol.bonds.idx),
        )
    ]
    attach_metabolites(mol, rdmol=rdmol)

    epox = [
        m
        for m in mol.results[0].metabolite or []
        if m.pathway == "Epoxidation" and m.smiles == "C1=CC2OC2C=C1"
    ]
    assert len(epox) == 2
    sites = {tuple(m.atom or []) for m in epox}
    assert sites == {(0, 1), (0, 5)}
    assert all(m.map_idx for m in epox)

    forest_keys = _forest_metabolite_keys(rdmol, "SO.Epoxidation")
    attached_keys = _attached_metabolite_keys(mol.results[0].metabolite)
    assert attached_keys == forest_keys
    assert sym_class  # bonds 0 and 5 are topologically equivalent on benzene


def test_exact_duplicate_forest_hits_deduped():
    """Identical pathway + site + product from multiple forest rules → one row."""
    rdmol, mol = parse_smiles(PROPANE)
    raw_cc = [
        (pathway, tuple(site_rdkit_indices(site)), smiles)
        for pathway, site, smiles, _ in enumerate_metabolites(rdmol, "UO")
        if smiles == "CC"
    ]
    assert len(raw_cc) == 2
    assert raw_cc[0][1] == raw_cc[1][1] == (0, 1)

    mol.results = [
        AtomBondResult(
            model="phase1.unstable_oxygenation",
            version="0",
            atom=[0.0, 0.0, 0.0],
            bond=[0.0, 0.0],
        )
    ]
    attach_metabolites(mol, rdmol=rdmol)
    cc = [m for m in mol.results[0].metabolite or [] if m.smiles == "CC"]
    assert len(cc) == 1
    assert cc[0].atom == [0, 1]
    assert len(mol.results[0].metabolite) == _unique_forest_count(rdmol, "UO")


def test_attach_metabolites_multiple_model_results():
    """Each ``ModelResult`` gets metabolites from its own forest ruleset."""
    rdmol, mol = parse_smiles(PROPANE)
    mol.results = [
        AtomBondResult(
            model="phase1.stable_oxygenation",
            version="0",
            atom=[0.0, 0.72, 0.0],
            bond=[0.0, 0.0],
        ),
        AtomBondResult(
            model="phase1.unstable_oxygenation",
            version="0",
            atom=[0.0, 0.0, 0.61],
            bond=[0.0, 0.0],
        ),
    ]
    attach_metabolites(mol, rdmol=rdmol)
    stable, unstable = mol.results
    assert len(stable.metabolite) == _unique_forest_count(rdmol, "SO")
    assert len(unstable.metabolite) == _unique_forest_count(rdmol, "UO")
    assert all(m.atom == [1] for m in stable.metabolite if abs(float(m.score or 0) - 0.72) < 1e-9)
    assert all(m.atom == [2] for m in unstable.metabolite if abs(float(m.score or 0) - 0.61) < 1e-9)


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
    be = OnnxBackend(onnx_root())
    mol = predict(ETHYLENE, model="epoxidation", backend=be, metabolites=True)
    result = next(r for r in mol.results if r.model == "epoxidation")
    assert result.metabolite
    _assert_rdkit_indices(mol)
    _assert_sorted_by_score_desc(result.metabolite)
    epox = next(m for m in result.metabolite if m.pathway == "Epoxidation")
    assert epox.atom == [0, 1]
    assert epox.score == pytest.approx(result.bond[0])


@pytest.mark.skipif(not _ob.installed(), reason="OpenBabel not installed")
@pytest.mark.skipif(not onnx_weights_present("ndealk"), reason="no ndealk ONNX")
def test_predict_ndealk_metabolites_end_to_end():
    be = OnnxBackend(onnx_root())
    mol = predict(NDEALK, model="ndealk", backend=be, metabolites=True)
    result = next(r for r in mol.results if r.model == "ndealk")
    assert result.metabolite
    _assert_rdkit_indices(mol)
    _assert_sorted_by_score_desc(result.metabolite)
    for met in result.metabolite:
        assert met.map_idx
        assert site_score(mol, result, frozenset(met.atom or [])) == pytest.approx(
            float(met.score or 0.0)
        )


@pytest.mark.skipif(not _ob.installed(), reason="OpenBabel not installed")
@pytest.mark.skipif(not onnx_weights_present("quinone"), reason="no quinone ONNX")
def test_predict_quinone_metabolites_end_to_end():
    be = OnnxBackend(onnx_root())
    mol = predict(PHENOL, model="quinone", backend=be, metabolites=True)
    result = next(r for r in mol.results if r.model == "quinone")
    assert result.metabolite
    _assert_rdkit_indices(mol)
    _assert_sorted_by_score_desc(result.metabolite)
    for met in result.metabolite:
        assert met.map_idx
        assert site_score(mol, result, frozenset(met.atom or [])) == pytest.approx(
            float(met.score or 0.0)
        )


@pytest.mark.skipif(not _ob.installed(), reason="OpenBabel not installed")
@pytest.mark.skipif(not onnx_weights_present("reactivity"), reason="no reactivity ONNX")
def test_gap_smiles_nh_reactivity_indices():
    """Legacy [nH] gap molecule: metabolite atoms must stay in RDKit index range."""
    be = OnnxBackend(onnx_root())
    mol = predict(GAP_SMILES, model="reactivity", backend=be, metabolites=True)
    _assert_rdkit_indices(mol)
    rdmol, _ = parse_smiles(mol.smiles)
    assert rdmol.GetNumAtoms() == mol.atoms.num


@pytest.mark.skipif(not _ob.installed(), reason="OpenBabel not installed")
@pytest.mark.skipif(not onnx_weights_present("phase1"), reason="no phase1 ONNX")
def test_phase1_metabolites_match_site_scores():
    be = OnnxBackend(onnx_root())
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
    sites = {site for _, site, _, _ in enumerate_metabolites(rdmol, "SO.Epoxidation")}
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
        for _p, site, _s, _m in enumerate_metabolites(rdmol, "SO")
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


def test_cc_map_probe_is_rdkit_zero():
    _clear_forest_indexing_caches()
    assert forest_map_indexing() == ForestMapIndexing.RDKIT_ZERO


def test_wrong_site_indexing_detection_fails(monkeypatch):
    """Forcing the opposite site convention must break ethylene attach invariants."""
    _clear_forest_indexing_caches()
    detected = forest_site_indexing()
    wrong = _opposite_site_indexing(detected)
    monkeypatch.setenv("XENOSITE_FOREST_SITE_INDEXING", wrong.value)
    _clear_forest_indexing_caches()

    mol = Molecule(
        smiles=ETHYLENE,
        atoms=Atoms(num=2),
        bonds=Bonds(idx=[(0, 1)]),
        results=[MolBondResult(model="epoxidation", version="0", mol=0.9, bond=[0.85])],
    )
    with pytest.raises(ValueError, match="invalid RDKit index"):
        attach_metabolites(mol)

    _clear_forest_indexing_caches()
    monkeypatch.delenv("XENOSITE_FOREST_SITE_INDEXING", raising=False)


def test_wrong_map_indexing_detection_fails(monkeypatch):
    """Forcing the opposite map convention must fail ethylene ``map_idx`` checks."""
    from xenosite.forest import load_ruleset

    _clear_forest_indexing_caches()
    detected = forest_map_indexing()
    wrong = _opposite_map_indexing(detected)
    monkeypatch.setenv("XENOSITE_FOREST_MAP_INDEXING", wrong.value)
    _clear_forest_indexing_caches()

    rdmol = Chem.MolFromSmiles(ETHYLENE)
    rs = load_ruleset("SO.Epoxidation")
    product = next(rs.metabolites(rdmol, unique=True))[1][-1]
    maps = metabolite_map_indices(product)

    with pytest.raises(AssertionError):
        _assert_ethylene_epoxidation_map_idx(maps)

    _clear_forest_indexing_caches()
    monkeypatch.delenv("XENOSITE_FOREST_MAP_INDEXING", raising=False)
    _assert_ethylene_epoxidation_map_idx(metabolite_map_indices(product))


def test_ethylene_epoxidation_map_idx():
    from xenosite.forest import load_ruleset

    rdmol = Chem.MolFromSmiles(ETHYLENE)
    rs = load_ruleset("SO.Epoxidation")
    (rule, site), mols = next(rs.metabolites(rdmol, unique=True))
    product = mols[-1]
    _assert_ethylene_epoxidation_map_idx(metabolite_map_indices(product))


def test_attach_metabolites_map_idx_always_mapped_smiles_optional():
    mol = Molecule(
        smiles=ETHYLENE,
        atoms=Atoms(num=2),
        bonds=Bonds(idx=[(0, 1)]),
        results=[MolBondResult(model="epoxidation", version="0", mol=0.9, bond=[0.85])],
    )
    attach_metabolites(mol)
    met = mol.results[0].metabolite[0]
    assert met.map_idx
    assert met.mapped_smiles is None
    assert met.atom == [0, 1]
    assert 0 in met.map_idx

    mol2 = Molecule(
        smiles=ETHYLENE,
        atoms=Atoms(num=2),
        bonds=Bonds(idx=[(0, 1)]),
        results=[MolBondResult(model="epoxidation", version="0", mol=0.9, bond=[0.85])],
    )
    attach_metabolites(mol2, mapped_smiles=True)
    met2 = mol2.results[0].metabolite[0]
    assert met2.map_idx
    assert met2.mapped_smiles
    assert ":" in met2.mapped_smiles
    assert met2.smiles == "C1CO1"


@pytest.mark.skipif(not _ob.installed(), reason="OpenBabel not installed")
@pytest.mark.skipif(not onnx_weights_present("epoxidation"), reason="no epoxidation ONNX")
def test_predict_mapped_smiles_end_to_end():
    be = OnnxBackend(onnx_root())
    mol = predict(
        ETHYLENE,
        model="epoxidation",
        backend=be,
        metabolites=True,
        mapped_smiles=True,
    )
    met = next(r for r in mol.results if r.model == "epoxidation").metabolite[0]
    assert met.map_idx
    assert met.mapped_smiles
    assert ":" in met.mapped_smiles
    assert max(met.map_idx) <= mol.atoms.num
