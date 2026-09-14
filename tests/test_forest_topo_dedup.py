"""Forest topological emission must match XenoSite UI identity keys.

Requires ``xenosite-forest>=0.2.8`` (pathway + sorted topo ranks + product SMILES).
Fails on 0.2.7, which still emits duplicate topo-equivalent sites.
"""

from __future__ import annotations

from rdkit import Chem

from xenosite.predict.forest import enumerate_metabolites
from xenosite.predict.molecule import parse_smiles

BENZENE = "c1ccccc1"
NDEALK = "CN(C)Cc1ccccc1"
PHENOL = "Oc1ccccc1"
NAPHTHALENE = "c1ccc2ccccc2c1"


def _ui_keys(rdmol, rows):
    # Rank before enumeration: forest metabolize kekulizes the reactant in place.
    ranks = list(
        Chem.CanonicalRankAtoms(rdmol, includeChirality=False, breakTies=False)
    )
    return {
        (pathway, smiles, tuple(sorted(ranks[i] for i in site)))
        for pathway, site, smiles, _ in rows
    }


def _assert_topo_dedup(smiles: str, ruleset: str, *, expected_n: int | None = None) -> None:
    rdmol, _ = parse_smiles(smiles)
    # Snapshot topology on the aromatic reactant (matches forest metabolize).
    ranks = list(
        Chem.CanonicalRankAtoms(rdmol, includeChirality=False, breakTies=False)
    )
    rows = list(enumerate_metabolites(rdmol, ruleset))
    keys = {
        (pathway, smiles_i, tuple(sorted(ranks[i] for i in site)))
        for pathway, site, smiles_i, _ in rows
    }
    assert rows, f"no metabolites for {smiles} / {ruleset}"
    assert len(rows) == len(keys), (
        f"{smiles} / {ruleset}: emitted {len(rows)} rows but only {len(keys)} "
        f"UI topo keys (forest <0.2.8 leaks symmetry-equivalent sites)"
    )
    if expected_n is not None:
        assert len(keys) == expected_n


def test_enumerate_benzene_epoxidation_single_topo_product():
    _assert_topo_dedup(BENZENE, "SO.Epoxidation", expected_n=1)


def test_enumerate_phenol_epoxidation_ortho_meta_para():
    _assert_topo_dedup(PHENOL, "SO.Epoxidation", expected_n=3)


def test_enumerate_naphthalene_quinone_topo_dedup():
    _assert_topo_dedup(NAPHTHALENE, "QF.QuinoneFormation")


def test_enumerate_ndealk_topo_dedup():
    _assert_topo_dedup(NDEALK, "UO.Dealkylation")


def test_enumerate_topo_dedup_keeps_distinct_product_smiles():
    """Collapse must not drop chemically distinct product SMILES."""
    rdmol, _ = parse_smiles(NAPHTHALENE)
    ranks = list(
        Chem.CanonicalRankAtoms(rdmol, includeChirality=False, breakTies=False)
    )
    rows = list(enumerate_metabolites(rdmol, "QF.QuinoneFormation"))
    keys = {
        (pathway, smiles_i, tuple(sorted(ranks[i] for i in site)))
        for pathway, site, smiles_i, _ in rows
    }
    assert len(rows) == len(keys)
    smiles = {s for _p, _site, s, _m in rows}
    # More than one quinone regioisomer / pattern on naphthalene
    assert len(smiles) >= 2
