"""RDKit 2026 valence-cache shims for forest (no ONNX)."""

from __future__ import annotations

from rdkit import Chem

from xenosite.forest import PhaseOneRS
from xenosite.forest.rules import Dehydrogenation
from xenosite.predict.forest_rdkit import apply_rdkit2026_patches, refresh_mol

# Diphenhydramine: suite crash in Dehydrogenation RunReactants on resonance copies.
DIPHENHYDRAMINE = "CN(C)CCOC(c1ccccc1)c1ccccc1"
IBUPROFEN = "CC(C)Cc1ccc(C(C)C(=O)O)cc1"
ETHANE = "CC"


def test_patches_are_idempotent():
    apply_rdkit2026_patches()
    apply_rdkit2026_patches()


def test_dehydrogenation_diphenhydramine_does_not_crash():
    mol = Chem.MolFromSmiles(DIPHENHYDRAMINE)
    n = sum(1 for _ in Dehydrogenation().metabolize(mol))
    assert n > 0


def test_phaseone_unique_on_suite_crashers():
    for smi in (DIPHENHYDRAMINE, IBUPROFEN):
        mol = Chem.MolFromSmiles(smi)
        rows = list(PhaseOneRS.metabolites(mol, unique=True))
        assert rows, smi


def test_phaseone_ethane_still_enumerates():
    mol = Chem.MolFromSmiles(ETHANE)
    rows = list(PhaseOneRS.metabolites(mol, unique=True))
    assert rows


def test_refresh_mol_allows_runreactants_on_resonance_copy():
    mol = Chem.MolFromSmiles(DIPHENHYDRAMINE)
    D = Dehydrogenation()
    res = list(D.resonance_structures(mol))
    assert len(res) > 1
    copy = res[1]
    refresh_mol(copy)
    prods = D.rxns[1].RunReactants([copy])
    assert len(prods) >= 1
