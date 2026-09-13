"""N-dealkylation: ND ruleset mapping, N-only metabolites, zero non-N bond scores."""

from __future__ import annotations

import pytest
from rdkit import Chem

from xenosite.predict import predict
from xenosite.predict.backends.onnx import OnnxBackend
from xenosite.predict.features import _ob
from xenosite.predict.forest import (
    attach_metabolites,
    ruleset_for_model,
)
from xenosite.predict.molecule import parse_smiles
from xenosite.predict.types import BondResult

from tests.support import onnx_root, onnx_weights_present
from tests.test_forest_metabolites import _bond_scores, _unique_forest_count

# Amine + ether (diphenhydramine): N–C and O–C / C–C sites.
DIPHENHYDRAMINE = "CN(C)CCOC(c1ccccc1)c2ccccc2"
ETHER_ONLY = "CCOC"
NDEALK = "CN(C)Cc1ccccc1"


def _site_has_nitrogen(rdmol, atom_idxs) -> bool:
    return any(rdmol.GetAtomWithIdx(i).GetAtomicNum() == 7 for i in atom_idxs)


def test_ruleset_for_ndealk_and_isozyme_is_nd():
    assert ruleset_for_model("ndealk") == "ND"
    assert ruleset_for_model("isozyme.3a4") == "ND"
    assert ruleset_for_model("isozyme.hlm") == "ND"


def test_phase1_unstable_oxygenation_still_uses_uo():
    """Regression: full UO dealkylation (incl. non-N) must remain available."""
    assert ruleset_for_model("phase1.unstable_oxygenation") == "UO"


def test_attach_ndealk_metabolites_are_nitrogen_sites_only():
    rdmol, mol = parse_smiles(DIPHENHYDRAMINE)
    mol.results = [
        BondResult(
            model="ndealk",
            model_version="0",
            bond=[0.5] * len(mol.bonds.idx),
        )
    ]
    attach_metabolites(mol, rdmol=rdmol)
    mets = mol.results[0].metabolite
    assert mets
    assert len(mets) == _unique_forest_count(rdmol, "ND")
    assert len(mets) < _unique_forest_count(rdmol, "UO.Dealkylation")
    for m in mets:
        assert _site_has_nitrogen(rdmol, m.atom or [])


def test_attach_phase1_uo_still_emits_non_nitrogen_dealk_sites():
    """Regression: phase1 UO must not inherit the N-only filter."""
    rdmol, mol = parse_smiles(DIPHENHYDRAMINE)
    mol.results = [
        BondResult(
            model="phase1.unstable_oxygenation",
            model_version="0",
            bond=[0.5] * len(mol.bonds.idx),
        )
    ]
    attach_metabolites(mol, rdmol=rdmol)
    mets = mol.results[0].metabolite
    assert mets
    assert any(not _site_has_nitrogen(rdmol, m.atom or []) for m in mets)


@pytest.mark.skipif(not _ob.installed(), reason="OpenBabel not installed")
@pytest.mark.skipif(not onnx_weights_present("ndealk"), reason="no ndealk ONNX")
def test_ndealk_scores_zero_on_bonds_without_nitrogen():
    be = OnnxBackend(onnx_root())
    mol = predict(DIPHENHYDRAMINE, model="ndealk", backend=be)
    result = next(r for r in mol.results if r.model == "ndealk")
    rdmol = Chem.MolFromSmiles(DIPHENHYDRAMINE)
    assert rdmol is not None
    saw_n_bond = False
    for score, (a, b) in zip(result.bond, mol.bonds.idx):
        has_n = (
            rdmol.GetAtomWithIdx(a).GetAtomicNum() == 7
            or rdmol.GetAtomWithIdx(b).GetAtomicNum() == 7
        )
        if has_n:
            saw_n_bond = True
        else:
            assert score == 0.0, f"non-N bond {(a, b)} scored {score}"
    assert saw_n_bond
    assert any(s > 0.0 for s in result.bond)


@pytest.mark.skipif(not _ob.installed(), reason="OpenBabel not installed")
@pytest.mark.skipif(not onnx_weights_present("ndealk"), reason="no ndealk ONNX")
def test_isozyme_scores_zero_on_bonds_without_nitrogen():
    be = OnnxBackend(onnx_root())
    mol = predict(DIPHENHYDRAMINE, model="isozyme", backend=be)
    result = next(r for r in mol.results if r.model == "isozyme.3a4")
    rdmol = Chem.MolFromSmiles(DIPHENHYDRAMINE)
    for score, (a, b) in zip(result.bond, mol.bonds.idx):
        has_n = (
            rdmol.GetAtomWithIdx(a).GetAtomicNum() == 7
            or rdmol.GetAtomWithIdx(b).GetAtomicNum() == 7
        )
        if not has_n:
            assert score == 0.0


@pytest.mark.skipif(not _ob.installed(), reason="OpenBabel not installed")
@pytest.mark.skipif(not onnx_weights_present("ndealk"), reason="no ndealk ONNX")
def test_ndealk_short_circuits_when_molecule_has_no_nitrogen():
    """No N → all-zero bond vector (no reason to run the model)."""
    be = OnnxBackend(onnx_root())
    mol = predict(ETHER_ONLY, model="ndealk", backend=be)
    result = next(r for r in mol.results if r.model == "ndealk")
    assert result.bond
    assert all(s == 0.0 for s in result.bond)
    attach_metabolites(mol)
    assert not result.metabolite
