"""Tests for heavy-atom numbering (legacy gapped OB keys vs dense RDKit)."""

from __future__ import annotations

import numpy as np
import pytest

from xenosite.predict.numbering import (
    ObNumberingMode,
    detect_raw_numbering_mode,
    legacy_keys_are_gapped,
    legacy_site_to_atom_vector,
    map_legacy_ob_to_rdkit,
    map_legacy_pair_to_rdkit,
    numbering_mode,
    pattern_ob_index,
    raw_numbering_is_gapped,
)
from xenosite.predict.features import _ob


GAP_SMILES = "COc1cc2nc(SCc3ccccc3C)[nH]c2cc1OC"
OK_SMILES = "CC(=O)Oc1ccccc1C(=O)O"


def test_numbering_mode_default_dense(monkeypatch):
    monkeypatch.delenv("XENOSITE_OB_NUMBERING", raising=False)
    assert numbering_mode() == ObNumberingMode.DENSE_HEAVY


def test_numbering_mode_raw_env(monkeypatch):
    monkeypatch.setenv("XENOSITE_OB_NUMBERING", "raw")
    assert numbering_mode() == ObNumberingMode.RAW


def test_pattern_ob_index_respects_mode():
    assert pattern_ob_index(20, 18, ObNumberingMode.RAW) == 20
    assert pattern_ob_index(20, 18, ObNumberingMode.DENSE_HEAVY) == 18


def test_gapped_legacy_keys_detection():
    keys = list(range(1, 18)) + list(range(19, 24))
    assert legacy_keys_are_gapped(keys, 22)
    assert not legacy_keys_are_gapped(list(range(1, 14)), 13)


def test_legacy_site_row_order_mapping():
    """Gapped legacy keys map by sorted row order → RDKit index."""
    site = {i: float(i) for i in list(range(1, 18)) + list(range(19, 24))}
    vec = legacy_site_to_atom_vector(site, 22)
    assert len(vec) == 22
    assert vec[17] == 19.0
    assert vec[18] == 20.0


def test_dense_legacy_site_uses_one_based():
    site = {i: float(i) * 0.1 for i in range(1, 14)}
    vec = legacy_site_to_atom_vector(site, 13)
    assert np.allclose(vec[0], 0.1)
    assert np.allclose(vec[12], 1.3)


def test_map_legacy_ob_gapped():
    order = list(range(1, 18)) + list(range(19, 24))
    assert map_legacy_ob_to_rdkit(20, order, 22) == 18
    assert map_legacy_ob_to_rdkit(18, order, 22) == 17  # missing key → position fallback


def test_map_legacy_pair_gapped():
    order = list(range(1, 18)) + list(range(19, 24))
    assert map_legacy_pair_to_rdkit(19, 20, legacy_ob_order=order, n_heavy=22) == (17, 18)


@pytest.mark.skipif(not _ob.available(), reason="OpenBabel not installed")
def test_ob_molblock_numbering_is_dense():
    from rdkit import Chem

    from xenosite.predict.numbering import heavy_atom_raw_indices

    _, pybel = _ob.load()
    mol = pybel.readstring("mol", Chem.MolToMolBlock(Chem.MolFromSmiles(GAP_SMILES)))
    raw = heavy_atom_raw_indices(mol.OBMol)
    assert not raw_numbering_is_gapped(raw)
    assert detect_raw_numbering_mode(mol.OBMol) == ObNumberingMode.DENSE_HEAVY


@pytest.mark.skipif(not _ob.available(), reason="OpenBabel not installed")
def test_ob_smiles_gapped_detection_optional():
    """SMILES read path may differ by OB version; flag documents runtime layout."""
    ob, pybel = _ob.load()
    mol = pybel.readstring("smi", GAP_SMILES)
    mode = detect_raw_numbering_mode(mol.OBMol)
    assert mode in (ObNumberingMode.RAW, ObNumberingMode.DENSE_HEAVY)


@pytest.mark.live
def test_legacy_reactivity_gapped_smiles_aligns_with_onnx(legacy_api_url):
    """End-to-end: legacy ingest uses row-order numbering on [nH] gap molecule."""
    from pathlib import Path

    from xenosite.predict import predict
    from xenosite.predict.backends.legacy import LegacyTestBackend
    from xenosite.predict.backends.onnx import OnnxBackend

    be = LegacyTestBackend(legacy_api_url)
    onx = OnnxBackend(Path(__file__).resolve().parents[1] / "weights" / "onnx")
    leg = predict(GAP_SMILES, models=["reactivity"], backend=be)
    ort = predict(GAP_SMILES, models=["reactivity"], backend=onx)
    lg = next(r for r in leg.results if r.model == "reactivity.gsh")
    og = next(r for r in ort.results if r.model == "reactivity.gsh")
    diff = np.max(np.abs(np.array(lg.atom) - np.array(og.atom)))
    assert diff < 0.05, f"legacy vs onnx gsh max diff {diff}"


def test_quinone_normalize_parses_smiles_once(monkeypatch):
    from xenosite.predict import molecule as mol_mod
    from xenosite.predict.numbering import (
        normalize_quinone_pair_fields,
        quinone_normalize_context,
    )
    from xenosite.predict.molecule import parse_smiles

    quinone_normalize_context.cache_clear()
    calls: list[str] = []
    real_parse = mol_mod.parse_smiles

    def counted(smiles: str, *, detailed: bool = False):
        calls.append(smiles)
        return real_parse(smiles, detailed=detailed)

    monkeypatch.setattr(mol_mod, "parse_smiles", counted)
    fields = {"pair_idx": [[1, 2]], "pair": [0.5]}
    mol, _ = real_parse(OK_SMILES)
    normalize_quinone_pair_fields(dict(fields), dict(fields), mol=mol)
    assert len(calls) == 0

    quinone_normalize_context.cache_clear()
    calls.clear()
    normalize_quinone_pair_fields(dict(fields), dict(fields), smiles=OK_SMILES)
    assert len(calls) == 1


def test_quinone_pair_fields_leaves_onnx_rdkit_indices():
    """ONNX pair_idx is already 0-based RDKit; only golden legacy ids remap."""
    from xenosite.predict.molecule import parse_smiles
    from xenosite.predict.numbering import normalize_quinone_pair_fields

    want = {"pair_idx": [[19, 22]], "pair": [0.5]}
    have = {"pair_idx": [[18, 21]], "pair": [0.5]}
    mol, _ = parse_smiles(GAP_SMILES)
    normalize_quinone_pair_fields(want, have, mol=mol)
    assert want["pair_idx"] == [[18, 21]]
    assert have["pair_idx"] == [[18, 21]]


def test_quinone_pair_fields_skips_already_rdkit_golden():
    from xenosite.predict.molecule import parse_smiles
    from xenosite.predict.numbering import normalize_quinone_pair_fields

    want = {"pair_idx": [[2, 20]], "pair": [0.5]}
    have = {"pair_idx": [[2, 17]], "pair": [0.5]}
    mol, _ = parse_smiles(GAP_SMILES)
    normalize_quinone_pair_fields(want, have, mol=mol)
    assert want["pair_idx"] == [[2, 20]]
    assert have["pair_idx"] == [[2, 17]]


def test_ok_smiles_dense():
    site = {i: 0.1 for i in range(1, 14)}
    vec = legacy_site_to_atom_vector(site, 13)
    assert len(vec) == 13
