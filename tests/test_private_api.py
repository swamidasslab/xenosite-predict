"""Tests for ``xenosite.predict._private`` (xenosite-api integration surface)."""

from __future__ import annotations

import pytest

from xenosite.predict._private import (
    add_metabolites,
    attach_metabolites,
    metabolite_supported,
    supported_metabolite_models,
)


def test_add_metabolites_is_attach_metabolites_alias():
    assert add_metabolites is attach_metabolites
from xenosite.predict.types import Atoms, Bonds, MolBondResult, Molecule


def test_supported_metabolite_models():
    assert "epoxidation" in supported_metabolite_models()
    assert "ugt" in supported_metabolite_models()
    assert "reactivity.gsh" in supported_metabolite_models()
    assert "reactivity.protein" in supported_metabolite_models()
    assert "reactivity" not in supported_metabolite_models()
    assert "reactivity.cyanide" not in supported_metabolite_models()


def test_metabolite_supported_isozyme_prefix():
    assert metabolite_supported("isozyme.3a4")
    assert metabolite_supported("ugt")
    assert metabolite_supported("reactivity.gsh")
    assert metabolite_supported("reactivity.protein")
    assert not metabolite_supported("reactivity.cyanide")


def test_add_metabolites_skips_unsupported_models():
    mol = Molecule(
        smiles="C=C",
        atoms=Atoms(num=2),
        bonds=Bonds(idx=[(0, 1)]),
        results=[
            MolBondResult(model="epoxidation", model_version="0", mol=0.9, bond=[0.85]),
        ],
    )
    add_metabolites(mol, models={"ndealk"})
    assert mol.results[0].metabolite is None


def test_add_metabolites_attaches_when_model_filtered():
    mol = Molecule(
        smiles="C=C",
        atoms=Atoms(num=2),
        bonds=Bonds(idx=[(0, 1)]),
        results=[
            MolBondResult(model="epoxidation", model_version="0", mol=0.9, bond=[0.85]),
        ],
    )
    add_metabolites(mol, models={"epoxidation"})
    assert mol.results[0].metabolite
    assert mol.results[0].metabolite[0].map_idx
    assert mol.results[0].metabolite[0].score == pytest.approx(0.85)
