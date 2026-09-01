"""Unit tests for ``tests/sampling.py`` corpus helpers and Hypothesis settings."""

from __future__ import annotations

from hypothesis import given
from hypothesis import settings as hyp_settings
from hypothesis import strategies as st

from tests.v0_legacy.sampling import (
    _SCORE_MODELS,
    equiv_models_for_smiles,
    equiv_pairs,
    equiv_smiles,
    equiv_smiles_model,
    golden_parity_models_for_smiles,
    golden_parity_pairs,
    golden_parity_smiles,
    golden_parity_smiles_model,
    golden_suite_models_for_smiles,
    golden_suite_pairs,
    golden_suite_smiles,
    golden_suite_smiles_model,
    molecule_sample_settings,
    ob_dump_models_for_smiles,
    ob_dump_pairs,
    ob_dump_smiles,
    ob_dump_smiles_model,
    principled_descriptor_pairs,
    sample_max_examples,
)
from tests.support import MODELS

_ATOM_MODELS = ("quinone", "reactivity", "ugt")
_BOND_MODELS = ("epoxidation", "ndealk")


def test_sample_max_examples_empty():
    assert sample_max_examples(()) == 1


def test_sample_max_examples_defaults_to_corpus_size():
    corpus = ("smi-a", "smi-b", "smi-c")
    assert sample_max_examples(corpus) == 3


def test_sample_max_examples_env_override(monkeypatch):
    monkeypatch.setenv("XENOSITE_HYPOTHESIS_MAX_EXAMPLES", "2")
    assert sample_max_examples(("a", "b", "c", "d")) == 2


def test_sample_max_examples_env_override_caps_at_corpus(monkeypatch):
    monkeypatch.setenv("XENOSITE_HYPOTHESIS_MAX_EXAMPLES", "99")
    assert sample_max_examples(("only",)) == 1


def test_molecule_sample_settings_from_tuple():
    cfg = molecule_sample_settings(("x", "y", "z"))
    assert cfg.max_examples == 3
    assert cfg.deadline is None


def test_molecule_sample_settings_from_callable():
    cfg = molecule_sample_settings(lambda: ("x", "y"))
    assert cfg.max_examples == 2


def test_molecule_sample_settings_explicit_max_examples():
    cfg = molecule_sample_settings(("a", "b", "c"), max_examples=1)
    assert cfg.max_examples == 1


def test_ob_dump_corpus_consistent():
    smiles = ob_dump_smiles(MODELS)
    pairs = ob_dump_pairs(MODELS)
    assert smiles
    assert pairs
    assert len(pairs) >= len(smiles)
    for smi, model in pairs:
        assert smi in smiles
        assert model in ob_dump_models_for_smiles(smi, MODELS)


def test_equiv_corpus_consistent():
    for smi in equiv_smiles():
        models = equiv_models_for_smiles(smi)
        assert models
        assert smi in ob_dump_smiles(_SCORE_MODELS)
        for model in models:
            assert (smi, model) in equiv_pairs()


def test_golden_suite_corpus_consistent():
    for smi in golden_suite_smiles():
        models = golden_suite_models_for_smiles(smi)
        assert models
        for model in models:
            assert (model, smi) in golden_suite_pairs()


def test_golden_parity_corpus_consistent():
    for smi in golden_parity_smiles():
        models = golden_parity_models_for_smiles(smi)
        assert models
        for model in models:
            assert (model, smi) in golden_parity_pairs()


def test_principled_descriptor_pairs_subset_of_ob_dump():
    atom_pairs = principled_descriptor_pairs(_ATOM_MODELS)
    bond_pairs = principled_descriptor_pairs(_BOND_MODELS)
    assert atom_pairs
    assert bond_pairs
    dump_pairs = set(ob_dump_pairs(_ATOM_MODELS + _BOND_MODELS))
    assert set(atom_pairs) <= dump_pairs
    assert set(bond_pairs) <= dump_pairs


@hyp_settings(max_examples=5, deadline=None)
@given(data=st.data())
def test_ob_dump_smiles_model_draw(data):
    smiles, model = data.draw(ob_dump_smiles_model(MODELS))
    assert model in ob_dump_models_for_smiles(smiles, MODELS)


@hyp_settings(max_examples=5, deadline=None)
@given(data=st.data())
def test_equiv_smiles_model_draw(data):
    smiles, model = data.draw(equiv_smiles_model())
    assert model in equiv_models_for_smiles(smiles)


@hyp_settings(max_examples=5, deadline=None)
@given(data=st.data())
def test_golden_suite_smiles_model_draw(data):
    model, smiles = data.draw(golden_suite_smiles_model())
    assert model in golden_suite_models_for_smiles(smiles)


@hyp_settings(max_examples=5, deadline=None)
@given(data=st.data())
def test_golden_parity_smiles_model_draw(data):
    model, smiles = data.draw(golden_parity_smiles_model())
    assert model in golden_parity_models_for_smiles(smiles)
