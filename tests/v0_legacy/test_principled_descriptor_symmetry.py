"""Principled feature rows are identical within each symmetry class.

Atom models (quinone ``omp_mode=principled``, reactivity, ugt): identical ONNX
columns within each multi-member RDKit atom rank.

Bond models (epoxidation, ndealk): ``bond_nrings_mode=principled`` uses RDKit
``RingInfo.NumAtomRings`` per directed BondTD endpoint; rows in the same
directed OpenBabel bond class ``(GID(Atom1), GID(Atom2), bond order)`` must
match. Legacy ``bond_nrings_mode=legacy`` keeps DFS back-edge atom counts for
ob dump / golden parity (``test_ob_features.py``).
"""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from xenosite.predict.features import load_names
from xenosite.predict.molecule import parse_smiles

from tests.v0_legacy.rdkit_equiv import (
    assert_atom_descriptor_rows_symmetric,
    assert_bond_descriptor_rows_symmetric,
)
from tests.v0_legacy.sampling import (
    molecule_sample_settings,
    ob_dump_smiles,
    principled_descriptor_smiles_model,
)
from tests.support import (
    PRINCIPLED_PARAMETER,
    golden_name_by_smiles,
    load_ob_dumps,
    rows_for_model,
)

_ATOM_MODELS = ("quinone", "reactivity", "ugt")
_BOND_MODELS = ("epoxidation", "ndealk")
_ALL_MODELS = _ATOM_MODELS + _BOND_MODELS
_ATOM_PARAMS_CACHE: list | None = None
_BOND_PARAMS_CACHE: list | None = None


def _params(models: tuple[str, ...], cache_name: str):
    cache = globals()[cache_name]
    if cache is not None:
        return cache
    names = golden_name_by_smiles()
    params = []
    used: set[str] = set()
    for mol in load_ob_dumps():
        smi = mol.get("smiles") or ""
        label = names.get(smi) or smi[:32]
        for model in models:
            if model not in (mol.get("models") or {}):
                continue
            pid = f"{model}:{label}"
            if pid in used:
                pid = f"{model}:{smi[:40]}"
            used.add(pid)
            params.append(pytest.param(smi, model, id=pid))
    globals()[cache_name] = params
    return params


def test_principled_descriptor_symmetry_fixture_present():
    atom_params = _params(_ATOM_MODELS, "_ATOM_PARAMS_CACHE")
    bond_params = _params(_BOND_MODELS, "_BOND_PARAMS_CACHE")
    assert atom_params, "need ob_dumps with atom model rows"
    assert bond_params, "need ob_dumps with bond model rows"
    assert len(atom_params) >= 100


def _assert_principled_atom_descriptors(smiles: str, model: str) -> None:
    rdmol, _ = parse_smiles(smiles)
    rows = rows_for_model(
        model,
        rdmol,
        omp_mode="principled" if model == "quinone" else None,
        _parameter=PRINCIPLED_PARAMETER,
    )
    names = load_names(model, "atom")
    assert names, f"missing {model}/atom names in name_tables"
    assert_atom_descriptor_rows_symmetric(rows, rdmol, names)


def _assert_principled_bond_descriptors(smiles: str, model: str) -> None:
    rdmol, _ = parse_smiles(smiles)
    rows = rows_for_model(model, rdmol, _parameter=PRINCIPLED_PARAMETER)
    names = load_names(model, "bond")
    assert names, f"missing {model}/bond names in name_tables"
    assert_bond_descriptor_rows_symmetric(rows, rdmol, names)


@molecule_sample_settings(lambda: ob_dump_smiles(_ATOM_MODELS))
@given(data=st.data())
def test_principled_atom_descriptors_identical_within_rdkit_class_sampled(data):
    smiles, model = data.draw(principled_descriptor_smiles_model(_ATOM_MODELS))
    _assert_principled_atom_descriptors(smiles, model)


@pytest.mark.full
@pytest.mark.parametrize("smiles,model", _params(_ATOM_MODELS, "_ATOM_PARAMS_CACHE"))
def test_principled_atom_descriptors_identical_within_rdkit_class(smiles, model):
    _assert_principled_atom_descriptors(smiles, model)


@molecule_sample_settings(lambda: ob_dump_smiles(_BOND_MODELS))
@given(data=st.data())
def test_principled_bond_descriptors_identical_within_directed_ob_class_sampled(data):
    smiles, model = data.draw(principled_descriptor_smiles_model(_BOND_MODELS))
    _assert_principled_bond_descriptors(smiles, model)


@pytest.mark.full
@pytest.mark.parametrize("smiles,model", _params(_BOND_MODELS, "_BOND_PARAMS_CACHE"))
def test_principled_bond_descriptors_identical_within_directed_ob_class(smiles, model):
    _assert_principled_bond_descriptors(smiles, model)
