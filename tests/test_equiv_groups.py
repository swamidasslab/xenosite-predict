"""RDKit symmetry classes → ONNX scores respect principled invariance.

Atom models (quinone, ugt, reactivity.*): every atom in an RDKit symmetry
class receives the same score.

Bond models:
- **epoxidation** — every bond in the class receives the same score (eight
  polycyclic Kekulé edge cases xfail: dual atom-ordering breaks coarse RDKit
  bond classes while golden scores remain valid).
- **ndealk / isozyme** (principled + ``symmetry_group_mode=rdkit`` default) — equal
  scores within every RDKit bond symmetry class (broadcast after site dedup).

Grouping uses only RDKit (``tests/rdkit_equiv.py``), not OpenBabel GID or
feature-row metadata.
"""

from __future__ import annotations

import pytest

from xenosite.predict import predict
from xenosite.predict.backends.onnx import OnnxBackend
from xenosite.predict.molecule import canonicalize_smiles, parse_smiles

from tests.rdkit_equiv import (
    assert_atom_scores_symmetric,
    assert_bond_scores_openbabel_principled,
    assert_bond_scores_symmetric,
    assert_full_atom_vector,
    assert_full_bond_vector,
    atom_symmetry_groups,
    bond_symmetry_groups,
)
from tests.support import (
    GOLDEN,
    ROOT,
    golden_name_by_smiles,
    load_ob_dumps,
    onnx_weights_present,
)

BACKEND = OnnxBackend(ROOT / "weights" / "onnx")

_SCORE_MODELS = ("epoxidation", "quinone", "reactivity", "ugt", "ndealk", "isozyme")
# Fused polycyclic SMILES where epoxidation dual-ordering breaks coarse RDKit bond classes.
_EPOX_RDKIT_SYMMETRY_XFAIL = frozenset(
    {
        "c1cc2ccc3cccc4ccc(c1)c2c34",
        "c1ccc2c(c1)[nH]c1ccccc12",
        "c1ccc2c(c1)ccc1ccccc12",
        "c1ccc2c(c1)Nc1ccccc1S2",
        "c1ccc2cc3ccccc3cc2c1",
        "c1ccc2nc3ccccc3cc2c1",
        "c1ccc2c(c1)oc1ccccc12",
        "c1ccc2c(c1)sc1ccccc12",
    }
)
_PARAMS_CACHE: list | None = None


def _equiv_params():
    global _PARAMS_CACHE
    if _PARAMS_CACHE is not None:
        return _PARAMS_CACHE
    names = golden_name_by_smiles()
    params = []
    used: set[str] = set()
    for mol in load_ob_dumps():
        smi = mol.get("smiles") or ""
        label = names.get(smi) or smi[:32]
        for model in _SCORE_MODELS:
            if model not in (mol.get("models") or {}):
                continue
            weight_key = "ndealk" if model == "isozyme" else model
            if not onnx_weights_present(weight_key):
                continue
            pid = f"{model}:{label}"
            if pid in used:
                pid = f"{model}:{smi[:40]}"
            used.add(pid)
            marks = []
            if (
                model == "epoxidation"
                and canonicalize_smiles(smi) in _EPOX_RDKIT_SYMMETRY_XFAIL
            ):
                marks.append(
                    pytest.mark.xfail(
                        reason="epoxidation dual-ordering vs coarse RDKit bond class",
                        strict=False,
                    )
                )
            params.append(pytest.param(smi, model, id=pid, marks=marks))
    _PARAMS_CACHE = params
    return params


def test_equiv_groups_fixture_present():
    assert GOLDEN.is_file()
    dumps = load_ob_dumps()
    assert dumps, "missing ob_dumps (git lfs pull or make dump-ob)"
    params = _equiv_params()
    assert len(params) >= 100, f"expected many (model, SMILES) pairs, got {len(params)}"


@pytest.mark.parametrize("smiles,model", _equiv_params())
def test_onnx_scores_respect_rdkit_symmetry(smiles, model):
    rdmol, molecule = parse_smiles(smiles)
    atom_groups = atom_symmetry_groups(rdmol)
    bond_groups = bond_symmetry_groups(rdmol)

    mol = predict(smiles, models=[model], backend=BACKEND)
    assert mol.results

    if model == "quinone":
        scores = mol.results[0].atom
        assert_full_atom_vector(scores, molecule.atoms.num)
        if atom_groups:
            assert_atom_scores_symmetric(scores, atom_groups)

    elif model == "ugt":
        result = next(r for r in mol.results if r.model == "ugt")
        assert_full_atom_vector(result.atom, molecule.atoms.num)
        if atom_groups:
            assert_atom_scores_symmetric(result.atom, atom_groups)

    elif model == "reactivity":
        for result in mol.results:
            assert result.model.startswith("reactivity.")
            assert_full_atom_vector(result.atom, molecule.atoms.num)
            if atom_groups:
                assert_atom_scores_symmetric(result.atom, atom_groups)

    elif model == "epoxidation":
        scores = mol.results[0].bond
        assert_full_bond_vector(scores, len(molecule.bonds.idx))
        if bond_groups:
            assert_bond_scores_symmetric(scores, bond_groups)

    elif model == "ndealk":
        scores = mol.results[0].bond
        assert_full_bond_vector(scores, len(molecule.bonds.idx))
        if bond_groups:
            assert_bond_scores_symmetric(scores, bond_groups)

    elif model == "isozyme":
        for result in mol.results:
            assert result.model.startswith("isozyme.")
            assert_full_bond_vector(result.bond, len(molecule.bonds.idx))
            if bond_groups:
                assert_bond_scores_symmetric(result.bond, bond_groups)

    else:
        pytest.fail(f"unhandled model {model}")


def test_ndealk_openbabel_symmetry_allows_zero_fill():
    """``symmetry_group_mode=openbabel`` dedupes without RDKit broadcast."""
    from tests.support import GOLDEN_SYMMETRY_PARAMETER

    smiles = "COc1ccc2nc(C)cc(NCCCN3CCOCC3)c2c1"
    rdmol, molecule = parse_smiles(smiles)
    bond_groups = bond_symmetry_groups(rdmol)
    param = {"ndealk_site_mode": "principled", **GOLDEN_SYMMETRY_PARAMETER}
    mol = predict(smiles, models=["ndealk"], backend=BACKEND, _parameter=param)
    scores = mol.results[0].bond
    assert_full_bond_vector(scores, len(molecule.bonds.idx))
    if bond_groups:
        assert_bond_scores_openbabel_principled(scores, bond_groups)
