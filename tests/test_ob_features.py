"""Internal OpenBabel features vs dump-oracle rows.

Dumps come from committed ``ob_dumps.json.gz`` (Git LFS) or ``make dump-ob``.
Missing dumps fail. Each (model, molecule) dump is its own test. Do not loosen atol.

Quinone ortho/meta/para mismatches that are *only* those columns are xfailed:
the dump picked one BFS path via Python 2 ``set`` order; we use any shortest
path on an aromatic ring instead. Revisit if golden scores move.

Phase1 dumps concatenate Bond_and_LonePair with Possible_Sites SMARTS masks.
We port BLP only (404 site.onnx inputs); dump comparison skips Possible_Sites.
Revive ``possible_site_flags`` and OB 2.4 SMARTS parity if we expose legacy
``ReactionType`` sub-scores (post-NN class × mask), not for class golden heads.
"""

from __future__ import annotations

import pytest

from xenosite.predict.molecule import parse_smiles

from tests.support import (
    MODELS,
    OB_ASPIRIN,
    OB_DUMPS_GZ,
    compare_feature_dump_rows,
    dump_compare_skip_columns,
    golden_name_by_smiles,
    load_descriptor_smiles,
    load_ob_dump,
    load_ob_dumps,
    rows_for_model,
)


_DUMPS_CACHE: list[dict] | None = None


def _dumps() -> list[dict]:
    global _DUMPS_CACHE
    if _DUMPS_CACHE is None:
        _DUMPS_CACHE = load_ob_dumps()
    return _DUMPS_CACHE


def _dump_params():
    names = golden_name_by_smiles()
    params = []
    used: set[str] = set()
    for mol in _dumps():
        smi = mol.get("smiles") or ""
        label = names.get(smi) or smi[:32]
        for model in MODELS:
            if model not in (mol.get("models") or {}):
                continue
            pid = f"{model}:{label}"
            if pid in used:
                pid = f"{model}:{smi[:40]}"
            used.add(pid)
            params.append(pytest.param(smi, model, id=pid))
    return params


def test_descriptor_smiles_suite_size():
    smiles = load_descriptor_smiles()
    assert smiles, "missing tests/fixtures/descriptor_smiles.json"
    assert 100 <= len(smiles) <= 400
    assert len(set(smiles)) == len(smiles)
    assert "CC(=O)Oc1ccccc1C(=O)O" in smiles
    assert "CCCC1CCCNC1C=O" in smiles
    for s in smiles:
        _, mol = parse_smiles(s)
        assert mol.atoms.num >= 2


def test_descriptor_smiles_chemistry_coverage():
    """Suite is drug-sized and spans heteroatoms, fusion, and charge — not fragments."""
    from statistics import median

    from rdkit import Chem

    smiles = load_descriptor_smiles()
    mols = []
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        assert m is not None, s
        mols.append(m)
    heavy = [m.GetNumHeavyAtoms() for m in mols]
    assert median(heavy) >= 15
    assert max(heavy) >= 30
    assert sum(n >= 20 for n in heavy) >= 40
    zs = {z for m in mols for z in {a.GetAtomicNum() for a in m.GetAtoms()}}
    for z, name in ((9, "F"), (17, "Cl"), (35, "Br"), (53, "I"), (15, "P"), (16, "S")):
        assert z in zs, f"suite missing {name}"
    fused = charged = four_rings = 0
    for m in mols:
        rings = m.GetRingInfo().AtomRings()
        if len(rings) >= 4:
            four_rings += 1
        if any(set(rings[i]) & set(rings[j]) for i in range(len(rings)) for j in range(i + 1, len(rings))):
            fused += 1
        if any(a.GetFormalCharge() for a in m.GetAtoms()):
            charged += 1
    assert fused >= 20
    assert charged >= 4
    assert four_rings >= 3


def test_ob_dumps_present():
    from xenosite.predict.molecule import canonicalize_smiles

    dumps = _dumps()
    assert dumps, f"missing {OB_DUMPS_GZ} (git lfs pull, or run make dump-ob)"
    smiles = {d["smiles"] for d in dumps}
    assert "CC(=O)Oc1ccccc1C(=O)O" in smiles
    assert "CCCC1CCCNC1C=O" in smiles
    want = {canonicalize_smiles(s) for s in load_descriptor_smiles()}
    missing = want - smiles
    assert not missing, (
        f"ob_dumps missing {len(missing)} descriptor SMILES (run make dump-ob); "
        f"e.g. {next(iter(missing))}"
    )


def _omp_dump_mismatch(columns: list[str]) -> bool:
    """True when every mismatch is an ortho/meta/para column (set-order artifact)."""
    if not columns:
        return False
    return all(
        "Ortho_" in c or "Meta_" in c or "Para_" in c
        for c in columns
        if not c.startswith("_")
    )


@pytest.mark.parametrize("smiles,model", _dump_params())
def test_internal_ob_vs_dump(smiles, model):
    dump_mol = next(d for d in _dumps() if d.get("smiles") == smiles)
    payload = dump_mol["models"][model]
    mol, _ = parse_smiles(smiles)
    mm = compare_feature_dump_rows(
        rows_for_model(model, mol),
        payload,
        skip_columns=dump_compare_skip_columns(model),
    )
    if _omp_dump_mismatch(mm):
        pytest.xfail(
            "ortho/meta/para uses any shortest path on an aromatic ring; "
            "the 2.4 dump picked one BFS path via Python 2 set order "
            f"({', '.join(mm)}). See docs/vendored-diffs.md."
        )
    assert not mm, (
        f"internal OpenBabel vs dump mismatch for {model} {smiles} "
        f"(do not loosen atol): " + ", ".join(mm)
    )


def test_ndealk_ob_dump_has_net_columns():
    dump = load_ob_dump()
    assert dump, f"missing {OB_ASPIRIN} (run make dump-ob)"
    cols = dump["models"]["ndealk"]["columns"]
    assert len(cols) == 386
    assert cols[0] == "otherN_C"


def test_ob_dump_aligns_epoxidation_bonds():
    dump = load_ob_dump()
    assert dump, f"missing {OB_ASPIRIN} (run make dump-ob)"
    from xenosite.predict.features import bond_rows
    from tests.support import _row_for_ob_index

    mol, _ = parse_smiles(dump["smiles"])
    rows = bond_rows(mol, original_atom_ordering=True)
    payload = dump["models"]["epoxidation"]
    assert len(payload["rows"]) == len(rows) == 13
    for ix in payload["index"]:
        assert _row_for_ob_index(rows, ix) is not None
