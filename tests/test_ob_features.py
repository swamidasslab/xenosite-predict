"""Internal OpenBabel features vs dump-oracle rows.

Dumps come from ``make dump-ob`` (Debian OpenBabel 2.4). Skip without dumps
or without OpenBabel on the host. Do not loosen atol.
"""

from __future__ import annotations

import pytest

from xenosite.predict.molecule import parse_smiles

from tests.support import (
    MODELS,
    OB_ASPIRIN,
    OB_DUMPS,
    compare_feature_dump_rows,
    load_ob_dump,
    load_ob_dumps,
    openbabel_available,
    rows_for_model,
)

need_ob = pytest.mark.skipif(
    not openbabel_available(), reason="OpenBabel 2.4 not installed"
)

_CASES: list[tuple[str, str]] = []
for _mol in load_ob_dumps():
    smi = _mol.get("smiles") or ""
    for _model in MODELS:
        if _model in (_mol.get("models") or {}):
            _CASES.append((smi, _model))


def test_ob_dump_suite_covers_golden_and_extra():
    dumps = load_ob_dumps()
    if not dumps:
        pytest.skip(f"missing {OB_DUMPS} (run make dump-ob)")
    smiles = {d["smiles"] for d in dumps}
    assert "CC(=O)Oc1ccccc1C(=O)O" in smiles
    assert "CCCC1CCCNC1C=O" in smiles
    assert len(smiles) >= 14


@need_ob
@pytest.mark.parametrize(
    "smiles,model",
    _CASES or [pytest.param("", "", marks=pytest.mark.skip(reason="no OB dumps"))],
)
def test_internal_ob_vs_dump(smiles, model):
    dump_mol = next(d for d in load_ob_dumps() if d.get("smiles") == smiles)
    payload = dump_mol["models"][model]
    mol, _ = parse_smiles(smiles)
    mm = compare_feature_dump_rows(rows_for_model(model, mol), payload)
    assert not mm, (
        f"internal OpenBabel vs dump mismatch for {model} {smiles} "
        f"(do not loosen atol): " + ", ".join(mm)
    )


def test_ndealk_ob_dump_has_net_columns():
    dump = load_ob_dump()
    if not dump:
        pytest.skip(f"missing {OB_ASPIRIN} (run make dump-ob)")
    cols = dump["models"]["ndealk"]["columns"]
    assert len(cols) == 386
    assert cols[0] == "otherN_C"


@need_ob
def test_ob_dump_aligns_epoxidation_bonds():
    dump = load_ob_dump()
    if not dump:
        pytest.skip(f"missing {OB_ASPIRIN} (run make dump-ob)")
    from xenosite.predict.features import bond_rows
    from tests.support import _row_for_ob_index

    mol, _ = parse_smiles(dump["smiles"])
    rows = bond_rows(mol, original_atom_ordering=True)
    payload = dump["models"]["epoxidation"]
    assert len(payload["rows"]) == len(rows) == 13
    for ix in payload["index"]:
        assert _row_for_ob_index(rows, ix) is not None
