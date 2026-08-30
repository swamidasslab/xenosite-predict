"""Importable test helpers (conftest fixtures stay in conftest.py)."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = Path(__file__).resolve().parent / "fixtures" / "golden_smiles.json"
OB_ASPIRIN = Path(__file__).resolve().parent / "fixtures" / "ob_dump_aspirin.json"
OB_DUMPS = Path(__file__).resolve().parent / "fixtures" / "ob_dumps.json"
MODELS = ("epoxidation", "quinone", "reactivity", "ugt", "ndealk")


def onnx_weights_present(model: str | None = None) -> bool:
    root = ROOT / "weights" / "onnx"
    if model:
        return any((root / model).glob("*.onnx"))
    return any(root.rglob("*.onnx"))


def load_golden():
    if not GOLDEN.is_file():
        return []
    return json.loads(GOLDEN.read_text(encoding="utf-8"))


def load_ob_dumps() -> list[dict]:
    """All OpenBabel dumps (golden SMILES + extras). Empty if the suite is missing."""
    if OB_DUMPS.is_file():
        data = json.loads(OB_DUMPS.read_text(encoding="utf-8"))
        mols = data.get("molecules") if isinstance(data, dict) else data
        return list(mols or [])
    dump = load_ob_dump()
    return [dump] if dump else []


def rows_for_model(model: str, mol) -> list[dict]:
    from xenosite.predict.features import (
        bond_rows,
        ndealk_bond_rows,
        quinone_atom_rows,
        reactivity_atom_rows,
        ugt_atom_rows,
    )

    if model == "epoxidation":
        return bond_rows(mol, original_atom_ordering=True)
    if model == "ndealk":
        return ndealk_bond_rows(mol)
    if model == "ugt":
        return ugt_atom_rows(mol)
    if model == "quinone":
        return quinone_atom_rows(mol)
    if model == "reactivity":
        return reactivity_atom_rows(mol)
    raise KeyError(model)


def load_ob_dump(path: Path | None = None) -> dict:
    p = path or OB_ASPIRIN
    if not p.is_file():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def openbabel_available() -> bool:
    from xenosite.predict.features import _ob

    return _ob.available()


def _row_for_ob_index(rows: list[dict], index: str) -> dict | None:
    """Match a dump index to an inference row.

    BondTD: ``{mol}.{a1}.{a2}`` (1-based OpenBabel atom idx).
    AtomTD: ``{mol}.{group}.{atom}``. UGT: ``{mol}.{atom}``.
    """
    parts = str(index).split(".")
    by_bond = {frozenset(r["_atoms"]): r for r in rows if "_atoms" in r}
    by_atom = {int(r["_atom"]): r for r in rows if "_atom" in r}
    if by_bond and len(parts) >= 3:
        a1, a2 = int(parts[-2]) - 1, int(parts[-1]) - 1
        hit = by_bond.get(frozenset((a1, a2)))
        if hit is not None:
            return hit
    if by_atom:
        return by_atom.get(int(parts[-1]) - 1)
    return None


def compare_feature_dump_rows(
    rows: list[dict], dump: dict, *, atol: float = 1e-4
) -> list[str]:
    """Return unique overlapping column names that disagree (atol 1e-4, rtol=0).

    Aligns by OpenBabel index when possible. Unaligned dump rows are
    ``_unaligned``. Do not loosen atol.
    """
    import numpy as np

    cols = dump.get("columns") or []
    ob_rows = dump.get("rows") or []
    index = dump.get("index") or []
    mismatches: list[str] = []
    seen: set[str] = set()
    unaligned = 0
    for i, ob in enumerate(ob_rows):
        got = None
        if i < len(index):
            got = _row_for_ob_index(rows, index[i])
        if got is None and i < len(rows):
            got = rows[i]
        if got is None:
            unaligned += 1
            continue
        ob_map = dict(zip(cols, ob))
        for c, val in ob_map.items():
            if c not in got or c in seen:
                continue
            try:
                if not np.isclose(float(val), float(got[c]), atol=atol, rtol=0):
                    mismatches.append(c)
                    seen.add(c)
            except (TypeError, ValueError):
                continue
    if unaligned:
        mismatches.append(f"_unaligned:{unaligned}")
    return mismatches


# Back-compat alias used by older live-test drafts.
compare_rdkit_ob_rows = compare_feature_dump_rows
_rdkit_row_for_ob_index = _row_for_ob_index
