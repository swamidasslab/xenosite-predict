"""Importable test helpers (conftest fixtures stay in conftest.py)."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = Path(__file__).resolve().parent / "fixtures" / "golden_smiles.json"
DESCRIPTOR_SMILES = Path(__file__).resolve().parent / "fixtures" / "descriptor_smiles.json"
OB_ASPIRIN = Path(__file__).resolve().parent / "fixtures" / "ob_dump_aspirin.json"
OB_DUMPS = Path(__file__).resolve().parent / "fixtures" / "ob_dumps.json"
OB_DUMPS_GZ = OB_DUMPS.with_name(OB_DUMPS.name + ".gz")
ASPIRIN_SMILES = "CC(=O)Oc1ccccc1C(=O)O"
MODELS = ("epoxidation", "quinone", "reactivity", "ugt", "ndealk", "phase1")


def onnx_weights_present(model: str | None = None) -> bool:
    root = ROOT / "weights" / "onnx"
    if model:
        return any((root / model).glob("*.onnx"))
    return any(p for p in root.rglob("*.onnx") if "_dump" not in p.parts)


def list_onnx_heads() -> list[tuple[str, str]]:
    """``(model, head)`` pairs from ``weights/onnx``, skipping convert dump dirs."""
    root = ROOT / "weights" / "onnx"
    if not root.is_dir():
        return []
    found: list[tuple[str, str]] = []
    for model_dir in sorted(p for p in root.iterdir() if p.is_dir() and p.name != "_dump"):
        for onnx in sorted(model_dir.glob("*.onnx")):
            found.append((model_dir.name, onnx.stem))
    return found


def onnx_io_dims(model: str, head: str) -> tuple[int, int] | None:
    """Return ``(n_features, n_outputs)`` from convert metadata or the ONNX graph."""
    meta = ROOT / "weights" / "onnx" / model / f"{head}.meta.json"
    if meta.is_file():
        data = json.loads(meta.read_text(encoding="utf-8"))
        n_in, n_out = data.get("I"), data.get("O")
        if n_in is not None and n_out is not None:
            return int(n_in), int(n_out)
    path = ROOT / "weights" / "onnx" / model / f"{head}.onnx"
    if not path.is_file():
        return None
    from xenosite.predict.backends.onnx import OnnxBackend

    sess = OnnxBackend(ROOT / "weights" / "onnx").session(model, head)
    inp = sess.get_inputs()[0].shape
    out = sess.get_outputs()[0].shape
    n_in = inp[1] if len(inp) > 1 and isinstance(inp[1], int) else None
    n_out = 1
    if len(out) > 1 and isinstance(out[1], int):
        n_out = out[1]
    elif isinstance(out[0], int):
        n_out = out[0]
    if n_in is None:
        return None
    return int(n_in), int(n_out)


def load_descriptor_smiles() -> list[str]:
    if not DESCRIPTOR_SMILES.is_file():
        return []
    payload = json.loads(DESCRIPTOR_SMILES.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("smiles") or payload.get("molecules") or []
    return [str(s) for s in payload if s]


def load_golden():
    if not GOLDEN.is_file():
        return []
    return json.loads(GOLDEN.read_text(encoding="utf-8"))


def _load_json(path: Path):
    probe = path.read_bytes()[:80]
    if probe.startswith(b"version https://git-lfs.github.com/spec/v1"):
        raise RuntimeError(
            f"{path} is a Git LFS pointer; install git-lfs and run `git lfs pull`"
        )
    if path.name.endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            return json.load(fh)
    return json.loads(path.read_text(encoding="utf-8"))


def _suite_path() -> Path | None:
    if OB_DUMPS.is_file():
        return OB_DUMPS
    if OB_DUMPS_GZ.is_file():
        return OB_DUMPS_GZ
    return None


def _suite_molecules() -> list[dict]:
    path = _suite_path()
    if path is None:
        return []
    data = _load_json(path)
    mols = data.get("molecules") if isinstance(data, dict) else data
    return list(mols or [])


def load_ob_dumps() -> list[dict]:
    """All OpenBabel dumps (golden SMILES + extras). Empty if the suite is missing.

    Prefers uncompressed ``ob_dumps.json`` (fresh ``make dump-ob``), else the
    committed ``ob_dumps.json.gz`` (Git LFS).
    """
    mols = _suite_molecules()
    if mols:
        return mols
    dump = load_ob_dump()
    return [dump] if dump else []


def rows_for_model(model: str, mol) -> list[dict]:
    from xenosite.predict.features import (
        bond_rows,
        ndealk_bond_rows,
        phase1_rows,
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
    if model == "phase1":
        return phase1_rows(mol)
    raise KeyError(model)


def dump_compare_skip_columns(model: str) -> frozenset[str] | None:
    """Columns present in py2 dumps but not in our ported feature rows."""
    if model == "phase1":
        from xenosite.predict.features.bond_lonepair import POSSIBLE_SITE_COLUMNS

        return POSSIBLE_SITE_COLUMNS
    return None


def load_ob_dump(path: Path | None = None) -> dict:
    p = path or OB_ASPIRIN
    if p.is_file():
        return _load_json(p)
    if path is not None:
        return {}
    for mol in _suite_molecules():
        if mol.get("smiles") == ASPIRIN_SMILES:
            return mol
    return {}


def openbabel_available() -> bool:
    from xenosite.predict.features import _ob

    return _ob.available()


def _row_for_ob_index(rows: list[dict], index: str) -> dict | None:
    """Match a dump index to an inference row.

    BondTD: ``{mol}.{a1}.{a2}`` (1-based OpenBabel atom idx).
    AtomTD: ``{mol}.{group}.{atom}``. UGT: ``{mol}.{atom}``.
    """
    by_index = {str(r["_index"]): r for r in rows if "_index" in r}
    if index in by_index:
        return by_index[index]
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
    rows: list[dict],
    dump: dict,
    *,
    atol: float = 1e-4,
    skip_columns: set[str] | frozenset[str] | None = None,
) -> list[str]:
    """Return unique overlapping column names that disagree (rtol=0).

    Default atol is 1e-4. Do not loosen columns to hide OpenBabel 3.x drift.
    ``skip_columns`` are omitted (e.g. phase1 Possible_Sites in the py2 dump).
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
            if skip_columns and c in skip_columns:
                continue
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


def golden_name_by_smiles() -> dict[str, str]:
    return {
        g["smiles"]: str(g.get("name") or g["smiles"][:32])
        for g in load_golden()
        if g.get("smiles")
    }


def _jsonish(v):
    if v is None:
        return None
    if isinstance(v, (list, tuple)):
        return [_jsonish(x) for x in v]
    if isinstance(v, str):
        return v
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return float(v)
    if hasattr(v, "tolist"):
        return _jsonish(v.tolist())
    if hasattr(v, "item"):
        return _jsonish(v.item())
    return v


def score_fields(obj) -> dict:
    """``mol`` / ``bond`` / ``atom`` from a Result or a golden-result dict."""
    if isinstance(obj, dict):
        return {k: obj.get(k) for k in ("mol", "bond", "atom")}
    return {
        "mol": getattr(obj, "mol", None),
        "bond": getattr(obj, "bond", None),
        "atom": getattr(obj, "atom", None),
    }


def assert_golden_molecule(got, golden_row) -> None:
    """Compare every golden head to the matching ``got.results`` entry (atol 1e-4)."""
    from xenosite.predict.compare import assert_equiv_results

    golden_results = golden_row.get("results") or []
    assert golden_results, "golden row has no results"
    by_model = {r.model: r for r in got.results}
    for g in golden_results:
        name = g.get("model")
        assert name in by_model, f"missing result {name}; got {sorted(by_model)}"
        want = score_fields(g)
        have = score_fields(by_model[name])
        subset = {}
        got_subset = {}
        for k in ("mol", "bond", "atom"):
            if want.get(k) is not None and have.get(k) is not None:
                subset[k] = _jsonish(want[k])
                got_subset[k] = _jsonish(have[k])
        assert subset, (
            f"no overlapping score fields for {name} "
            f"(golden mol/bond/atom present: "
            f"{ {k: want.get(k) is not None for k in ('mol', 'bond', 'atom')} }; "
            f"got: { {k: have.get(k) is not None for k in ('mol', 'bond', 'atom')} })"
        )
        assert_equiv_results(subset, got_subset)


# Back-compat alias used by older live-test drafts.
compare_rdkit_ob_rows = compare_feature_dump_rows
_rdkit_row_for_ob_index = _row_for_ob_index
