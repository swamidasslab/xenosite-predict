"""Importable test helpers (conftest fixtures stay in conftest.py)."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = Path(__file__).resolve().parent / "fixtures" / "golden_smiles.json"
GOLDEN_SUITE = Path(__file__).resolve().parent / "fixtures" / "golden_descriptor_suite.json"
DESCRIPTOR_SMILES = Path(__file__).resolve().parent / "fixtures" / "descriptor_smiles.json"
OB_ASPIRIN = Path(__file__).resolve().parent / "fixtures" / "ob_dump_aspirin.json"
OB_DUMPS = Path(__file__).resolve().parent / "fixtures" / "ob_dumps.json"
OB_DUMPS_GZ = OB_DUMPS.with_name(OB_DUMPS.name + ".gz")
ASPIRIN_SMILES = "CC(=O)Oc1ccccc1C(=O)O"
MODELS = ("epoxidation", "quinone", "reactivity", "ugt", "ndealk", "phase1")
SUITE_MODELS = ("epoxidation", "quinone", "reactivity", "ugt", "ndealk", "isozyme", "phase1")
PARITY_ATOL = 3e-4  # TF1 float32 vs ORT on quinone near-zero atom scores
PARITY_ATOL_OMP = 0.02  # quinone OMP descriptor drift (mol head can lag ~0.02)


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


def load_golden_suite(*, merge_smoke: bool = True) -> list[dict]:
    """327-molecule suite golden rows; optionally include ``golden_smiles.json``."""
    rows: list[dict] = []
    if GOLDEN_SUITE.is_file():
        rows.extend(json.loads(GOLDEN_SUITE.read_text(encoding="utf-8")))
    if merge_smoke and GOLDEN.is_file():
        seen = {(r.get("model"), r.get("smiles")) for r in rows}
        for r in load_golden():
            key = (r.get("model"), r.get("smiles"))
            if key not in seen:
                rows.append(r)
    return rows


def parity_atol(smiles: str, model: str) -> float:
    """Score compare tolerance: looser for quinone OMP-only descriptor drift."""
    if model == "quinone" and descriptor_omp_only(smiles, "quinone"):
        return PARITY_ATOL_OMP
    return PARITY_ATOL


def serialize_molecule_results(mol) -> list[dict]:
    """Golden-row ``results`` list from a :class:`Molecule` after ``predict``."""
    out = []
    for r in mol.results:
        rec = {"model": r.model, "version": r.version}
        for key in ("mol", "atom", "bond", "pair", "pair_idx"):
            val = getattr(r, key, None)
            if val is not None:
                rec[key] = _jsonish(val)
        out.append(rec)
    return out


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


PARITY_MODELS = ("epoxidation", "quinone", "reactivity", "ugt", "ndealk", "isozyme")
ASPIRIN_SMILES_CANON = ASPIRIN_SMILES  # CC(=O)Oc1ccccc1C(=O)O

_DESCRIPTOR_CACHE: dict[tuple[str, str], list[str]] | None = None


def onnx_model_key(model: str) -> str:
    """Registry model name → ONNX weights directory."""
    return "ndealk" if model == "isozyme" else model


def descriptor_mismatch_columns(smiles: str, model: str) -> list[str]:
    """Overlapping dump columns that disagree with internal OpenBabel (no ``_`` meta)."""
    from xenosite.predict.molecule import parse_smiles

    dump_mol = next(
        (d for d in load_ob_dumps() if d.get("smiles") == smiles),
        None,
    )
    if dump_mol is None or model not in (dump_mol.get("models") or {}):
        return ["_missing_dump"]
    mol, _ = parse_smiles(smiles)
    mm = compare_feature_dump_rows(
        rows_for_model(model, mol),
        dump_mol["models"][model],
        skip_columns=dump_compare_skip_columns(model),
    )
    return [c for c in mm if not c.startswith("_")]


def descriptor_passes(smiles: str, model: str) -> bool:
    return not descriptor_mismatch_columns(smiles, model)


def descriptor_omp_only(smiles: str, model: str) -> bool:
    cols = descriptor_mismatch_columns(smiles, model)
    if not cols:
        return False
    return all("Ortho_" in c or "Meta_" in c or "Para_" in c for c in cols)


def prediction_score_fields(obj) -> dict:
    """Score vectors for legacy vs ONNX parity (includes quinone pair fields)."""
    keys = ("mol", "bond", "atom", "pair", "pair_idx")
    if isinstance(obj, dict):
        return {k: _jsonish(obj[k]) for k in keys if obj.get(k) is not None}
    out = {}
    for k in keys:
        v = getattr(obj, k, None)
        if v is not None:
            out[k] = _jsonish(v)
    return out


def assert_predictions_parity(
    legacy_mol,
    onnx_mol,
    *,
    atol: float | None = None,
    smiles: str = "",
    model: str = "",
) -> None:
    """Assert legacy-test-api and ONNX ``predict`` agree on every result head."""
    from xenosite.predict.compare import assert_equiv_results

    if atol is None:
        atol = parity_atol(smiles, model)
    by_legacy = {r.model: r for r in legacy_mol.results}
    by_onnx = {r.model: r for r in onnx_mol.results}
    assert set(by_legacy) == set(by_onnx), (
        f"result heads differ: legacy={sorted(by_legacy)} onnx={sorted(by_onnx)}"
    )
    for name in sorted(by_legacy):
        want = prediction_score_fields(by_legacy[name])
        have = prediction_score_fields(by_onnx[name])
        assert want, f"no score fields on legacy result {name}"
        assert have, f"no score fields on onnx result {name}"
        assert_equiv_results(want, have, atol=atol)


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


def golden_score_fields(obj) -> dict:
    """Score vectors for golden vs ONNX (mol/bond/atom/pair/pair_idx)."""
    keys = ("mol", "bond", "atom", "pair", "pair_idx")
    if isinstance(obj, dict):
        return {k: _jsonish(obj[k]) for k in keys if obj.get(k) is not None}
    out = {}
    for k in keys:
        v = getattr(obj, k, None)
        if v is not None:
            out[k] = _jsonish(v)
    return out


def score_fields(obj) -> dict:
    """Backward-compatible subset (mol/bond/atom only)."""
    full = golden_score_fields(obj)
    return {k: full[k] for k in ("mol", "bond", "atom") if k in full}


def assert_golden_molecule(
    got,
    golden_row,
    *,
    atol: float | None = None,
    smiles: str | None = None,
    model: str | None = None,
) -> None:
    """Compare every golden head to the matching ``got.results`` entry."""
    from xenosite.predict.compare import assert_equiv_results

    smiles = smiles or golden_row.get("smiles") or ""
    model = model or golden_row.get("model") or ""
    if atol is None:
        atol = parity_atol(smiles, model)
    golden_results = golden_row.get("results") or []
    assert golden_results, "golden row has no results"
    by_model = {r.model: r for r in got.results}
    for g in golden_results:
        name = g.get("model")
        assert name in by_model, f"missing result {name}; got {sorted(by_model)}"
        want = golden_score_fields(g)
        have = golden_score_fields(by_model[name])
        if name == "quinone" and smiles:
            from xenosite.predict.numbering import normalize_quinone_pair_fields

            normalize_quinone_pair_fields(want, have, smiles)
        subset = {k: want[k] for k in want if k in have and have.get(k) is not None}
        got_subset = {k: have[k] for k in subset}
        assert subset, (
            f"no overlapping score fields for {name} "
            f"(golden keys: {sorted(want)}; got: {sorted(have)})"
        )
        assert_equiv_results(subset, got_subset, atol=atol)


# Back-compat alias used by older live-test drafts.
compare_rdkit_ob_rows = compare_feature_dump_rows
_rdkit_row_for_ob_index = _row_for_ob_index
