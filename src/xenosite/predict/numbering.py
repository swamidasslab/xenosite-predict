"""Heavy-atom numbering policy for legacy OpenBabel score keys and pattern ids.

Graph construction follows legacy (including explicit ``[nH]`` in SMILES). All
hydrogens — implicit or explicit — are **excluded from numbering**. API and
RDKit score vectors use dense 0-based atom indices.

Legacy OB 2.4 can emit **gapped** 1-based ``GetIdx()`` keys on some ``[nH]``
molecules (e.g. ``1..17, 19..23``). Those keys are mapped to RDKit by **row
order** (sorted legacy ids → position → ``_atom``), not by treating the key as
a dense ``1..N`` index.

Set ``XENOSITE_OB_NUMBERING``:

- ``dense`` (default) — pattern-id third field is dense ``1..N`` heavy atoms.
- ``raw`` — preserve raw OB ``GetIdx()`` in pattern ids (legacy dump parity).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from typing import Any, Mapping, Sequence


def legacy_atom_lookup(site_map: dict) -> dict[int, float]:
    """Normalize legacy atom/site maps keyed by string or int atom ids."""
    out: dict[int, float] = {}
    for k, v in (site_map or {}).items():
        try:
            key = int(k)
        except (TypeError, ValueError):
            continue
        if isinstance(v, dict):
            v = next(iter(v.values()), 0.0)
        out[key] = float(v or 0.0)
    return out


class ObNumberingMode(str, Enum):
    """How the third field of AtomTD ``_index`` / legacy score keys behave."""

    RAW = "raw"
    DENSE_HEAVY = "dense"


_ENV_VAR = "XENOSITE_OB_NUMBERING"
_DEFAULT_MODE = ObNumberingMode.DENSE_HEAVY


def numbering_mode() -> ObNumberingMode:
    """Configured numbering mode (default ``dense``)."""
    raw = os.environ.get(_ENV_VAR, _DEFAULT_MODE.value).strip().lower()
    if raw in ("raw", "legacy", "getidx"):
        return ObNumberingMode.RAW
    if raw in ("dense", "dense_heavy", "heavy"):
        return ObNumberingMode.DENSE_HEAVY
    return _DEFAULT_MODE


def heavy_atom_raw_indices(obmol: Any) -> list[int]:
    """1-based OB ``GetIdx()`` for heavy atoms in molecule iteration order."""
    from .features import _ob

    ob, _pybel = _ob.load()
    out: list[int] = []
    for atom in ob.OBMolAtomIter(obmol):
        if atom.IsHydrogen():
            continue
        out.append(int(atom.GetIdx()))
    return out


def raw_numbering_is_gapped(raw_indices: Sequence[int]) -> bool:
    """True when heavy-atom OB indices are not exactly ``1..N``."""
    if not raw_indices:
        return False
    n = len(raw_indices)
    if max(raw_indices) > n:
        return True
    return any(i not in raw_indices for i in range(1, max(raw_indices) + 1))


def detect_raw_numbering_mode(obmol: Any) -> ObNumberingMode:
    """Classify this molecule's OB ``GetIdx()`` layout (runtime detection)."""
    raw = heavy_atom_raw_indices(obmol)
    if raw_numbering_is_gapped(raw):
        return ObNumberingMode.RAW
    return ObNumberingMode.DENSE_HEAVY


def pattern_ob_index(raw_ob_idx: int, dense_idx: int, mode: ObNumberingMode | None = None) -> int:
    """Third component of ``mol.group.atom`` pattern ids."""
    mode = mode or numbering_mode()
    if mode == ObNumberingMode.DENSE_HEAVY:
        return int(dense_idx)
    return int(raw_ob_idx)


def parse_pattern_ob_index(index: str | int) -> int:
    return int(str(index).split(".")[-1])


def ordered_legacy_ob_keys(site_map: Mapping[Any, Any]) -> list[int]:
    """Sorted 1-based legacy score keys (heavy atoms only)."""
    lookup = legacy_atom_lookup(dict(site_map))
    return sorted(lookup.keys())


def legacy_keys_are_gapped(keys: Sequence[int], n_heavy: int) -> bool:
    return raw_numbering_is_gapped(keys)


def map_legacy_ob_to_rdkit(
    ob_id: int,
    legacy_ob_order: Sequence[int],
    n_heavy: int,
    *,
    already_zero_based: bool = False,
) -> int:
    """Map a legacy site / pair atom id to 0-based RDKit index."""
    if already_zero_based:
        ob_id = int(ob_id) + 1
    ob_id = int(ob_id)
    keys = [int(k) for k in legacy_ob_order]
    if legacy_keys_are_gapped(keys, n_heavy):
        try:
            return keys.index(ob_id)
        except ValueError:
            pass
    if 1 <= ob_id <= n_heavy:
        return ob_id - 1
    if 0 <= ob_id < n_heavy:
        return ob_id
    return max(0, ob_id - 1)


def legacy_site_to_atom_vector(
    site_map: Mapping[Any, Any],
    n_atoms: int,
    *,
    rdkit_indices: Sequence[int] | None = None,
) -> list[float]:
    """Legacy atom-head scores → dense 0-based RDKit vector."""
    lookup = legacy_atom_lookup(dict(site_map))
    if not lookup:
        return [0.0] * n_atoms
    keys = ordered_legacy_ob_keys(site_map)
    if legacy_keys_are_gapped(keys, n_atoms):
        vec = [0.0] * n_atoms
        for pos, ob_id in enumerate(keys):
            if rdkit_indices is not None and pos < len(rdkit_indices):
                rd = int(rdkit_indices[pos])
            else:
                rd = pos
            if 0 <= rd < n_atoms:
                vec[rd] = lookup[ob_id]
        return vec
    return [lookup.get(i + 1, 0.0) for i in range(n_atoms)]


def build_ob_to_rdkit_from_rows(rows: Sequence[Mapping[str, Any]]) -> dict[int, int]:
    """Map pattern-id OB component → 0-based RDKit ``_atom`` from feature rows."""
    out: dict[int, int] = {}
    for row in rows:
        if "_index" not in row or "_atom" not in row:
            continue
        ob = parse_pattern_ob_index(row["_index"])
        out[ob] = int(row["_atom"])
    return out


def legacy_ob_order_from_rows(rows: Sequence[Mapping[str, Any]]) -> list[int]:
    return [parse_pattern_ob_index(r["_index"]) for r in rows if "_index" in r]


def map_legacy_pair_to_rdkit(
    a: int,
    b: int,
    *,
    legacy_ob_order: Sequence[int],
    n_heavy: int,
    ob_to_rd: Mapping[int, int] | None = None,
    already_zero_based: bool = False,
) -> tuple[int, int]:
    """Map legacy pair atom ids to sorted 0-based RDKit indices."""
    if ob_to_rd and not legacy_keys_are_gapped(list(legacy_ob_order), n_heavy):
        ra = ob_to_rd.get(int(a) if not already_zero_based else int(a) + 1, map_legacy_ob_to_rdkit(a, legacy_ob_order, n_heavy, already_zero_based=already_zero_based))
        rb = ob_to_rd.get(int(b) if not already_zero_based else int(b) + 1, map_legacy_ob_to_rdkit(b, legacy_ob_order, n_heavy, already_zero_based=already_zero_based))
    else:
        ra = map_legacy_ob_to_rdkit(a, legacy_ob_order, n_heavy, already_zero_based=already_zero_based)
        rb = map_legacy_ob_to_rdkit(b, legacy_ob_order, n_heavy, already_zero_based=already_zero_based)
    return tuple(sorted((ra, rb)))


def quinone_pairs_to_rdkit(
    pair_idx: Sequence[Sequence[int]],
    pair: Sequence[float],
    *,
    legacy_ob_order: Sequence[int],
    n_heavy: int,
    ob_to_rd: Mapping[int, int] | None = None,
) -> dict[str, list]:
    """Normalize quinone ``pair_idx`` + ``pair`` to RDKit 0-based indices."""
    mapped: list[tuple[tuple[int, int], float]] = []
    for idxs, score in zip(pair_idx, pair):
        a, b = int(idxs[0]), int(idxs[1])
        key = map_legacy_pair_to_rdkit(
            a,
            b,
            legacy_ob_order=legacy_ob_order,
            n_heavy=n_heavy,
            ob_to_rd=ob_to_rd,
        )
        mapped.append((key, float(score)))
    mapped.sort()
    return {
        "pair_idx": [list(i) for i, _ in mapped],
        "pair": [x for _, x in mapped],
    }


def normalize_legacy_quinone_fields(
    fields: dict[str, Any],
    *,
    legacy_ob_order: Sequence[int],
    n_heavy: int,
    ob_to_rd: Mapping[int, int] | None = None,
) -> None:
    """In-place: map stored quinone pair indices to RDKit space."""
    if "pair_idx" not in fields or "pair" not in fields:
        return
    norm = quinone_pairs_to_rdkit(
        fields["pair_idx"],
        fields["pair"],
        legacy_ob_order=legacy_ob_order,
        n_heavy=n_heavy,
        ob_to_rd=ob_to_rd,
    )
    fields["pair_idx"] = norm["pair_idx"]
    fields["pair"] = norm["pair"]


@dataclass(frozen=True)
class QuinoneNormalizeContext:
    legacy_ob_order: tuple[int, ...]
    n_heavy: int
    ob_to_rd: tuple[tuple[int, int], ...]


@lru_cache(maxsize=512)
def quinone_normalize_context(smiles: str) -> QuinoneNormalizeContext:
    """Parse *smiles* once and reuse for quinone ``pair_idx`` normalization."""
    from .features import quinone_atom_rows
    from .molecule import parse_smiles

    mol, _ = parse_smiles(smiles)
    rows = quinone_atom_rows(mol)
    ob_to_rd = build_ob_to_rdkit_from_rows(rows)
    return QuinoneNormalizeContext(
        legacy_ob_order=tuple(legacy_ob_order_from_rows(rows)),
        n_heavy=mol.GetNumAtoms(),
        ob_to_rd=tuple(sorted(ob_to_rd.items())),
    )


def _apply_quinone_context(fields: dict[str, Any], ctx: QuinoneNormalizeContext) -> None:
    normalize_legacy_quinone_fields(
        fields,
        legacy_ob_order=ctx.legacy_ob_order,
        n_heavy=ctx.n_heavy,
        ob_to_rd=dict(ctx.ob_to_rd),
    )


def normalize_quinone_fields_for_smiles(fields: dict[str, Any], smiles: str) -> None:
    """Map quinone ``pair_idx`` in *fields* to 0-based RDKit for *smiles*."""
    _apply_quinone_context(fields, quinone_normalize_context(smiles))


def normalize_quinone_pair_fields(
    want: dict[str, Any],
    have: dict[str, Any],
    smiles: str,
) -> None:
    """Normalize golden and ONNX quinone fields with one molecule parse."""
    ctx = quinone_normalize_context(smiles)
    _apply_quinone_context(want, ctx)
    _apply_quinone_context(have, ctx)


def legacy_predictions_atom_vector(
    predictions: Sequence[Mapping[str, Any]],
    n_atoms: int,
    score_key: str,
) -> list[float]:
    """Build RDKit atom vector from legacy REST ``predictions`` rows."""
    site: dict[int, float] = {}
    for row in predictions:
        atom_id = row.get("atom_id")
        if atom_id in (None, "", "0", 0):
            continue
        try:
            k = int(atom_id)
        except (TypeError, ValueError):
            continue
        val = row.get(score_key)
        if val is None:
            continue
        site[k] = float(val)
    return legacy_site_to_atom_vector(site, n_atoms)


def collect_legacy_ob_ids_from_pairs(predictions: Sequence[Mapping[str, Any]]) -> list[int]:
    ids: set[int] = set()
    for row in predictions:
        for key in ("atom_1_id", "atom_2_id"):
            v = row.get(key)
            if v in (None, "", "0", 0):
                continue
            try:
                ids.add(int(v))
            except (TypeError, ValueError):
                continue
    return sorted(ids)
