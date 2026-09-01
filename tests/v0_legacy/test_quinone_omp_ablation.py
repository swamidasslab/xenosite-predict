"""Ablation tests for ``quinone_omp_mode`` semantics and score attribution.

Quinone legacy-vs-default drift is **entirely** ``quinone_omp_mode``: legacy picks
one BFS shortest path for ortho/meta/para ring indicators; principled averages
all tied shortest paths. No other ``_parameter`` flag affects quinone scores.

See ``docs/legacy-vs-principled.md`` chapter 1 and ``study_fix_attribution.py``.
"""

from __future__ import annotations

import numpy as np
import pytest

from xenosite.predict import predict
from xenosite.predict.backends.onnx import OnnxBackend
from xenosite.predict.features import load_names
from xenosite.predict.molecule import parse_smiles

from tests.support import (
    GOLDEN_PARAMETER,
    PARITY_ATOL,
    ROOT,
    golden_score_fields,
    load_golden_suite,
    onnx_weights_present,
    rows_for_model,
    onnx_root,
)

BACKEND = OnnxBackend(onnx_root())

NAPHTHALENE = "c1ccc2ccccc2c1"
POLYPHENOL = "O=c1c(O)c(-c2cc(O)c(O)c(O)c2)oc2cc(O)cc(O)c12"
WORST_LEGACY_GAP = "O=C1C(O)=C2C(=O)c3cccc(O)c3C(=O)C2(O)C(O)=C1C1CCCCC1"

_OMP_DERIVED_PREFIXES = ("Ortho_", "Meta_", "Para_", "Site_Ortho_", "Site_Meta_", "Site_Para_")


def _max_score_delta(a: dict, b: dict) -> float:
    delta = 0.0
    for key in ("mol", "atom", "bond", "pair"):
        va, vb = a.get(key), b.get(key)
        if va is None or vb is None:
            continue
        if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
            delta = max(delta, abs(float(va) - float(vb)))
        elif isinstance(va, list) and isinstance(vb, list):
            for x, y in zip(va, vb):
                delta = max(delta, abs(float(x) - float(y)))
    return delta


def _quinone_scores(smiles: str, *, parameter: dict | None = None) -> dict:
    if not onnx_weights_present("quinone"):
        pytest.skip("no ONNX weights for quinone")
    kwargs: dict = {"models": ["quinone"], "backend": BACKEND}
    if parameter is not None:
        kwargs["_parameter"] = parameter
    mol = predict(smiles, **kwargs)
    assert mol.results
    return golden_score_fields(mol.results[0])


def _quinone_smiles_from_golden() -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for row in load_golden_suite():
        if row.get("model") != "quinone":
            continue
        smi = row.get("smiles")
        if not smi or smi in seen:
            continue
        seen.add(smi)
        out.append(smi)
    return out


def _omp_derived_column(name: str) -> bool:
    return name.startswith(_OMP_DERIVED_PREFIXES)


def _assert_only_omp_derived_columns_differ(
    legacy_rows: list[dict],
    principled_rows: list[dict],
    names: list[str],
    *,
    require_diff: bool,
) -> None:
    from xenosite.predict.features.names import select_columns

    assert len(legacy_rows) == len(principled_rows)
    saw_diff = False
    for leg, pr in zip(legacy_rows, principled_rows):
        l_vec = select_columns(leg, names)
        p_vec = select_columns(pr, names)
        diff_cols = [
            names[i]
            for i in range(len(names))
            if not np.isclose(l_vec[i], p_vec[i], atol=0.0, rtol=0.0)
        ]
        if diff_cols:
            saw_diff = True
            assert all(_omp_derived_column(c) for c in diff_cols), (
                f"non-OMP columns drifted: {diff_cols}"
            )
    if require_diff:
        assert saw_diff


def test_quinone_omp_principled_from_golden_equals_production_default() -> None:
    """``GOLDEN + quinone_omp_mode=principled`` is bitwise-identical to ``predict()`` default."""
    default = _quinone_scores(NAPHTHALENE)
    omp_only = _quinone_scores(
        NAPHTHALENE,
        parameter={**GOLDEN_PARAMETER, "quinone_omp_mode": "principled"},
    )
    assert _max_score_delta(default, omp_only) <= PARITY_ATOL


def test_quinone_omp_flag_alone_explains_legacy_to_default_drift() -> None:
    """On naphthalene and a large polyphenol outlier, only OMP mode moves scores."""
    for smiles in (NAPHTHALENE, POLYPHENOL):
        default = _quinone_scores(smiles)
        legacy = _quinone_scores(smiles, parameter=GOLDEN_PARAMETER)
        omp_only = _quinone_scores(
            smiles,
            parameter={**GOLDEN_PARAMETER, "quinone_omp_mode": "principled"},
        )
        full = _max_score_delta(default, legacy)
        partial = _max_score_delta(omp_only, legacy)
        assert full > PARITY_ATOL
        assert abs(full - partial) <= max(PARITY_ATOL, 0.05 * full)


def test_quinone_other_golden_flags_are_inert() -> None:
    """Symmetry, ndealk site, and bond NRings do not touch quinone."""
    only_omp_legacy = _quinone_scores(NAPHTHALENE, parameter={"quinone_omp_mode": "legacy"})
    full_golden = _quinone_scores(NAPHTHALENE, parameter=GOLDEN_PARAMETER)
    assert _max_score_delta(only_omp_legacy, full_golden) <= PARITY_ATOL


def test_quinone_descriptor_ablation_only_omp_derived_columns_differ() -> None:
    """Atom ONNX inputs change only on OMP-derived ortho/meta/para and Site_* columns."""
    rdmol, _ = parse_smiles(NAPHTHALENE)
    legacy = rows_for_model("quinone", rdmol, _parameter={"quinone_omp_mode": "legacy"})
    principled = rows_for_model("quinone", rdmol, _parameter={"quinone_omp_mode": "principled"})
    names = load_names("quinone", "atom")
    assert names
    _assert_only_omp_derived_columns_differ(
        legacy, principled, names, require_diff=True
    )


def test_quinone_worst_legacy_gap_still_omp_only() -> None:
    """Largest remaining legacy gap is still OMP-only (not weights or other flags)."""
    default = _quinone_scores(WORST_LEGACY_GAP)
    legacy = _quinone_scores(WORST_LEGACY_GAP, parameter=GOLDEN_PARAMETER)
    omp_only = _quinone_scores(
        WORST_LEGACY_GAP,
        parameter={**GOLDEN_PARAMETER, "quinone_omp_mode": "principled"},
    )
    full = _max_score_delta(default, legacy)
    assert full > PARITY_ATOL
    assert _max_score_delta(omp_only, default) <= PARITY_ATOL
    assert abs(full - _max_score_delta(omp_only, legacy)) <= max(PARITY_ATOL, 0.05 * full)


def test_quinone_polyphenol_much_closer_under_max_omp() -> None:
    """Polyphenol outlier: max-OMP shrinks legacy gap vs old mean-OMP (~0.8 → ~0.08)."""
    default = _quinone_scores(POLYPHENOL)
    legacy = _quinone_scores(POLYPHENOL, parameter=GOLDEN_PARAMETER)
    mean_mode = _quinone_scores(POLYPHENOL, parameter={"quinone_omp_mode": "mean"})
    assert _max_score_delta(default, legacy) < 0.15
    assert _max_score_delta(mean_mode, legacy) > 0.5


@pytest.mark.full
def test_quinone_omp_ablation_explains_drift_on_golden_suite() -> None:
    """Principled max-OMP matches legacy on ~90% of golden quinone molecules (atol 5e-3)."""
    smiles_list = _quinone_smiles_from_golden()
    assert smiles_list
    within = 0
    for smiles in smiles_list:
        default = _quinone_scores(smiles)
        legacy = _quinone_scores(smiles, parameter=GOLDEN_PARAMETER)
        if _max_score_delta(default, legacy) <= PARITY_ATOL:
            within += 1
    assert within / len(smiles_list) >= 0.85
