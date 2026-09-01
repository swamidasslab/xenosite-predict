"""Educational guide: legacy vs principled ``predict()`` behavior.

This module is written for **human readers first**, pytest second. Each test is a
short chapter with inline commentary explaining *why* production defaults differ
from golden/legacy parity modes, not just *that* they differ.

Run it like any other test file::

    uv run pytest tests/test_legacy_vs_principled_guide.py -v

For a prose overview see ``docs/legacy-vs-principled.md``. For tight regressions
see ``tests/test_onnx_principled.py`` and golden parity in ``test_golden*.py``.

The four internal flags (all on ``molecule._parameter`` via ``_parameter=``):

``ndealk_site_mode``
    How BondTD rows collapse into ndealk/isozyme site keys before bond mapping.
``quinone_omp_mode``
    Whether quinone ortho/meta/para uses one BFS path (legacy) or all shortest
    paths averaged (principled).
``symmetry_group_mode``
    RDKit ``CanonicalRankAtoms`` bond classes + score pooling (production) vs
    OpenBabel GID classes without RDKit pooling (golden).
``bond_nrings_mode``
    Whether ``Atom1_NRings`` / ``Atom2_NRings`` use legacy DFS back-edge atom
    counts (dump parity) or RDKit ``RingInfo.NumAtomRings`` (principled).
"""

from __future__ import annotations

import pytest

from xenosite.predict import predict
from xenosite.predict.backends.onnx import OnnxBackend
from xenosite.predict.compare import assert_equiv_results
from xenosite.predict.molecule import parse_smiles

from tests.v0_legacy.rdkit_equiv import (
    assert_bond_descriptor_rows_symmetric,
    assert_bond_scores_openbabel_principled,
    assert_bond_scores_symmetric,
    bond_symmetry_groups,
)
from tests.support import (
    GOLDEN_BOND_NRINGS_PARAMETER,
    GOLDEN_PARAMETER,
    GOLDEN_SYMMETRY_PARAMETER,
    PRINCIPLED_PARAMETER,
    ROOT,
    golden_predict_kwargs,
    golden_score_fields,
    onnx_weights_present,
    rows_for_model,
    onnx_root,
)

BACKEND = OnnxBackend(onnx_root())

# Molecules chosen because the committed regression suite already shows legacy ≠
# principled on them — they are small enough to reason about in comments.
NAPHTHALENE = "c1ccc2ccccc2c1"
NDEALK_DIVERGENT = "COc1ccc2nc(C)cc(NCCCN3CCOCC3)c2c1"
DIBENZOFURAN = "c1ccc2c(c1)oc1ccccc12"
PHENANTHRENE = "c1ccc2c(c1)ccc1ccccc12"
ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"


def _predict(smiles: str, model: str, *, parameter: dict | None = None):
    if not onnx_weights_present("ndealk" if model == "isozyme" else model):
        pytest.skip(f"no ONNX weights for {model}")
    kwargs: dict = {"models": [model], "backend": BACKEND}
    if parameter is not None:
        kwargs["_parameter"] = parameter
    return predict(smiles, **kwargs)


def _fields(mol, model: str) -> dict:
    if model == "isozyme":
        hit = next(r for r in mol.results if r.model.startswith("isozyme."))
        return golden_score_fields(hit)
    if model == "reactivity":
        hit = next(r for r in mol.results if r.model.startswith("reactivity."))
        return golden_score_fields(hit)
    return golden_score_fields(mol.results[0])


def _max_delta(a: dict, b: dict) -> float:
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


# ---------------------------------------------------------------------------
# Chapter 0 — what callers should know
# ---------------------------------------------------------------------------


def test_chapter_0_public_predict_uses_principled_defaults():
    """Production ``predict()`` needs no ``_parameter``; golden tests opt into legacy.

    Golden JSON was captured from legacy-test-api with historical site keys and
    OpenBabel symmetry. Application code should omit ``_parameter`` unless it
    intentionally reproduces those numbers.
    """
    mol = _predict(NAPHTHALENE, "quinone")
    # Runners leave _parameter empty unless the caller set it.
    assert mol._parameter == {}

    explicit = _predict(NAPHTHALENE, "quinone", parameter=PRINCIPLED_PARAMETER)
    assert_equiv_results(_fields(mol, "quinone"), _fields(explicit, "quinone"), atol=0.0)


def test_chapter_0_golden_tests_use_legacy_bundle():
    """``golden_predict_kwargs`` merges legacy site/OMP/symmetry for fixture parity.

    This is why ``test_golden_suite.py`` can assert near-equality with rows that
    were regathered from legacy modes while ``predict()`` defaults stay principled.
    """
    kwargs = golden_predict_kwargs("ndealk")
    assert kwargs["_parameter"]["ndealk_site_mode"] == "legacy"
    assert kwargs["_parameter"]["quinone_omp_mode"] == "legacy"
    assert kwargs["_parameter"]["symmetry_group_mode"] == "openbabel"
    assert kwargs["_parameter"]["bond_nrings_mode"] == "legacy"


# ---------------------------------------------------------------------------
# Chapter 1 — quinone OMP (ortho / meta / para descriptors)
# ---------------------------------------------------------------------------


def test_chapter_1_quinone_legacy_uses_one_bfs_shortest_path():
    """Legacy OMP: one BFS path; principled: mean over *all* shortest paths.

    Naphthalene has tied shortest paths between several atom pairs. Legacy picks
    one path depending on BFS neighbor order; principled averages the {0,1}
    ring-touch indicators across every tied path. Atom ONNX inputs change → pair
    and mol heads see different features → scores diverge.

    Implementation: ``features/atom.py`` ``_paths_for_omp`` + ``_omp_paths``;
    ``features/molgraph.py`` ``all_shortest_paths``.
    """
    principled = _predict(NAPHTHALENE, "quinone")
    legacy = _predict(NAPHTHALENE, "quinone", parameter=GOLDEN_PARAMETER)

    p_scores = _fields(principled, "quinone")
    l_scores = _fields(legacy, "quinone")

    delta = _max_delta(p_scores, l_scores)
    # Both paths are valid chemistry; we only assert they are not bitwise equal.
    assert delta > 1e-6, (
        "expected naphthalene quinone legacy vs principled OMP to differ; "
        f"max delta was {delta}"
    )


def test_chapter_1_quinone_omp_flag_is_isolated():
    """Only ``quinone_omp_mode`` needs to flip for quinone; other flags are inert."""
    only_omp_legacy = _predict(
        NAPHTHALENE,
        "quinone",
        parameter={"quinone_omp_mode": "legacy"},
    )
    full_golden = _predict(NAPHTHALENE, "quinone", parameter=GOLDEN_PARAMETER)
    assert_equiv_results(
        _fields(only_omp_legacy, "quinone"),
        _fields(full_golden, "quinone"),
        atol=0.0,
    )


# ---------------------------------------------------------------------------
# Chapter 2 — ndealk site collapse (BondTD rows → bond vector)
# ---------------------------------------------------------------------------


def test_chapter_2_ndealk_principled_deduplicates_symmetry_classes():
    """Principled ndealk keeps one score per symmetry class, then RDKit-pools.

    Legacy emits one site key per BondTD row (including orphan ``max+1`` keys for
    some topo duplicates). Principled dedupes first, so the mapped bond vector can
    have fewer non-zero entries even though ONNX row scores are identical.

    Pooling then assigns the active class score to every sibling bond (only one
    score was active after dedup, so mean == that score).

    ``NDEALK_DIVERGENT`` is the same SMILES used in ``test_onnx_principled.py``.
    """
    principled = _predict(NDEALK_DIVERGENT, "ndealk")
    legacy = _predict(NDEALK_DIVERGENT, "ndealk", parameter=GOLDEN_PARAMETER)

    p_bond = _fields(principled, "ndealk")["bond"]
    l_bond = _fields(legacy, "ndealk")["bond"]

    assert _max_delta(_fields(principled, "ndealk"), _fields(legacy, "ndealk")) > 1e-6

    # Principled path: every RDKit-symmetric bond shares one value.
    rdmol, _ = parse_smiles(NDEALK_DIVERGENT)
    groups = bond_symmetry_groups(rdmol)
    assert groups
    assert_bond_scores_symmetric(p_bond, groups)


def test_chapter_2_ndealk_legacy_site_mode_alone_reproduces_golden_ndealk_flags():
    """Golden ndealk parity needs legacy site keys, OB symmetry, and legacy NRings."""
    legacy_site_only = _predict(
        NDEALK_DIVERGENT,
        "ndealk",
        parameter={
            "ndealk_site_mode": "legacy",
            **GOLDEN_SYMMETRY_PARAMETER,
            **GOLDEN_BOND_NRINGS_PARAMETER,
        },
    )
    golden = _predict(NDEALK_DIVERGENT, "ndealk", parameter=GOLDEN_PARAMETER)
    assert_equiv_results(
        _fields(legacy_site_only, "ndealk"),
        _fields(golden, "ndealk"),
        atol=0.0,
    )


# ---------------------------------------------------------------------------
# Chapter 3 — symmetry_group_mode (RDKit pooling vs OpenBabel GID)
# ---------------------------------------------------------------------------


def test_chapter_3_rdkit_pooling_averages_different_scores_in_one_class():
    """When symmetric bonds have different raw scores, pooling uses their mean.

    Epoxidation on fused rings is the motivating case: dual atom-ordering and
    slightly different BondTD rows can land in the same RDKit class with scores
    that differ by ~0.01. Copying the first score would leave residual asymmetry;
    averaging yields one principled value for the whole class.
    """
    from xenosite.predict.symmetry import broadcast_bond_scores_within_rdkit_groups

    rdmol, _ = parse_smiles(DIBENZOFURAN)
    groups = bond_symmetry_groups(rdmol)
    assert groups

    # Pick a multi-member class and inject unequal stand-in scores on those bonds.
    members = next(iter(groups.values()))
    raw = [0.0] * rdmol.GetNumBonds()
    raw[members[0]] = 0.52
    raw[members[1]] = 0.48
    pooled = broadcast_bond_scores_within_rdkit_groups(rdmol, raw)
    assert pooled[members[0]] == pytest.approx(0.50)
    assert pooled[members[1]] == pytest.approx(0.50)


def test_chapter_3_epoxidation_rdkit_pooling_fixes_fused_polycyclics():
    """Epoxidation averages dual orderings, then pools within RDKit classes on the vector.

    Without pooling, dual-ordering ONNX averages can leave ~0.01 score gaps between
    bonds RDKit considers equivalent (e.g. dibenzofuran). Production
    ``symmetry_group_mode=rdkit`` sets each class to the mean of its active bond
    scores. Golden uses ``openbabel`` so bond vectors stay identical to fixtures.
    """
    production = _predict(DIBENZOFURAN, "epoxidation")
    golden = _predict(DIBENZOFURAN, "epoxidation", parameter=GOLDEN_PARAMETER)

    rdmol, _ = parse_smiles(DIBENZOFURAN)
    groups = bond_symmetry_groups(rdmol)
    prod_bond = _fields(production, "epoxidation")["bond"]
    assert_bond_scores_symmetric(prod_bond, groups)

    assert _max_delta(_fields(production, "epoxidation"), _fields(golden, "epoxidation")) > 1e-6


def test_chapter_3_openbabel_principled_allows_single_active_bond_per_class():
    """OpenBabel grouping dedupes rows but does *not* RDKit-pool to siblings.

    This is the intermediate mode used in ``test_equiv_groups.py`` for ndealk:
    principled site collapse + OpenBabel classes → at most one non-zero score per
    coarse RDKit bond symmetry group (zeros on symmetric siblings are OK).

    Production uses RDKit classes *and* pooling so siblings share one score.
    """
    rdmol, molecule = parse_smiles(NDEALK_DIVERGENT)
    groups = bond_symmetry_groups(rdmol)
    param = {"ndealk_site_mode": "principled", **GOLDEN_SYMMETRY_PARAMETER}
    mol = _predict(NDEALK_DIVERGENT, "ndealk", parameter=param)
    bond = _fields(mol, "ndealk")["bond"]
    assert len(bond) == len(molecule.bonds.idx)
    assert_bond_scores_openbabel_principled(bond, groups)

    # Same molecule under full production defaults is stricter (full pooling).
    prod = _predict(NDEALK_DIVERGENT, "ndealk")
    prod_bond = _fields(prod, "ndealk")["bond"]
    assert_bond_scores_symmetric(prod_bond, groups)


# ---------------------------------------------------------------------------
# Chapter 4 — bond_nrings_mode (Atom1_/Atom2_ NRings in BondTD)
# ---------------------------------------------------------------------------


def _bond_row_nrings(rows: list[dict], atoms: tuple[int, int]) -> tuple[float, float]:
    for row in rows:
        if tuple(row["_atoms"]) == atoms:
            return float(row["Atom1_NRings"]), float(row["Atom2_NRings"])
    raise AssertionError(f"no bond row for {atoms}")


def test_chapter_4_bond_nrings_counts_endpoint_atoms_not_the_bond():
    """``Atom*_NRings`` is how many rings contain that endpoint atom, not the bond.

    BondTD fills one row per heavy-atom bond with directed ``Atom1_`` / ``Atom2_``
    columns. Confusing these with ``RingInfo.NumBondRings`` leads to wrong
    expectations about symmetry.
    """
    from xenosite.predict.features import ndealk_bond_rows

    rdmol, _ = parse_smiles(PHENANTHRENE)
    rows = ndealk_bond_rows(rdmol, symmetry_group_mode="openbabel", bond_nrings_mode="principled")
    a1, a2 = _bond_row_nrings(rows, (0, 1))
    b1, b2 = _bond_row_nrings(rows, (10, 11))
    assert (a1, a2) == (b1, b2) == (1.0, 1.0)


def test_chapter_4_bond_nrings_legacy_dfs_breaks_fused_polycyclic_symmetry():
    """Legacy DFS back-edge counts differ on symmetric bonds; principled RDKit does not.

    Phenanthrene bonds ``(0,1)`` and ``(10,11)`` share the same directed OpenBabel
    class but legacy ``dfs_cycles()`` assigns different ``Atom*_NRings`` because
    fusion atoms pick up extra perimeter cycles in the DFS walk. That is faithful
    to the 2.4 dump oracle but breaks descriptor symmetry — so dumps keep
    ``bond_nrings_mode=legacy`` while production defaults to principled.
    """
    from xenosite.predict.features import ndealk_bond_rows

    rdmol, _ = parse_smiles(PHENANTHRENE)
    legacy = ndealk_bond_rows(rdmol, symmetry_group_mode="openbabel", bond_nrings_mode="legacy")
    principled = ndealk_bond_rows(
        rdmol, symmetry_group_mode="openbabel", bond_nrings_mode="principled"
    )
    assert _bond_row_nrings(legacy, (0, 1)) == (1.0, 1.0)
    assert _bond_row_nrings(legacy, (10, 11)) == (2.0, 2.0)
    assert _bond_row_nrings(principled, (0, 1)) == (1.0, 1.0)
    assert _bond_row_nrings(principled, (10, 11)) == (1.0, 1.0)


def test_chapter_4_bond_nrings_principled_rows_match_within_directed_ob_class():
    """Principled NRings + directed OB class ⇒ identical full bond descriptor rows.

    ``tests/test_principled_descriptor_symmetry.py`` runs this over all ob_dumps;
    here we spell out the reasoning on one fused polycyclic example.
    """
    from xenosite.predict.features import load_names

    rdmol, _ = parse_smiles(PHENANTHRENE)
    rows = rows_for_model("ndealk", rdmol, _parameter=PRINCIPLED_PARAMETER)
    names = load_names("ndealk", "bond")
    assert names
    assert_bond_descriptor_rows_symmetric(rows, rdmol, names)


def test_chapter_4_bond_nrings_legacy_flag_required_for_ob_dump_parity():
    """``rows_for_model`` defaults to legacy NRings unless ``PRINCIPLED_PARAMETER`` is set."""
    rdmol, _ = parse_smiles(PHENANTHRENE)
    dump_rows = rows_for_model("ndealk", rdmol)
    legacy_rows = rows_for_model(
        "ndealk",
        rdmol,
        _parameter={"bond_nrings_mode": "legacy"},
    )
    assert _bond_row_nrings(dump_rows, (10, 11)) == _bond_row_nrings(legacy_rows, (10, 11))
    assert _bond_row_nrings(dump_rows, (10, 11)) == (2.0, 2.0)


# ---------------------------------------------------------------------------
# Chapter 5 — models outside the four-flag surface
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model", ["reactivity", "ugt"])
def test_chapter_5_reactivity_and_ugt_ignore_legacy_parameter_bundle(model: str):
    """Reactivity and UGT have no site/OMP/symmetry branches — scores are unchanged.

    Passing ``GOLDEN_PARAMETER`` on aspirin is a no-op for these models, which is
    why golden parity tests can share one helper without special-casing them off.
    """
    plain = _predict(ASPIRIN, model)
    with_golden = _predict(ASPIRIN, model, parameter=GOLDEN_PARAMETER)
    assert_equiv_results(_fields(plain, model), _fields(with_golden, model), atol=0.0)


# ---------------------------------------------------------------------------
# Chapter 6 — mental model checklist (documentation-as-test)
# ---------------------------------------------------------------------------


def test_chapter_6_checklist_defaults_vs_golden():
    """Sanity checklist tying the flags to their production vs golden values."""
    assert PRINCIPLED_PARAMETER["ndealk_site_mode"] == "principled"
    assert PRINCIPLED_PARAMETER["quinone_omp_mode"] == "principled"
    assert PRINCIPLED_PARAMETER["symmetry_group_mode"] == "rdkit"
    assert PRINCIPLED_PARAMETER["bond_nrings_mode"] == "principled"

    assert GOLDEN_PARAMETER["ndealk_site_mode"] == "legacy"
    assert GOLDEN_PARAMETER["quinone_omp_mode"] == "legacy"
    assert GOLDEN_PARAMETER["symmetry_group_mode"] == "openbabel"
    assert GOLDEN_PARAMETER["bond_nrings_mode"] == "legacy"
