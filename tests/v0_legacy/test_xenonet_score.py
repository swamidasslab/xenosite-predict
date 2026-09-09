"""Unit tests for XenoNet Class-row lookup (no ONNX)."""

from __future__ import annotations

from xenosite.predict.v1.xenonet.score import (
    Phase1SiteTable,
    edge_weight,
    get_prob_for_one_site,
    phase1_site_strings,
    reaction_type,
)


def _table() -> Phase1SiteTable:
    # title.gid.atom1.atom2 — 1-based OB. n_heavy=2 so index ...1.3 is a hydrogen row.
    index = [
        "m.1.1.2",
        "m.1.2.1",
        "m.2.1.3",
        "m.2.2.3",
    ]
    so = [0.4, 0.1, 0.25, 0.05]
    return Phase1SiteTable(
        index=index,
        scores={"StableOxygenation": so, "UnstableOxygenation": [0.0] * 4},
        n_heavy=2,
        atom={"StableOxygenation": [0.4, 0.1]},
        bond={"StableOxygenation": {frozenset({0, 1}): 0.4}},
    )


def test_hydroxylation_h_site():
    t = _table()
    assert get_prob_for_one_site(t, "1.h", "StableOxygenation") == 0.25
    assert get_prob_for_one_site(t, "2.h", "StableOxygenation") == 0.05


def test_bond_site_and_swap():
    t = _table()
    assert get_prob_for_one_site(t, "1.2", "StableOxygenation") == 0.4
    # Legacy mutates the query on a miss: "2.1" flips to "1.2" on the first row.
    assert get_prob_for_one_site(t, "2.1", "StableOxygenation") == 0.4


def test_missing_site_is_none():
    t = _table()
    assert get_prob_for_one_site(t, "9.h", "StableOxygenation") is None


def test_epoxide_opening_is_one():
    t = _table()
    assert get_prob_for_one_site(t, "1.2", "EpoxideOpening") == 1.0


def test_edge_weight_product_and_missing():
    t = _table()
    w = edge_weight(
        t,
        site_strings=frozenset(["1.h"]),
        site_rdkit_zero=frozenset([0]),
        rxn_type="StableOxygenation",
        scoring="0",
    )
    assert w == 0.25
    w0 = edge_weight(
        t,
        site_strings=frozenset(["9.h"]),
        site_rdkit_zero=frozenset([8]),
        rxn_type="StableOxygenation",
        scoring="0",
    )
    assert w0 == 0.0


def test_v1_uses_pooled_bond():
    t = _table()
    w1 = edge_weight(
        t,
        site_strings=frozenset(["1.2"]),
        site_rdkit_zero=frozenset([0, 1]),
        rxn_type="StableOxygenation",
        scoring="1",
    )
    assert w1 == 0.4


def test_phase1_site_strings_and_rxn_dict():
    assert phase1_site_strings("Hydroxylation", frozenset([0])) == frozenset(["1.h"])
    assert phase1_site_strings("Epoxidation", frozenset([0, 1])) == frozenset(["1.2"])
    assert reaction_type("Hydroxylation") == "StableOxygenation"
    assert reaction_type("Dealkylation") == "UnstableOxygenation"
