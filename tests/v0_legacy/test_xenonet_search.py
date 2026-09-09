"""PhaseOneRS one-step + beam graphs. Needs phase1 ONNX + xenosite.forest."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from rdkit import Chem

from xenosite.predict.backends.onnx import OnnxBackend
from xenosite.predict.features import _ob
from xenosite.predict.v0_legacy.xenonet import build_network, site_table_for
from xenosite.predict.v0_legacy.xenonet.score import (
    get_prob_for_one_site,
    phase1_site_strings,
)
from xenosite.predict.v0_legacy.xenonet.search import possible_metabolites

from tests.support import PARITY_ATOL, onnx_root, onnx_weights_present

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "xenonet"

ETHANE = "CC"
ETHYLENE = "C=C"
ETHANOL = "CCO"


def _backend():
    if not _ob.installed():
        pytest.skip("OpenBabel not installed")
    root = onnx_root()
    if not onnx_weights_present("phase1"):
        from xenosite.predict.weights import default_cache_dir

        cache = default_cache_dir()
        if cache is None or not any((cache / "phase1").glob("*.onnx")):
            pytest.skip("phase1 ONNX missing")
        root = cache
    return OnnxBackend(root)


def test_class_rows_have_h_and_bond_indexes():
    be = _backend()
    table = site_table_for(ETHANE, backend=be)
    assert table.index
    assert table.n_heavy == 2
    assert any(int(idx.split(".")[-1]) > 2 for idx in table.index)
    p = get_prob_for_one_site(table, "1.h", "StableOxygenation")
    assert p is None or (0.0 <= p <= 1.0)


def test_one_step_ethane_has_hydroxylation_edges():
    be = _backend()
    g = build_network(ETHANE, depth_limit=1, beam_width=1000, backend=be, scoring="0")
    rules = {e.rule for e in g.all_edges()}
    assert "Hydroxylation" in rules
    for e in g.all_edges():
        assert e.weight >= 0.0


def test_one_step_ethylene_epoxidation():
    be = _backend()
    g = build_network(ETHYLENE, depth_limit=1, beam_width=1000, backend=be)
    rules = {e.rule for e in g.all_edges()}
    assert "Epoxidation" in rules


def test_possible_metabolites_sites_are_rdkit_zero():
    mol = Chem.MolFromSmiles(ETHANE)
    sites = [site for (rule, site), _p in possible_metabolites(mol) if rule == "Hydroxylation"]
    assert sites
    n = mol.GetNumAtoms()
    for site in sites:
        assert all(0 <= i < n for i in site)
        assert phase1_site_strings("Hydroxylation", site)


def test_depth_two_has_likelihoods():
    be = _backend()
    g = build_network(ETHANE, depth_limit=2, beam_width=20, backend=be)
    assert g.nodes[g.root].metabolism_score == 1.0


def test_targeted_ethanol_from_ethane():
    be = _backend()
    g = build_network(
        ETHANE,
        depth_limit=1,
        beam_width=1000,
        targets=[ETHANOL],
        backend=be,
    )
    assert g.root in g.nodes


def test_v0_v1_same_topology_large_beam():
    be = _backend()
    a = build_network(ETHANE, depth_limit=1, beam_width=1000, backend=be, scoring="0")
    b = build_network(ETHANE, depth_limit=1, beam_width=1000, backend=be, scoring="1")
    ta = {(e.parent, e.child, e.rule, e.site) for e in a.all_edges()}
    tb = {(e.parent, e.child, e.rule, e.site) for e in b.all_edges()}
    assert ta == tb


def _load_fixture(name: str) -> dict | None:
    path = FIXTURES / name
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _assert_graph_match(got: dict, expect: dict) -> None:
    ge = {(e["parent"], e["child"], e["rule"], tuple(e["site"])) for e in got["edges"]}
    ee = {(e["parent"], e["child"], e["rule"], tuple(e["site"])) for e in expect["edges"]}
    assert ge == ee, f"edge topology mismatch extra={ge - ee} missing={ee - ge}"
    w_got = {
        (e["parent"], e["child"], e["rule"], tuple(e["site"])): e["weight"]
        for e in got["edges"]
    }
    for e in expect["edges"]:
        key = (e["parent"], e["child"], e["rule"], tuple(e["site"]))
        assert abs(w_got[key] - e["weight"]) <= PARITY_ATOL
    if expect.get("nodes") and any(
        n.get("metabolism_score") is not None for n in expect["nodes"]
    ):
        gs = {n["smiles"]: n.get("metabolism_score") for n in got["nodes"]}
        for n in expect["nodes"]:
            if n.get("metabolism_score") is None:
                continue
            assert n["smiles"] in gs
            if gs[n["smiles"]] is None:
                continue
            assert abs(gs[n["smiles"]] - n["metabolism_score"]) <= PARITY_ATOL


@pytest.mark.parametrize(
    "fixture,depth",
    [
        ("ethane_depth1.json", 1),
        ("ethylene_depth1.json", 1),
        ("ethane_depth2.json", 2),
    ],
)
def test_golden_fixtures(fixture, depth):
    payload = _load_fixture(fixture)
    if payload is None:
        pytest.skip(f"missing {FIXTURES / fixture}; run tools/gather_xenonet.py")
    be = _backend()
    g = build_network(
        payload["smiles"],
        depth_limit=depth,
        beam_width=int(payload.get("beam_width", 1000)),
        targets=payload.get("targets") or (),
        backend=be,
        scoring="0",
    )
    _assert_graph_match(g.to_dict(), payload)
