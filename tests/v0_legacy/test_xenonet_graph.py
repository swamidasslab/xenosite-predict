"""Likelihoods and trim_weights on synthetic graphs (no ONNX)."""

from __future__ import annotations

from xenosite.predict.v0_legacy.xenonet.graph import XenoGraph


def _site(rule: str, *atoms: int):
    return (rule, frozenset(atoms))


def test_likelihoods_two_children():
    g = XenoGraph(root="A")
    g.add_edge("A", "B", 0.4, _site("Hydroxylation", 0))
    g.add_edge("A", "C", 0.6, _site("Hydroxylation", 1))
    scores = g.compute_metabolite_likelihoods()
    assert scores["A"] == 1.0
    assert abs(scores["B"] - 0.4) < 1e-12
    assert abs(scores["C"] - 0.6) < 1e-12


def test_likelihoods_max_multiedge_and_normalize():
    g = XenoGraph(root="A")
    g.add_edge("A", "B", 0.2, _site("Hydroxylation", 0))
    g.add_edge("A", "B", 0.5, _site("Epoxidation", 0, 1))
    g.add_edge("A", "C", 0.5, _site("Hydroxylation", 1))
    scores = g.compute_metabolite_likelihoods(mode="max")
    # max(0.2, 0.5)=0.5 to B; 0.5 to C; each gets half of A's mass.
    assert abs(scores["B"] - 0.5) < 1e-12
    assert abs(scores["C"] - 0.5) < 1e-12


def test_one_step_cycle_dropped():
    g = XenoGraph(root="A")
    g.add_edge("A", "B", 0.5, _site("Hydroxylation", 0))
    g.add_edge("B", "A", 0.9, _site("Dehydration", 0, 1))
    scores = g.compute_metabolite_likelihoods()
    assert scores["A"] == 1.0
    # B is A's only remaining child, so it receives all of A's mass.
    assert abs(scores["B"] - 1.0) < 1e-12


def test_trim_drops_zero_weight():
    g = XenoGraph(root="A")
    g.add_edge("A", "B", 0.3, _site("Hydroxylation", 0))
    g.add_edge("A", "C", 0.0, _site("Hydroxylation", 1))
    trimmed = g.trim_weights(0.0)
    children = {e.child for e in trimmed.all_edges()}
    assert children == {"B"}


def test_to_dict_sorted():
    g = XenoGraph(root="CC")
    g.add_edge("CC", "CCO", 0.1, _site("Hydroxylation", 0))
    d = g.to_dict()
    assert d["root"] == "CC"
    assert d["edges"][0]["parent"] == "CC"
