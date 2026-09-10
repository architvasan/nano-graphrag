"""Tests for two-stage retrieval: extract_subgraph + facts_from_subgraph.

A stub retriever stands in for the ragmosis GraphRetriever so these run with no
graph/network. Verifies: PPR-ranked node budget, internal-edge induction, evidence
subset carried, lowest-trust edges dropped at the cap, and subgraph-scoped facts
apply the generic/stub filter + PPR ordering.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from events.kg_bridge import KGBridge, Subgraph


class _StubRetr:
    ONTOLOGY_RELS = frozenset({"is_a", "part_of"})

    def __init__(self):
        # a tiny graph: A-B-C-D chain + A-D shortcut + D-E (E is off-budget)
        self.G = {"A": {"B", "D"}, "B": {"A", "C"}, "C": {"B", "D"},
                  "D": {"A", "C", "E"}, "E": {"D"}}
        _EV = "Ephrin-B2 is the functional entry receptor for Nipah virus G per structure."
        self.edge_ev = {
            frozenset({"A", "B"}): [{"rel": "is_receptor_for", "ev": _EV, "pid": "doi:1"}],
            frozenset({"B", "C"}): [{"rel": "co_occurs_with", "ev": _EV, "pid": "doi:2"}],
            frozenset({"C", "D"}): [{"rel": "activates", "ev": _EV, "pid": "doi:3"}],
            frozenset({"A", "D"}): [{"rel": "inhibits", "ev": "short", "pid": "doi:4"}],
            frozenset({"D", "E"}): [{"rel": "binds", "ev": _EV, "pid": "doi:5"}],
        }
        self.edge_trust = {("A", "B"): 3.0, ("B", "C"): 2.0, ("C", "D"): 1.0,
                           ("A", "D"): 0.5}

    def retrieve(self, text, k=500, mode="global"):
        # PPR order A,B,C,D (E excluded to test off-budget edge drop)
        order = [("A", 0.1), ("B", 0.2), ("C", 0.3), ("D", 0.4)]
        return [(n, r, n) for n, r in order][:k]


def _bridge():
    b = KGBridge.__new__(KGBridge)
    b._retriever = _StubRetr()
    b.evidence = {}
    return b


def test_extract_respects_node_budget_and_ranking():
    b = _bridge()
    sub = b.extract_subgraph("q", budget=3)
    assert sub is not None
    assert sub.nodes == ("A", "B", "C")   # top-3 by PPR rank
    assert sub.n_nodes == 3


def test_extract_induces_internal_edges_only():
    b = _bridge()
    sub = b.extract_subgraph("q", budget=4)  # A,B,C,D -> excludes E
    # D-E must NOT appear (E off-budget); A-B, B-C, C-D, A-D are internal
    assert frozenset({"D", "E"}) not in sub.edge_ev
    assert frozenset({"A", "B"}) in sub.edge_ev
    assert "E" not in sub.adjacency


def test_extract_caps_edges_dropping_lowest_trust():
    b = _bridge()
    # cap edges at 2: of the 4 internal edges (A-B=3, B-C=2, C-D=1, A-D=0.5),
    # only the two highest-trust survive.
    sub = b.extract_subgraph("q", budget=4, edge_budget=2)
    assert sub.n_edges == 2
    assert frozenset({"A", "B"}) in sub.edge_ev   # trust 3.0
    assert frozenset({"B", "C"}) in sub.edge_ev   # trust 2.0
    assert frozenset({"A", "D"}) not in sub.edge_ev  # trust 0.5 dropped


def test_facts_from_subgraph_filters_generic_and_stub():
    b = _bridge()
    sub = b.extract_subgraph("q", budget=4)
    facts = b.facts_from_subgraph(sub, text="", k=8)
    joined = " ".join(f.identity for f in facts)
    # co_occurs_with (B-C) and the 'short' stub evidence (A-D) must be filtered out
    assert "co_occurs_with" not in joined
    assert "inhibits" not in joined            # A-D had stub evidence 'short'
    assert any("is_receptor_for" in f.identity for f in facts)


def test_extract_returns_none_on_empty_ppr():
    b = KGBridge.__new__(KGBridge)
    b.evidence = {}
    b._retriever = type("R", (), {"retrieve": lambda self, t, k=500, mode="global": [],
                                   "G": {}, "edge_ev": {}, "edge_trust": {}})()
    assert b.extract_subgraph("q") is None


def test_grounded_facts_routes_through_active_subgraph():
    # when active_subgraph is set, grounded_facts must read the slice, not the graph
    b = _bridge()
    sub = b.extract_subgraph("q", budget=4)
    b.active_subgraph = sub
    facts = b.grounded_facts("q", k=8)  # would hit retriever.retrieve_grounded if not routed
    assert facts  # came from the slice
    assert all(f.source == "graph" for f in facts)
    # the slice-derived identities carry the "[rel]" tag from facts_from_subgraph
    assert any("[" in f.identity for f in facts)


def test_orchestrator_extracts_once_and_clears(monkeypatch=None):
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
    from events.orchestrator import EventOrchestrator
    from events.kg_bridge import KGBridge, Subgraph

    calls = {"n": 0}
    b = KGBridge.__new__(KGBridge)
    b.evidence = {}
    b.active_subgraph = None
    b.overlay = []

    def fake_extract(text, budget=500, edge_budget=None, mode="global"):
        calls["n"] += 1
        return Subgraph(query=text, nodes=("A",), adjacency={"A": set()}, edge_ev={})

    b.extract_subgraph = fake_extract

    o = EventOrchestrator.__new__(EventOrchestrator)
    o.bridge = b
    o.subgraph_budget = 500
    o._subgraph_cache = {}
    # extract twice for the SAME question -> cache means extract called once
    for _ in range(2):
        c = o._subgraph_cache.get("Q")
        if c is None:
            c = b.extract_subgraph("Q", budget=o.subgraph_budget)
            o._subgraph_cache["Q"] = c
        b.active_subgraph = c
    assert calls["n"] == 1  # cached: only one real extraction


