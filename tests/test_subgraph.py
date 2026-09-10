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


def _ev_bridge():
    """A bridge with a stub embedder that counts calls, for evidence-cache tests."""
    from events.kg_bridge import KGBridge
    import numpy as np
    b = KGBridge.__new__(KGBridge)
    b._embed = None
    b._ev_vecs = {}
    b._ev_cache_loaded = True   # skip disk load
    b._ev_cache_dirty = False
    b.evidence_cache_path = "/tmp/hermes-test-ev-cache.npz"
    b.edge_index_path = "/tmp/hermes-test-edge-index.json"
    b._edge_index = None
    calls = {"n": 0, "texts": 0}

    def fake_embed_many(texts):
        calls["n"] += 1
        calls["texts"] += len(texts)
        return [np.ones(8, dtype="float32") * (len(t) + 1) for t in texts]

    b.embed_many = fake_embed_many
    return b, calls


def test_evidence_vectors_cache_hit_and_dedupe():
    b, calls = _ev_bridge()
    # first lookup with duplicates: 3 distinct strings -> 1 embed call, 3 texts
    out = b.evidence_vectors(["aaa", "bbb", "aaa", "ccc", "bbb"])
    assert len(out) == 5
    assert calls["n"] == 1
    assert calls["texts"] == 3          # deduped: only 3 unique embedded
    assert b._ev_cache_dirty is True
    # duplicates share one vector
    assert (out[0] == out[2]).all()     # both "aaa"
    # second lookup of same strings -> ZERO new embeds (all cache hits)
    calls["n"] = 0
    out2 = b.evidence_vectors(["aaa", "ccc"])
    assert calls["n"] == 0


def test_evidence_cache_roundtrip_persist_and_load():
    import os
    from events.kg_bridge import KGBridge
    b, _ = _ev_bridge()
    b.evidence_vectors(["nipah g attaches ephrinb2", "measles h binds cd150"])
    assert b.flush_ev_cache() is True
    assert os.path.exists(b.evidence_cache_path)
    # a fresh bridge loads the persisted cache -> no embed needed
    import numpy as np
    b2 = KGBridge.__new__(KGBridge)
    b2._embed = None
    b2._ev_vecs = {}
    b2._ev_cache_loaded = False
    b2._ev_cache_dirty = False
    b2.evidence_cache_path = b.evidence_cache_path
    called = {"n": 0}
    b2.embed_many = lambda ts: (_ for _ in ()).throw(AssertionError("should not embed"))
    out = b2.evidence_vectors(["measles h binds cd150"])  # pure cache hit
    assert out[0] is not None
    os.remove(b.evidence_cache_path)


def test_web_disabled_blocks_all_web():
    """web_disabled=True must return no web hits AND no web_fill, even with an
    injected web_search_fn present (no silent fallback)."""
    from events.kg_bridge import KGBridge
    b = KGBridge.__new__(KGBridge)
    b.web_disabled = True
    b.web_search_fn = lambda q, n: [{"url": "x", "title": "t", "description": "d"}]
    assert b._web_hits("anything", 3) == []
    assert b.web_fill("anything") is None


def test_edge_key_stable_and_direction_insensitive():
    from events.kg_bridge import KGBridge
    k1 = KGBridge.edge_key("Nipah G", "is_receptor_for", "ephrin-B2")
    k2 = KGBridge.edge_key("ephrin-B2", "is_receptor_for", "Nipah G")  # swapped
    k3 = KGBridge.edge_key("Nipah G", "binds", "ephrin-B2")            # diff rel
    assert k1 == k2            # endpoint order does not matter
    assert k1 != k3            # relation does
    assert len(k1) == 16


def test_update_evidence_cache_embeds_only_new():
    import os, json
    b, calls = _ev_bridge()
    b._edge_index = None
    b.edge_index_path = "/tmp/hermes-test-edge-index.json"
    edges = [
        ("A", "binds", "B", "alpha binds beta with high affinity per assay"),
        ("C", "inhibits", "D", "gamma inhibits delta in the pathway strongly"),
        ("A", "binds", "B", "alpha binds beta with high affinity per assay"),  # dup edge+ev
    ]
    summ = b.update_evidence_cache(edges)
    assert summ["edges"] == 2               # dup edge collapsed
    assert summ["unique_ev"] == 2
    assert summ["embedded"] == 2
    assert os.path.exists(b.edge_index_path)
    # second call with ONE new edge -> embeds only the new evidence
    calls["n"] = 0
    edges2 = edges + [("E", "causes", "F", "epsilon causes phi in the model system")]
    summ2 = b.update_evidence_cache(edges2)
    assert summ2["embedded"] == 1           # only the new one
    assert summ2["already"] == 2
    # vector_for_edge resolves via the join
    v = b.vector_for_edge("A", "binds", "B")
    assert v is not None
    assert b.vector_for_edge("A", "binds", "B") is b.vector_for_edge("B", "binds", "A")
    os.remove(b.edge_index_path)




