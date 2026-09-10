"""Focused tests for the spawnable web-search child + graph link-back.

Pure/deterministic behavior only (walk-plan gate, overlay link-back, proposal
export) exercised with a stub bridge — no network, proxy, or KG required. The
numerical method core stays untested per docs/ACQUISITION_LOOP.md.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from events.kg_bridge import GraphFact, OverlayEdge
from events.sources import SubproblemSource, FrontierSource


class _StubBridge:
    """Minimal bridge: fixed frontier + embeddings for gate/link-back tests."""

    def __init__(self, frontier=None, qvec=(1.0, 0.0), fvec=(1.0, 0.0), overlay=None):
        self._frontier = frontier or []
        self._qvec = list(qvec)
        self._fvec = list(fvec)
        self.overlay = overlay or []
        self.evidence = {}

    def grounded_facts(self, text, k=8, mechanism_only=True):
        return list(self._frontier)

    def locate(self, text, m=12):
        return 1, []

    def embed(self, text):
        return self._fvec if "beta-propeller" in text or "graph" in text else self._qvec

    def _cosine(self, a, b):
        return sum(x * y for x, y in zip(a, b))


def _gf(name):
    return GraphFact(identity=name, text=f"{name} graph", source="graph")


def test_web_spawns_when_frontier_empty():
    ss = SubproblemSource(ctx=None, bridge=_StubBridge(frontier=[]),
                          subproblem_text="q", max_walks=2, spawn_web=True)
    ss._locate()
    assert ss._next_kind() == "web"  # first walk leads with web on empty graph


def test_web_spawns_when_offtarget():
    # frontier present but its top fact embeds orthogonally -> below the gate.
    b = _StubBridge(frontier=[_gf("x")], qvec=(1.0, 0.0), fvec=(0.0, 1.0))
    ss = SubproblemSource(ctx=None, bridge=b, subproblem_text="q",
                          max_walks=2, spawn_web=True, web_relevance_gate=0.30)
    ss._locate()
    assert ss._next_kind() == "web"


def test_no_web_when_ontarget():
    # top fact embeds parallel to the query -> cosine 1.0 > gate -> graph first.
    b = _StubBridge(frontier=[_gf("x")], qvec=(1.0, 0.0), fvec=(1.0, 0.0))
    ss = SubproblemSource(ctx=None, bridge=b, subproblem_text="q",
                          max_walks=2, spawn_web=True, web_relevance_gate=0.30)
    ss._locate()
    assert ss._next_kind() == "graph"


def test_spawn_web_false_never_spawns():
    ss = SubproblemSource(ctx=None, bridge=_StubBridge(frontier=[]),
                          subproblem_text="q", max_walks=2, spawn_web=False)
    ss._locate()
    assert ss._next_kind() == "graph"


def test_graph_walk_traverses_overlay_nodes():
    # a web-search walk left an overlay edge; the graph walk must SEE that node.
    edge = OverlayEdge(web_identity="WEB_NODE", base_node="x",
                       score=0.5, text="web claim", paper_id="doi:1")
    b = _StubBridge(frontier=[_gf("x")], overlay=[edge])
    fs = FrontierSource(b, "q")
    fs._load()
    idents = [f.identity for f in fs._frontier]
    assert "WEB_NODE" in idents  # overlay node linked into the traversable frontier
    assert "x" in idents         # base frontier still present


def test_overlay_node_not_duplicated_if_already_in_frontier():
    edge = OverlayEdge(web_identity="x", base_node="y", score=0.5, text="t")
    b = _StubBridge(frontier=[_gf("x")], overlay=[edge])
    fs = FrontierSource(b, "q")
    fs._load()
    assert [f.identity for f in fs._frontier].count("x") == 1
