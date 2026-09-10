"""Tests for the gated merge of web edges into a parallel graph copy.

Both gates must hold: (A) the web edge carries real evidence, and (B) it may only
attach to a non-generic, contentful base edge. The base KG (retriever.edge_ev) is
never mutated — accepts land in a deep-copied parallel table.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from events.merge import merge_overlay_into_parallel, MergeGate
from events.kg_bridge import OverlayEdge


class _Retr:
    ONTOLOGY_RELS = frozenset({"is_a", "part_of"})
    STUB = "STUB:"

    def __init__(self, edge_ev):
        self.edge_ev = edge_ev

    def _is_stub(self, ev):
        e = (ev or "").strip()
        return (not e) or e.startswith(self.STUB) or len(e) < 25


class _Bridge:
    def __init__(self, edge_ev, overlay):
        self.retriever = _Retr(edge_ev)
        self.overlay = overlay


_GOOD_EV = "Ephrin-B2 is the functional entry receptor for Nipah virus G, per structure."
_CONTENTFUL_BASE = {
    frozenset({"NiV-G", "ephrin-B2"}): [{"rel": "is_receptor_for", "ev": _GOOD_EV}],
}
_GENERIC_BASE = {
    frozenset({"NiV-G", "ephrin-B2"}): [{"rel": "co_occurs_with", "ev": _GOOD_EV}],
}


def _edge(ev=_GOOD_EV, score=0.5, base="ephrin-B2"):
    return OverlayEdge(web_identity="WEBNODE", base_node=base, score=score,
                       text=ev, paper_id="doi:x")


def test_accept_when_both_gates_pass():
    b = _Bridge(_CONTENTFUL_BASE, [_edge()])
    parallel, report = merge_overlay_into_parallel(b)
    assert len(report.accepted) == 1
    assert parallel is not None
    assert frozenset({"WEBNODE", "ephrin-B2"}) in parallel


def test_reject_web_edge_without_real_evidence():
    b = _Bridge(_CONTENTFUL_BASE, [_edge(ev="short")])  # < 25 chars -> stub
    parallel, report = merge_overlay_into_parallel(b)
    assert len(report.accepted) == 0
    assert parallel is None
    assert "real evidence" in report.rejected[0]["reason"]


def test_reject_generic_base_anchor():
    b = _Bridge(_GENERIC_BASE, [_edge()])  # base rel is co_occurs_with
    parallel, report = merge_overlay_into_parallel(b)
    assert len(report.accepted) == 0
    assert "generic" in report.rejected[0]["reason"]


def test_reject_weak_link_score():
    b = _Bridge(_CONTENTFUL_BASE, [_edge(score=0.1)])
    parallel, report = merge_overlay_into_parallel(b, merge_min=0.30)
    assert len(report.accepted) == 0
    assert "link_score" in report.rejected[0]["reason"]


def test_base_kg_never_mutated():
    original = {frozenset({"NiV-G", "ephrin-B2"}): [{"rel": "is_receptor_for", "ev": _GOOD_EV}]}
    b = _Bridge(original, [_edge()])
    parallel, report = merge_overlay_into_parallel(b)
    # accepted into parallel, but the base edge_ev is unchanged
    assert frozenset({"WEBNODE", "ephrin-B2"}) not in b.retriever.edge_ev
    assert frozenset({"WEBNODE", "ephrin-B2"}) in parallel
