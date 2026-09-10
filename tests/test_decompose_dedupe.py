"""Focused tests for the decompose-dedupe guards in events.sources.

These cover the two pure/deterministic guards only (string-level and
community-level dedupe). The numerical method core is deliberately untested
per docs/ACQUISITION_LOOP.md; these string helpers are not part of it.
"""
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from events.sources import _dedupe, QuestionSource


def test_string_dedupe_normalizes_case_and_whitespace():
    items = [
        "Which receptor does NiV G bind?",
        "which receptor  does niv g bind?",   # dup: case + double space
        "What residues drive selectivity?",
    ]
    assert _dedupe(items) == [items[0], items[2]]


def test_string_dedupe_keeps_first_seen_order_and_drops_empties():
    assert _dedupe(["b", "", "a", "B", "a "]) == ["b", "a"]


class _StubBridge:
    """Maps each subproblem text to a fixed coarse community id."""

    def __init__(self, mapping):
        self._m = mapping

    def locate(self, text, m=12):
        return self._m.get(text), []


def test_community_dedupe_collapses_same_region_distinct_text():
    # three distinctly-worded subs; A and C resolve to the SAME community 7.
    bridge = _StubBridge({"A": 7, "B": 9, "C": 7})
    qs = QuestionSource(ctx=None, bridge=bridge, question="q", llm=None)
    assert qs._community_dedupe(["A", "B", "C"]) == ["A", "B"]


def test_community_dedupe_fail_open_when_unlocatable():
    # None community => keep (never silently drop unlocatable work).
    bridge = _StubBridge({"A": None, "B": None})
    qs = QuestionSource(ctx=None, bridge=bridge, question="q", llm=None)
    assert qs._community_dedupe(["A", "B"]) == ["A", "B"]


def test_community_dedupe_never_returns_empty():
    bridge = _StubBridge({})
    qs = QuestionSource(ctx=None, bridge=bridge, question="q", llm=None)
    assert qs._community_dedupe(["only"]) == ["only"]
