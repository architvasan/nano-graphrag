"""Tests for the Summary distilled-memory channel (summary_from_record).

Pure/deterministic reduction over a stub record tree — no network, KG, or LLM.
Verifies the nested Summary mirrors the episode tree, carries verbatim relations,
counts identities, and (with a stub distiller) distills only at reasoning levels.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from events.summary import Summary, summary_from_record, distill_summary


class _Rec:
    """Stub EpisodeRecord: scope_key, ended_by, distinct_identities, unit_records."""

    def __init__(self, scope_key, ended_by="exhausted", identities=(), children=()):
        self.scope_key = scope_key
        self.ended_by = ended_by
        self.distinct_identities = list(identities)
        self.unit_records = [_Unit(c) for c in children]


class _Unit:
    def __init__(self, child):
        self.child = child


def _tree():
    walk = _Rec("graph1", "yield_stop", ["A --binds--> B", "C --is--> D"])
    sub = _Rec("sub1", "exhausted", ["A --binds--> B", "C --is--> D"], [walk])
    return _Rec("q", "exhausted", ["A --binds--> B", "C --is--> D"], [sub])


def test_summary_mirrors_episode_tree():
    s = summary_from_record(_tree())
    assert s.kind == "question" and s.scope_key == "q"
    assert len(s.children) == 1 and s.children[0].scope_key == "sub1"
    assert s.children[0].children[0].scope_key == "graph1"


def test_summary_carries_verbatim_relations_and_counts():
    s = summary_from_record(_tree())
    assert "A --binds--> B" in s.key_relations
    assert s.n_identities == 2
    walk = s.children[0].children[0]
    assert walk.kind == "graph" and walk.n_identities == 2


def test_distill_only_at_reasoning_levels():
    calls = []

    def stub_distill(text, relations):
        calls.append((text, tuple(relations)))
        return "DISTILLED"

    s = summary_from_record(_tree(), question="Qtext", distill=stub_distill)
    # question + subproblem have children -> distilled; the graph walk does not
    assert s.text == "DISTILLED"
    assert s.children[0].text == "DISTILLED"
    assert s.children[0].children[0].text == ""  # walk level keeps raw relations
    assert len(calls) == 2


def test_distill_summary_digest_without_llm():
    out = distill_summary("t", ["X --y--> Z", "P --q--> R"], distill=None)
    assert out.startswith("[digest]") and "X --y--> Z" in out


def test_distill_summary_empty_relations():
    assert distill_summary("t", [], distill=None) == "(no evidence reached)"


def test_flatten_relations_dedupes_across_tree():
    s = summary_from_record(_tree())
    flat = s.flatten_relations()
    assert flat.count("A --binds--> B") == 1  # deduped across levels
