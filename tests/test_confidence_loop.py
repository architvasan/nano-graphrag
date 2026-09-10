"""Tests for the confidence-target escalation loop in SubproblemSource.

The stop gate is the DETERMINISTIC confidence read from completed walks — never
an LLM self-report (guards the no-abstain gaming pathology). Verifies: stop when
target met, patience bailout when confidence stalls, hard max_walks ceiling.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from events.sources import SubproblemSource


class _Bridge:
    evidence = {}
    overlay = []

    def grounded_facts(self, text, k=8, mechanism_only=True):
        return []  # empty -> off-target -> escalation always has room

    def locate(self, text, m=12):
        return 1, []

    def embed(self, text):
        return [1.0, 0.0]

    def _cosine(self, a, b):
        return sum(x * y for x, y in zip(a, b))


class _Walk:
    """Stub child EpisodeRecord with a fixed per-hop confidence via conf notes."""

    def __init__(self, conf, ended_by="exhausted"):
        self.ended_by = ended_by
        self.unit_records = [_Hop(conf)]


class _Hop:
    def __init__(self, conf):
        self.credit_note = f"conf={conf:.4f}"
        self.child = None


class _UnitOf:
    def __init__(self, walk):
        self.child = walk


class _View:
    def __init__(self, walks):
        self.units = [_UnitOf(w) for w in walks]


def _ss(**kw):
    return SubproblemSource(ctx=None, bridge=_Bridge(), subproblem_text="q", **kw)


def test_stops_when_target_met():
    ss = _ss(conf_target=0.5, max_walks=6, patience=3)
    ss._issued = 1  # pretend one walk already ran
    # a high-confidence completed walk -> running conf >= target -> stop
    assert ss.next(_View([_Walk(0.9)])) is None


def test_keeps_going_below_target():
    ss = _ss(conf_target=0.9, max_walks=6, patience=3)
    ss._issued = 1
    # low-confidence walk, target not met, patience not exhausted -> issue another
    assert ss.next(_View([_Walk(0.2)])) is not None


def test_patience_bailout_when_confidence_stalls():
    ss = _ss(conf_target=0.99, max_walks=10, patience=2)
    # feed the same low confidence repeatedly; should bail after patience dry walks
    ss._issued = 1
    r1 = ss.next(_View([_Walk(0.2)]))            # improves 0 -> 0.2 (best updates)
    ss._issued += 1
    r2 = ss.next(_View([_Walk(0.2), _Walk(0.0)]))  # pooled ~same -> dry 1
    ss._issued += 1
    r3 = ss.next(_View([_Walk(0.2), _Walk(0.0), _Walk(0.0)]))  # dry 2 -> bail
    assert r3 is None


def test_hard_ceiling_never_exceeds_max_walks():
    ss = _ss(conf_target=0.99, max_walks=2, patience=99)
    ss._issued = 2  # already at ceiling
    assert ss.next(_View([_Walk(0.1), _Walk(0.1)])) is None
