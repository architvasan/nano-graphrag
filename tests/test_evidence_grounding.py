"""Tests for grounding compose/distill on FULL evidence text, not identities."""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from events.orchestrator import EventOrchestrator


class _Bridge:
    def __init__(self, evidence):
        self.evidence = evidence


def _orch(evidence):
    o = EventOrchestrator.__new__(EventOrchestrator)
    o.bridge = _Bridge(evidence)
    return o


def test_evidence_for_expands_identities_to_full_text():
    ev = {"A --binds--> B": "A binds B via the G-H loop, per PMID:123."}
    out = _orch(ev)._evidence_for(["A --binds--> B"])
    assert "G-H loop" in out          # full sentence, not just the identity head
    assert "PMID:123" in out


def test_evidence_for_falls_back_to_identity_when_missing():
    out = _orch({})._evidence_for(["Unknown --x--> Y"])
    assert "Unknown --x--> Y" in out  # never drops the relation


def test_evidence_for_respects_limit():
    ev = {f"id{i}": f"text{i}" for i in range(10)}
    out = _orch(ev)._evidence_for([f"id{i}" for i in range(10)], limit=3)
    assert out.count("•") == 3
