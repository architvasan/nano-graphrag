"""Gated merge of web-rescued edges into a PARALLEL graph copy.

Standing instruction: perform the web rescue, add the result as a new edge,
retry. Made rigorous here with a two-gate merge and, per charter + the user's
parallel-copy rule, the mutation targets a COPY of the graph's evidence tables —
the base KG is never changed. A completed run emits a reviewable MergeReport.

Two gates (both must pass):
  A. the web edge carries REAL evidence text (not an id-stub / empty), judged by
     the retriever's own _is_stub rule so the bar matches retrieval.
  B. it may only attach to a NON-GENERIC, CONTENTFUL base edge: the base anchor's
     predicate is not ontology/taxonomy and not co_occurs_with, and the base edge
     itself carries real evidence. Web knowledge never fuses onto co-mention hubs.
  (plus a semantic-link floor: link_score >= merge_min.)
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Optional

__all__ = ["MergeGate", "MergeReport", "merge_overlay_into_parallel"]

#: predicates that are too generic to anchor merged web knowledge.
GENERIC_RELS = frozenset({"co_occurs_with", "co_occurs", "co-mention", "comention",
                          "mentioned_with", "associated_with", "related_to"})


@dataclass
class MergeReport:
    """Auditable outcome of a merge pass — accepted edges and rejects w/ reason."""

    accepted: list[dict] = field(default_factory=list)
    rejected: list[dict] = field(default_factory=list)

    def as_record(self) -> dict:
        return {
            "n_accepted": len(self.accepted),
            "n_rejected": len(self.rejected),
            "accepted": self.accepted,
            "rejected": self.rejected,
        }


@dataclass
class MergeGate:
    """The two-gate acceptance test, using the retriever's own rules."""

    retriever: Any
    merge_min: float = 0.30

    def _is_stub(self, ev: str) -> bool:
        """Real-evidence test, delegated to the retriever so the bar matches."""
        fn = getattr(self.retriever, "_is_stub", None)
        if fn is not None:
            try:
                return bool(fn(ev))
            except Exception:  # noqa: BLE001
                pass
        e = (ev or "").strip()
        return (not e) or len(e) < 25

    def _is_generic(self, rel: str) -> bool:
        onto = getattr(self.retriever, "ONTOLOGY_RELS", frozenset())
        rl = (rel or "").strip().lower()
        return rl in GENERIC_RELS or rl in {str(x).lower() for x in onto}

    def base_edge_ok(self, edge_ev: dict, base_node: str) -> tuple[bool, str]:
        """Gate B: base anchor must be a non-generic, contentful edge. Finds any
        edge incident to base_node whose predicate is typed and evidence real."""
        for key, entries in edge_ev.items():
            if base_node not in key:
                continue
            for e in entries or []:
                rel = e.get("rel", "")
                ev = e.get("ev", e.get("evidence", ""))
                if not self._is_generic(rel) and not self._is_stub(ev):
                    return True, rel
        return False, ""

    def web_edge_ok(self, evidence: str) -> bool:
        """Gate A: the web edge itself must carry real evidence text."""
        return not self._is_stub(evidence)


def merge_overlay_into_parallel(
    bridge: Any,
    *,
    merge_min: float = 0.30,
    inferred_rel: str = "web_evidence_for",
) -> tuple[Optional[dict], MergeReport]:
    """Merge this run's overlay edges into a PARALLEL copy of the KG evidence
    tables, gated by MergeGate. Returns (parallel_edge_ev, report). The base
    retriever's edge_ev is deep-copied on first accept and never mutated.

    inferred_rel is the typed predicate assigned to accepted web edges — a real
    (non-generic) relation label, so merged edges are themselves contentful.
    """
    retr = getattr(bridge, "retriever", None)
    overlay = list(getattr(bridge, "overlay", []))
    report = MergeReport()
    if retr is None or not overlay:
        return None, report
    base_ev = getattr(retr, "edge_ev", {}) or {}
    gate = MergeGate(retr, merge_min=merge_min)
    parallel: Optional[dict] = None

    for e in overlay:
        web_id, base, score = e.web_identity, e.base_node, e.score
        ev_text = getattr(e, "text", "")
        rec = {"web": web_id, "base": base, "score": round(score, 4),
               "paper_id": getattr(e, "paper_id", "")}
        if not base:
            report.rejected.append({**rec, "reason": "no base anchor"})
            continue
        if score < merge_min:
            report.rejected.append({**rec, "reason": f"link_score<{merge_min}"})
            continue
        if not gate.web_edge_ok(ev_text):
            report.rejected.append({**rec, "reason": "web edge lacks real evidence"})
            continue
        ok, base_rel = gate.base_edge_ok(base_ev, base)
        if not ok:
            report.rejected.append({**rec, "reason": "base edge generic or no evidence"})
            continue
        # ACCEPT: merge into the parallel copy (deep-copy base_ev on first accept)
        if parallel is None:
            parallel = copy.deepcopy(base_ev)
        key = frozenset({web_id, base})
        parallel.setdefault(key, []).append({  # type: ignore[union-attr]
            "rel": inferred_rel,
            "ev": ev_text,
            "pid": getattr(e, "paper_id", ""),
            "source": "web_merge",
        })
        report.accepted.append({**rec, "merged_rel": inferred_rel,
                                "anchored_on_rel": base_rel})
    return parallel, report
