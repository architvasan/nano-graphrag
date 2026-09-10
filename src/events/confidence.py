"""Deterministic confidence weighting for episodes (standardized 0-1).

The charter is strict: credit is the numerical component's, and a model never
emits a number a branch consumes. So confidence here is PROVENANCE-DERIVED and
deterministic — computed from measurable signals, never an LLM judgment. (An
LLM justification-quality judge may annotate the Summary channel separately, but
must not feed this weight.)

Two levels:
  fact_confidence  : per-retrieved-fact 0-1 from relevance x provenance tier
  episode_confidence : identity-weighted mean of a walk/subproblem/question's
                       fact confidences -> one 0-1 score for that episode.

Provenance tiers mirror the campaign's info_weight philosophy: a graph edge with
a real paper_id is strongest; an unsourced graph edge is weaker; a web node is
weighted by how well it linked into the subgraph (link_score); a weakly-linked
web node is weakest.
"""
from __future__ import annotations

from typing import Iterable, Optional

__all__ = ["fact_confidence", "episode_confidence", "pool_confidence", "PROV_TIERS"]

#: Provenance tier ceilings (the max confidence a fact of this kind can reach
#: before the relevance multiplier is applied). Deterministic, not learned.
PROV_TIERS = {
    "graph_sourced": 1.00,   # graph edge with a real paper_id (PMID/doi/url)
    "graph_unsourced": 0.70, # graph edge, no verifiable source id
    "web_linked": 0.85,      # web node linked into the subgraph (scaled by link_score)
    "web_weak": 0.40,        # web node with a weak/absent subgraph link
}

_WEAK_LINK = 0.30  # link_score below this => web_weak tier


def _prov_tier(source: str, paper_id: str, link_score: float) -> tuple[str, float]:
    """Classify a fact's provenance tier and its ceiling. Deterministic."""
    pid = (paper_id or "").strip().lower()
    real = bool(pid) and pid not in ("none", "nan", "")
    if source == "web_fill":
        if link_score >= _WEAK_LINK:
            # scale the linked ceiling by how strong the link is (0.30..1.0 -> factor)
            return "web_linked", PROV_TIERS["web_linked"]
        return "web_weak", PROV_TIERS["web_weak"]
    # graph
    return ("graph_sourced", PROV_TIERS["graph_sourced"]) if real else (
        "graph_unsourced", PROV_TIERS["graph_unsourced"]
    )


def fact_confidence(
    *,
    source: str,
    paper_id: str = "",
    relevance: float = 1.0,
    link_score: float = 0.0,
) -> float:
    """0-1 confidence for one fact = provenance ceiling x relevance.

    relevance is a 0-1 cosine of the fact to the query (1.0 when unknown, so an
    unscored fact keeps its full provenance ceiling rather than being penalized).
    For web facts, link_score further scales the linked-tier ceiling.
    """
    tier, ceiling = _prov_tier(source, paper_id, link_score)
    rel = _clamp01(relevance)
    if tier == "web_linked":
        # blend: the stronger the subgraph link, the closer to the full ceiling
        ceiling = PROV_TIERS["web_weak"] + (ceiling - PROV_TIERS["web_weak"]) * _clamp01(link_score)
    return round(_clamp01(ceiling * rel), 4)


def episode_confidence(
    confidences: Iterable[float],
    *,
    ended_by: str = "",
    counted: bool = True,
) -> float:
    """Aggregate a set of fact confidences into ONE 0-1 episode confidence.

    Identity-weighted mean (more crediting facts -> more evidence -> steadier
    estimate), lightly scaled by breadth so a single strong fact does not score
    the same as many. An episode that did not count toward the verdict (bound /
    source-failed) is floored to 0.0 — it made no crediting judgement.
    """
    vals = [_clamp01(c) for c in confidences]
    if not counted or not vals:
        return 0.0
    mean = sum(vals) / len(vals)
    # breadth factor: saturating, so 1 fact ~0.6x, ~5 facts ~0.9x, 10+ ~1.0x
    breadth = 1.0 - 1.0 / (1.0 + 0.4 * len(vals))
    return round(_clamp01(mean * (0.6 + 0.4 * breadth / _breadth_at(10))), 4)


def _breadth_at(n: int) -> float:
    return 1.0 - 1.0 / (1.0 + 0.4 * n)


def pool_confidence(confidences: Iterable[float], *, counted: bool = True) -> float:
    """Pool independent child confidences via noisy-OR: corroborating evidence
    RAISES confidence, and the parent is never less confident than its best
    child (no depth penalty from nesting).

        pooled = 1 - prod(1 - c_i)

    Use this to combine a parent episode's CHILD episode confidences (each a
    0-1 belief that its branch established something). Contrast episode_confidence,
    which averages a walk's per-fact confidences (facts within one walk are not
    independent draws). An uncounted parent floors to 0.0.
    """
    vals = [_clamp01(c) for c in confidences]
    if not counted or not vals:
        return 0.0
    prod = 1.0
    for c in vals:
        prod *= (1.0 - c)
    return round(_clamp01(1.0 - prod), 4)


def _clamp01(x: float) -> float:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return 0.0
    return 0.0 if x < 0 else 1.0 if x > 1 else x
