"""Tests for the deterministic provenance-weighted confidence scoring."""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from events.confidence import fact_confidence, episode_confidence, pool_confidence, PROV_TIERS


def test_sourced_graph_beats_unsourced():
    a = fact_confidence(source="graph", paper_id="PMID:123", relevance=1.0)
    b = fact_confidence(source="graph", paper_id="", relevance=1.0)
    assert a == PROV_TIERS["graph_sourced"]
    assert b == PROV_TIERS["graph_unsourced"]
    assert a > b


def test_relevance_scales_confidence():
    hi = fact_confidence(source="graph", paper_id="doi:x", relevance=1.0)
    lo = fact_confidence(source="graph", paper_id="doi:x", relevance=0.5)
    assert abs(hi - 1.0) < 1e-9
    assert abs(lo - 0.5) < 1e-9


def test_web_weak_link_lower_than_strong_link():
    strong = fact_confidence(source="web_fill", paper_id="url", relevance=1.0, link_score=0.9)
    weak = fact_confidence(source="web_fill", paper_id="url", relevance=1.0, link_score=0.1)
    assert strong > weak
    assert weak <= PROV_TIERS["web_weak"]


def test_confidence_bounded_0_1():
    for r in (-5.0, 0.0, 0.5, 1.0, 5.0):
        c = fact_confidence(source="graph", paper_id="p", relevance=r)
        assert 0.0 <= c <= 1.0


def test_episode_confidence_zero_when_not_counted():
    assert episode_confidence([0.9, 0.8], counted=False) == 0.0
    assert episode_confidence([], counted=True) == 0.0


def test_episode_confidence_breadth_more_facts_higher():
    one = episode_confidence([0.8], counted=True)
    many = episode_confidence([0.8] * 10, counted=True)
    assert many > one  # more corroborating facts -> steadier, higher score
    assert 0.0 <= one <= 1.0 and 0.0 <= many <= 1.0


def test_episode_confidence_mean_sensitivity():
    hi = episode_confidence([0.9, 0.9, 0.9], counted=True)
    lo = episode_confidence([0.2, 0.2, 0.2], counted=True)
    assert hi > lo


def test_pool_never_below_best_child():
    # noisy-OR: pooled >= max child; corroborating branches raise confidence.
    assert pool_confidence([0.6, 0.5]) >= 0.6
    assert pool_confidence([0.6, 0.5]) > 0.6  # strictly, since both > 0


def test_pool_single_child_is_passthrough():
    assert pool_confidence([0.68]) == 0.68  # no depth penalty for single child


def test_pool_zero_when_not_counted_or_empty():
    assert pool_confidence([0.9, 0.8], counted=False) == 0.0
    assert pool_confidence([]) == 0.0


def test_pool_bounded_0_1():
    assert 0.0 <= pool_confidence([0.9, 0.9, 0.9, 0.9]) <= 1.0
