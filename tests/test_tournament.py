"""Tests for hypothesis-tournament adjudication (combined score + hard flags)."""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from events.tournament import Hypothesis, adjudicate


def _h(label, ans, conf, just="reasoned", correct=None):
    return Hypothesis(label=label, answer=ans, justification=just,
                      confidence=conf, correct=correct)


def test_winner_is_highest_combined_score():
    hyps = [_h("A", "x", 0.3), _h("B", "y", 0.8), _h("C", "z", 0.5)]
    r = adjudicate("q", hyps, judge=None)  # no judge -> quality=confidence
    assert r.winner.label == "B"
    assert r.ranked[0].score >= r.ranked[1].score >= r.ranked[2].score


def test_judge_can_override_weak_confidence_via_quality():
    # low-confidence hyp with a strong justification judge score can win
    def judge(q, a, j):
        return 1.0 if a == "y" else 0.0
    hyps = [_h("A", "x", 0.55), _h("B", "y", 0.45)]
    r = adjudicate("q", hyps, judge=judge, w_conf=0.5, w_just=0.5)
    # A: 0.5*0.55 + 0.5*0 = 0.275 ; B: 0.5*0.45 + 0.5*1 = 0.725
    assert r.winner.label == "B"


def test_lucky_guess_flagged_hard():
    # confident answer, but the judge distrusts its justification
    def judge(q, a, j):
        return 0.1
    hyps = [_h("A", "x", 0.9, just="because vibes")]
    r = adjudicate("q", hyps, judge=judge)
    assert r.is_hard
    assert "lucky guess" in r.hard_reason


def test_weak_field_flagged_hard_3_of_4_rule():
    # 4 graded attempts, only 2 correct -> below the 3-of-4 bar -> hard
    hyps = [_h("A", "x", 0.7, correct=True), _h("B", "y", 0.6, correct=True),
            _h("C", "z", 0.5, correct=False), _h("D", "w", 0.4, correct=False)]
    r = adjudicate("q", hyps, judge=None)
    assert r.is_hard
    assert "weak field" in r.hard_reason


def test_strong_field_not_hard():
    def judge(q, a, j):
        return 0.8
    hyps = [_h("A", "x", 0.7, just="solid", correct=True),
            _h("B", "x", 0.65, just="solid", correct=True),
            _h("C", "x", 0.6, just="solid", correct=True),
            _h("D", "y", 0.5, just="ok", correct=False)]
    r = adjudicate("q", hyps, judge=judge)
    assert not r.is_hard


def test_judge_failure_falls_back_to_confidence():
    def bad_judge(q, a, j):
        raise RuntimeError("judge down")
    hyps = [_h("A", "x", 0.3), _h("B", "y", 0.9)]
    r = adjudicate("q", hyps, judge=bad_judge)
    assert r.winner.label == "B"  # fell back to deterministic confidence


def test_empty_hypotheses_safe():
    r = adjudicate("q", [], judge=None)
    assert r.winner is None and not r.is_hard


def test_tournament_answer_runs_n_attempts_and_selects_winner():
    # stub orchestrator: patch answer_question to return varying confidences
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
    from events.orchestrator import EventOrchestrator, AnswerResult
    from events.summary import Summary

    o = EventOrchestrator.__new__(EventOrchestrator)
    o.llm = None  # no judge -> quality defaults to deterministic confidence

    confs = iter([0.3, 0.9, 0.5, 0.4])

    def fake_answer(question, key="q"):
        c = next(confs)
        return AnswerResult(question=question, answer=f"ans@{c}", n_subproblems=1,
                            n_distinct_identities=1, confidence=c,
                            summary=Summary(scope_key="q", kind="question", confidence=c))

    o.answer_question = fake_answer
    winner_res, tourn = o.tournament_answer("q", n_hypotheses=4)
    assert winner_res.confidence == 0.9        # highest-confidence attempt wins
    assert tourn.winner.answer == "ans@0.9"
    assert len(tourn.ranked) == 4


def test_tournament_diverges_via_llm_factory():
    # the factory must be called once per attempt at rising temps, base llm restored
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
    from events.orchestrator import EventOrchestrator, AnswerResult
    from events.summary import Summary

    o = EventOrchestrator.__new__(EventOrchestrator)
    o.llm = "BASE"
    o.llm_factory = lambda t: f"llm@{t}"
    o.tournament_temps = (0.0, 0.5, 0.9)
    seen_temps = []

    def fake_answer(question, key="q"):
        seen_temps.append(o.llm)  # capture the swapped-in llm during the attempt
        return AnswerResult(question=question, answer="a", n_subproblems=1,
                            n_distinct_identities=1, confidence=0.5,
                            summary=Summary(scope_key="q", kind="question", confidence=0.5))

    o.answer_question = fake_answer
    o._judge = lambda: None
    o.tournament_answer("q", n_hypotheses=3)
    assert seen_temps == ["llm@0.0", "llm@0.5", "llm@0.9"]  # diverging per attempt
    assert o.llm == "BASE"  # base reasoner restored after


