"""Hypothesis tournaments: competing children adjudicated by the parent.

Children can relate to a parent in two modes:

  COOPERATIVE (default) — every child fans up; credits are the deduped union.
                          This is the existing subproblem behavior.
  TOURNAMENT            — children are RIVAL hypotheses for the same question;
                          the parent scores them and selects winner(s), fanning
                          only the winner up. Realizes dynamic per-hypothesis
                          competition instead of a blind merge.

Adjudication is a COMBINED score, by design:
  * deterministic confidence  — the provenance-weighted 0-1 (ungameable floor;
                                a weak model cannot talk its way to a high score)
  * justification quality     — an LLM judge rates each hypothesis's reasoning
                                (catches good/bad reasoning the floor is blind to)

A question is flagged HARD when the field is weak OR the winner looks like a lucky
guess — high deterministic confidence but a low-quality justification — matching
the "fewer than 3/N correct, or a correct answer with a shitty justification"
rule. HARD questions are the ones worth escalating or routing to a stronger model.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

__all__ = ["Hypothesis", "TournamentResult", "adjudicate", "JudgeFn"]

#: An LLM justification judge: (question, answer, justification) -> 0-1 quality.
JudgeFn = Callable[[str, str, str], float]

# adjudication weights (deterministic floor dominates; judge refines/validates)
W_CONF = 0.6
W_JUST = 0.4
# lucky-guess thresholds: confident answer whose reasoning the judge distrusts
LUCKY_CONF_HI = 0.60
LUCKY_JUST_LO = 0.40


@dataclass
class Hypothesis:
    """One rival attempt at the question: an answer + its grounded justification."""

    label: str
    answer: str
    justification: str = ""
    confidence: float = 0.0          # deterministic provenance confidence
    correct: Optional[bool] = None   # set only when a gold key is available
    judge_quality: float = 0.0       # filled by adjudicate() via the judge
    score: float = 0.0               # combined score, filled by adjudicate()

    def as_record(self) -> dict:
        return {
            "label": self.label,
            "answer": self.answer,
            "justification": self.justification,
            "confidence": round(self.confidence, 4),
            "correct": self.correct,
            "judge_quality": round(self.judge_quality, 4),
            "score": round(self.score, 4),
        }


@dataclass
class TournamentResult:
    """Ranked hypotheses + the selected winner + hardness classification."""

    question: str
    ranked: list[Hypothesis] = field(default_factory=list)
    winner: Optional[Hypothesis] = None
    is_hard: bool = False
    hard_reason: str = ""

    def as_record(self) -> dict:
        return {
            "question": self.question,
            "winner": self.winner.as_record() if self.winner else None,
            "is_hard": self.is_hard,
            "hard_reason": self.hard_reason,
            "ranked": [h.as_record() for h in self.ranked],
        }


def _clamp01(x: float) -> float:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return 0.0
    return 0.0 if x < 0 else 1.0 if x > 1 else x


def adjudicate(
    question: str,
    hypotheses: list[Hypothesis],
    *,
    judge: Optional[JudgeFn] = None,
    w_conf: float = W_CONF,
    w_just: float = W_JUST,
) -> TournamentResult:
    """Score rival hypotheses and select a winner.

    combined = w_conf * deterministic_confidence + w_just * judge_quality.
    Deterministic confidence is the ungameable floor; the judge (if provided)
    scores justification quality — without a judge, quality defaults to the
    hypothesis's own confidence so the floor still ranks the field.

    HARD when: fewer than ceil(N/2)+... correct (weak field), OR the winner is a
    likely lucky guess (confidence high but justification quality low).
    """
    if not hypotheses:
        return TournamentResult(question=question)
    for h in hypotheses:
        h.confidence = _clamp01(h.confidence)
        if judge is not None:
            try:
                h.judge_quality = _clamp01(judge(question, h.answer, h.justification))
            except Exception:  # noqa: BLE001
                h.judge_quality = h.confidence  # judge failed: fall back to floor
        else:
            h.judge_quality = h.confidence
        h.score = round(w_conf * h.confidence + w_just * h.judge_quality, 4)
    ranked = sorted(hypotheses, key=lambda x: x.score, reverse=True)
    winner = ranked[0]
    is_hard, reason = _classify_hard(ranked, winner)
    return TournamentResult(
        question=question, ranked=ranked, winner=winner,
        is_hard=is_hard, hard_reason=reason,
    )


def _classify_hard(ranked: list[Hypothesis], winner: Hypothesis) -> tuple[bool, str]:
    """Flag hard questions: weak agreement across the field, or a lucky-guess
    winner (high deterministic confidence but low justification quality)."""
    graded = [h for h in ranked if h.correct is not None]
    if graded:
        n_correct = sum(1 for h in graded if h.correct)
        # weak field: strictly fewer than a majority of graded attempts correct
        if n_correct < (len(graded) + 1) // 2 + (1 if len(graded) >= 4 else 0):
            # for N>=4 require >=3 correct (the 3-of-4 rule); else simple majority
            need = 3 if len(graded) >= 4 else (len(graded) + 1) // 2
            if n_correct < need:
                return True, f"weak field: {n_correct}/{len(graded)} correct"
    if winner.confidence >= LUCKY_CONF_HI and winner.judge_quality <= LUCKY_JUST_LO:
        return True, (f"lucky guess: winner confidence {winner.confidence:.2f} but "
                      f"justification quality {winner.judge_quality:.2f}")
    return False, ""
