"""The event-hierarchy HLE orchestrator.

Wires the root QUESTION Episode over the generic method_loop kernel, runs it,
and composes the final answer from the emitted EpisodeRecord tree. The root
Episode IS the orchestrator: its numerical verdict governs convergence; the LLM
only composes the answer string from the upward-fanned memory (rule 2).

    build_context() -> Context (grain order + frozen channel schemas)
    answer_question(question) -> AnswerResult (answer + the full record tree)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from method_loop import Context, Episode, EpisodeRecord

from .grains import CHANNEL_SCHEMAS, QUESTION_GRAIN, grain_order
from .kg_bridge import KGBridge
from .sources import LLMFn, QuestionSource
from .summary import Summary, summary_from_record
from .merge import merge_overlay_into_parallel
from .tournament import Hypothesis, TournamentResult, adjudicate

__all__ = ["AnswerResult", "EventOrchestrator", "build_context"]


def build_context(run_id: Optional[str] = None) -> Context:
    """Fresh Context declaring the grain order and frozen per-grain schemas."""
    return Context(
        channel_schemas=CHANNEL_SCHEMAS,
        order=grain_order(),
        run_id=run_id,
    )


@dataclass
class AnswerResult:
    """The orchestrator's output: the answer plus verifiable provenance."""

    question: str
    answer: str
    n_subproblems: int
    n_distinct_identities: int
    record: Optional[EpisodeRecord] = None
    subproblem_summaries: list[dict] = field(default_factory=list)
    ended_by: str = ""
    graph_addition_proposal: list[dict] = field(default_factory=list)
    summary: Optional[Summary] = None
    confidence: float = 0.0
    merge_report: dict = field(default_factory=dict)

    def as_record(self) -> dict:
        return {
            "question": self.question,
            "answer": self.answer,
            "n_subproblems": self.n_subproblems,
            "n_distinct_identities": self.n_distinct_identities,
            "ended_by": self.ended_by,
            "confidence": self.confidence,
            "subproblems": self.subproblem_summaries,
            "graph_addition_proposal": self.graph_addition_proposal,
            "merge_report": self.merge_report,
            "summary": self.summary.as_record() if self.summary else None,
            "episode_record": self.record.as_record() if self.record else None,
        }


@dataclass
class EventOrchestrator:
    """Runs the event hierarchy for one question and composes the answer."""

    bridge: KGBridge
    llm: Optional[LLMFn] = None
    max_subproblems: int = 4
    max_walks: int = 2

    def _summaries(self, record: EpisodeRecord) -> list[dict]:
        """Read the child (subproblem) records — memory that fanned up."""
        out: list[dict] = []
        for unit in record.unit_records:
            child = unit.child
            if child is None:
                continue
            out.append(
                {
                    "subproblem": child.scope_key,
                    "ended_by": child.ended_by,
                    "distinct_identities": list(child.distinct_identities)[:20],
                    "n_identities": len(child.distinct_identities),
                }
            )
        return out

    def _evidence_for(self, identities: list[str], limit: int = 6) -> str:
        """Expand identities into their FULL evidence text (edge sentence / web
        abstract) from the bridge registry — the reasoner grounds on documents,
        not bare node/edge names. Falls back to the identity when no evidence was
        recorded (never drops the relation silently)."""
        ev = getattr(self.bridge, "evidence", {})
        lines = []
        for ident in identities[:limit]:
            full = ev.get(ident, "")
            lines.append(f"  • {full}" if full else f"  • {ident}")
        return "\n".join(lines)

    def _compose(
        self,
        question: str,
        record: EpisodeRecord,
        summaries: list[dict],
        summary_tree: Optional[Summary] = None,
    ) -> str:
        """Compose the final answer from full grounded evidence text.

        This is the one model call at the root (rule 2: a string task). The
        reasoner is given each subproblem's FULL evidence documents (edge
        sentences and web abstracts), not bare identities, plus its distilled
        summary and confidence. No LLM -> deterministic evidence digest."""
        identities = list(record.distinct_identities)
        if self.llm is None:
            top = self._evidence_for(identities, limit=8) or "(no evidence reached)"
            return f"[no-LLM evidence digest]\n{top}"
        blocks = []
        if summary_tree is not None and summary_tree.children:
            for c in summary_tree.children:
                evidence = self._evidence_for(c.key_relations, limit=6)
                digest = c.text or "(no distilled summary)"
                blocks.append(
                    f"- {c.scope_key} [{c.ended_by}, confidence={c.confidence:.2f}]:\n"
                    f"  summary: {digest}\n  evidence:\n{evidence}"
                )
        else:
            for s in summaries:
                blocks.append(
                    f"- {s['subproblem']} [{s['ended_by']}]:\n"
                    f"{self._evidence_for(s['distinct_identities'], limit=6)}"
                )
        joined = "\n".join(blocks) if blocks else "(no subproblem evidence)"
        prompt = (
            f"Question:\n{question}\n\n"
            f"Evidence gathered across subproblems (each bullet is the full "
            f"supporting text of one graph edge or web source):\n"
            f"{joined}\n\n"
            f"Reason step by step over the EVIDENCE TEXT above (not just the "
            f"entity names) to give the best-supported concise answer. If the "
            f"evidence is insufficient, say what is missing."
        )
        try:
            return (self.llm(prompt) or "").strip()
        except Exception as exc:  # noqa: BLE001
            return f"[answer composition failed: {type(exc).__name__}]"

    def answer_question(self, question: str, key: str = "q") -> AnswerResult:
        ctx = build_context()
        source = QuestionSource(
            ctx,
            self.bridge,
            question,
            self.llm,
            max_subproblems=self.max_subproblems,
            max_walks=self.max_walks,
        )
        root = Episode(grain=QUESTION_GRAIN, key=key, source=source)
        record = root.run(ctx)  # the kernel owns the loop; verdict is numerical
        summaries = self._summaries(record)
        # distilled-memory tree (parallel to the numerical credit channel)
        distill = self._distiller() if self.llm else None
        summary_tree = summary_from_record(record, question=question, distill=distill)
        answer = self._compose(question, record, summaries, summary_tree)
        # gated merge of web-rescued edges into a parallel copy (base KG untouched)
        parallel_ev, mreport = merge_overlay_into_parallel(self.bridge)
        if parallel_ev is not None:
            self.bridge.parallel_edge_ev = parallel_ev  # available for a retry pass
        return AnswerResult(
            question=question,
            answer=answer,
            n_subproblems=len(summaries),
            n_distinct_identities=len(record.distinct_identities),
            record=record,
            subproblem_summaries=summaries,
            ended_by=record.ended_by,
            graph_addition_proposal=self.bridge.overlay_proposal(),
            summary=summary_tree,
            confidence=summary_tree.confidence if summary_tree else 0.0,
            merge_report=mreport.as_record(),
        )

    def _distiller(self):
        """Wrap the LLM as a Summary distiller over FULL evidence text.

        Relations arrive as identities; expand each to its recorded evidence
        (edge sentence / web abstract) so the distillation grounds on documents,
        not entity names — the reasoner must see the actual text."""
        def _d(text: str, relations: list[str]) -> str:
            ev = getattr(self.bridge, "evidence", {})
            joined = "\n".join(f"- {ev.get(r, r)}" for r in relations[:8])
            prompt = (
                f"Topic: {text}\n\nEvidence relations reached:\n{joined}\n\n"
                "In 1-2 sentences, state what these relations establish about the "
                "topic. Use only the evidence; do not add outside facts."
            )
            return (self.llm(prompt) or "").strip()
        return _d

    # -- tournament mode --------------------------------------------------
    def tournament_answer(
        self, question: str, key: str = "q", *, n_hypotheses: int = 4,
    ) -> tuple[AnswerResult, TournamentResult]:
        """Run N independent rival attempts and let the parent adjudicate.

        Each attempt is a full, independent answer_question pass (its own episode
        tree, escalation, web rescue), yielding an answer + justification + the
        DETERMINISTIC confidence. adjudicate() then scores them by confidence
        (ungameable floor) + an LLM justification-quality judge, selects a winner,
        and flags HARD (weak field / lucky guess). Returns the WINNER's full
        AnswerResult plus the TournamentResult. Divergence across attempts comes
        from the stochastic web/graph escalation ordering; identical attempts
        simply collapse to one effective hypothesis (honest, not theater)."""
        attempts: list[tuple[AnswerResult, Hypothesis]] = []
        for i in range(max(1, n_hypotheses)):
            res = self.answer_question(question, key=f"{key}-h{i}")
            hyp = Hypothesis(
                label=f"h{i}",
                answer=res.answer,
                justification=self._justification(res),
                confidence=res.confidence,
            )
            attempts.append((res, hyp))
        tourn = adjudicate(question, [h for _, h in attempts], judge=self._judge())
        winner_res = next(
            (r for r, h in attempts if tourn.winner and h.label == tourn.winner.label),
            attempts[0][0],
        )
        return winner_res, tourn

    def _justification(self, res: AnswerResult) -> str:
        """The winner's reasoning trace: distilled subproblem summaries + the
        confidence they carried — what the answer is grounded on."""
        if res.summary is None:
            return res.answer
        parts = [
            f"{c.scope_key} (conf {c.confidence:.2f}): {c.text or '; '.join(c.key_relations[:3])}"
            for c in res.summary.children
        ]
        return " | ".join(parts) or res.answer

    def _judge(self):
        """LLM justification-quality judge: (question, answer, justification) ->
        0-1. Rates how well the reasoning supports the answer — the tiebreaker
        that catches lucky guesses. No LLM -> None (adjudicate uses the floor)."""
        if self.llm is None:
            return None
        llm = self.llm

        def _j(question: str, answer: str, justification: str) -> float:
            prompt = (
                f"Question:\n{question}\n\nProposed answer:\n{answer}\n\n"
                f"Reasoning offered:\n{justification}\n\n"
                "Rate ONLY how well the reasoning supports the answer with concrete, "
                "on-topic evidence (not confident phrasing). Reply with a single "
                "number 0.0-1.0 and nothing else."
            )
            try:
                raw = (llm(prompt) or "").strip()
                import re
                m = re.search(r"[01](?:\.\d+)?", raw)
                return float(m.group()) if m else 0.0
            except Exception:  # noqa: BLE001
                return 0.0
        return _j
