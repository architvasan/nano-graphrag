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

    def _compose(
        self,
        question: str,
        record: EpisodeRecord,
        summaries: list[dict],
        summary_tree: Optional[Summary] = None,
    ) -> str:
        """Compose the final answer string from the distilled summary tree.

        This is the one model call at the root (rule 2: a string task). If no LLM
        is injected, return a deterministic evidence digest instead of fabricating
        an answer. Prefers the distilled per-subproblem summaries over raw
        identity strings when a summary tree is available."""
        identities = list(record.distinct_identities)
        if self.llm is None:
            top = ", ".join(identities[:12]) if identities else "(no identities reached)"
            return f"[no-LLM evidence digest] reached nodes: {top}"
        # Prefer distilled subproblem summaries (memory) over raw identities.
        blocks = []
        if summary_tree is not None and summary_tree.children:
            for c in summary_tree.children:
                digest = c.text or "; ".join(c.key_relations[:6]) or "(none)"
                blocks.append(f"- {c.scope_key} [{c.ended_by}]: {digest}")
        else:
            for s in summaries:
                ids = ", ".join(s["distinct_identities"][:10]) or "(none)"
                blocks.append(f"- {s['subproblem']} [{s['ended_by']}]: {ids}")
        joined = "\n".join(blocks) if blocks else "(no subproblem evidence)"
        prompt = (
            f"Question:\n{question}\n\n"
            f"Distilled findings across subproblems "
            f"(each line is one subproblem and what it established):\n"
            f"{joined}\n\n"
            f"Using only these findings, give the best-supported concise answer. "
            f"If the evidence is insufficient, say what is missing."
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
        )

    def _distiller(self):
        """Wrap the LLM as a Summary distiller: (text, relations) -> 1-2 sentences."""
        def _d(text: str, relations: list[str]) -> str:
            joined = "\n".join(f"- {r}" for r in relations[:8])
            prompt = (
                f"Topic: {text}\n\nEvidence relations reached:\n{joined}\n\n"
                "In 1-2 sentences, state what these relations establish about the "
                "topic. Use only the evidence; do not add outside facts."
            )
            return (self.llm(prompt) or "").strip()
        return _d
