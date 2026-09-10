"""Event-hierarchy HLE orchestrator — a surface binding of the method_loop Episode.

Answers a question by decomposing it into subproblems (event hierarchy), running
graph pathfinding toward answer regions with web-fill of missing nodes, fanning
distinct identities upward through the scope tree, and composing an answer at the
root. Decisions are numerical (rarefaction verdict); the LLM does string tasks.

See DESIGN.md for the full algorithm and the grain nesting.
"""
from __future__ import annotations

from .grains import (
    CHANNEL_SCHEMAS,
    PATHFIND_GRAIN,
    QUESTION_GRAIN,
    SUBPROBLEM_GRAIN,
    grain_order,
)
from .kg_bridge import GraphFact, KGBridge, KGUnavailable, OverlayEdge
from .orchestrator import AnswerResult, EventOrchestrator, build_context
from .sources import (
    FrontierSource,
    QuestionSource,
    SubproblemSource,
    WebSearchSource,
)
from .summary import Summary, SummaryBus, summary_from_record

__all__ = [
    "CHANNEL_SCHEMAS",
    "PATHFIND_GRAIN",
    "QUESTION_GRAIN",
    "SUBPROBLEM_GRAIN",
    "grain_order",
    "GraphFact",
    "OverlayEdge",
    "KGBridge",
    "KGUnavailable",
    "AnswerResult",
    "EventOrchestrator",
    "build_context",
    "FrontierSource",
    "QuestionSource",
    "SubproblemSource",
    "WebSearchSource",
    "Summary",
    "SummaryBus",
    "summary_from_record",
]
