"""UnitSources for the event-hierarchy HLE orchestrator.

A source answers one question only: "what is the next unit for this episode?"
(charter rule 1 — no loop is written here; only ``Episode.run`` loops). Each
source returns an ``Acquirable`` (a ``Leaf`` or a child ``Episode``), ``None``
for exhaustion, or a ``SourceEnd`` naming why the stream died.

The model lives ONLY inside a source's proposal step or a leaf's ``extract``
(rule 2). Credit is a deterministic projection (``credit`` callables here never
call a model). Verdicts are the numerical controller's; no source decides
continue/stop.

    QuestionSource    -> yields SUBPROBLEM Episodes (decompose; or N graph locations)
    SubproblemSource  -> yields PATHFIND Episodes  (locate community, seed a walk)
    FrontierSource    -> yields hop Leaves         (graph hop or web-fill node)
"""
from __future__ import annotations

from typing import Any, Callable, Optional

from method_loop import (
    Context,
    CreditResult,
    Episode,
    EpisodeView,
    Leaf,
    SourceEnd,
)

from .grains import PATHFIND_GRAIN, SUBPROBLEM_GRAIN
from .kg_bridge import GraphFact, KGBridge

__all__ = [
    "QuestionSource",
    "SubproblemSource",
    "FrontierSource",
    "make_pathfind_episode",
    "make_subproblem_episode",
]

# An LLM callable: prompt -> completion string. Injected so the surface is not
# bound to any one provider (rule: model is a string expert, swappable).
LLMFn = Callable[[str], str]


# --------------------------------------------------------------------------
# leaf parts (extract + credit) for a graph hop / web-fill node
# --------------------------------------------------------------------------
def _fact_extract(fact: GraphFact) -> GraphFact:
    """Read the unit into observations. Identity handoff: the fact already
    carries its text and identity, so extract is the identity projection."""
    return fact


def _fact_credit(fact: GraphFact, extracted: GraphFact) -> CreditResult:
    """Deterministic projection: the node identity this hop reached. No model.

    A hop that reached nothing usable (empty identity) makes no crediting
    judgement (active=False) rather than a fake zero-yield."""
    ident = (extracted.identity or "").strip()
    if not ident:
        return CreditResult.disabled("hop reached no usable node identity")
    return CreditResult(credits=(ident,))


def _hop_leaf(fact: GraphFact, index: int) -> Leaf:
    return Leaf(
        unit=fact,
        extract=_fact_extract,
        credit=_fact_credit,
        label=f"{fact.source}:{fact.identity[:60]}",
    )


# --------------------------------------------------------------------------
# FrontierSource — the innermost loop: hop toward an event-end, web-fill gaps
# --------------------------------------------------------------------------
class FrontierSource:
    """Yields the next hop for one PATHFIND episode.

    Precomputes the grounded frontier for the subproblem (KG PPR + evidence),
    then serves one hop at a time. When the graph frontier is thin, it web-fills
    a missing node and links it to the retrieved subgraph. Ends with yield_stop
    when the target answer region is reached, exhausted when the frontier empties.
    """

    def __init__(
        self,
        bridge: KGBridge,
        subproblem_text: str,
        *,
        k: int = 8,
        min_graph_facts: int = 3,
        mechanism_only: bool = True,
    ) -> None:
        self._bridge = bridge
        self._text = subproblem_text
        self._min_graph_facts = min_graph_facts
        self._frontier: list[GraphFact] = []
        self._served = 0
        self._loaded = False
        self._k = k
        self._mechanism_only = mechanism_only
        self._seed_node = ""

    def _load(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        try:
            self._frontier = self._bridge.grounded_facts(
                self._text, k=self._k, mechanism_only=self._mechanism_only
            )
        except Exception:  # noqa: BLE001 — KG unavailable is a source failure, not a crash
            self._frontier = []
        if self._frontier:
            self._seed_node = self._frontier[0].identity
        # Web-fill when the graph gave too few nodes: propose missing nodes
        # linked to the subgraph seed. Bounded top-up, not an open loop.
        if len(self._frontier) < self._min_graph_facts:
            need = self._min_graph_facts - len(self._frontier)
            for _ in range(need):
                wf = self._bridge.web_fill(self._text, link_to=self._seed_node)
                if wf is None:
                    break
                self._frontier.append(wf)

    def next(self, view: EpisodeView) -> Any:
        self._load()
        if not self._frontier:
            return SourceEnd(kind="source_failed", reason="empty_frontier_no_web_fill")
        if self._served >= len(self._frontier):
            # Reached the end of the reachable path: this is the event-end.
            return None  # exhausted — the only spelling of exhaustion
        fact = self._frontier[self._served]
        leaf = _hop_leaf(fact, self._served)
        self._served += 1
        return leaf


def make_pathfind_episode(
    ctx: Context,
    bridge: KGBridge,
    subproblem_text: str,
    key: str,
    **frontier_kwargs: Any,
) -> Episode:
    """Build one PATHFIND child Episode (a walk toward an event-end)."""
    return Episode(
        grain=PATHFIND_GRAIN,
        key=key,
        source=FrontierSource(bridge, subproblem_text, **frontier_kwargs),
    )


# --------------------------------------------------------------------------
# SubproblemSource — yields PATHFIND episodes for one domain thread
# --------------------------------------------------------------------------
class SubproblemSource:
    """Yields PATHFIND episodes for one subproblem.

    Locates the subproblem's community (embed -> seeds -> community vote) and
    seeds walks from there. Serves a bounded number of walks; the numerical
    verdict decides when to stop before the bound. Communication is UP only:
    each pathfind's reached identities fan up to this subproblem's credits.
    """

    def __init__(
        self,
        ctx: Context,
        bridge: KGBridge,
        subproblem_text: str,
        *,
        max_walks: int = 2,
    ) -> None:
        self._ctx = ctx
        self._bridge = bridge
        self._text = subproblem_text
        self._max_walks = max_walks
        self._issued = 0
        self._coarse: Optional[int] = None
        self._located = False

    def _locate(self) -> None:
        if self._located:
            return
        self._located = True
        try:
            self._coarse, _seeds = self._bridge.locate(self._text)
        except Exception:  # noqa: BLE001
            self._coarse = None

    def next(self, view: EpisodeView) -> Any:
        self._locate()
        if self._issued >= self._max_walks:
            return None  # exhausted: no more walks planned for this subproblem
        self._issued += 1
        key = f"walk{self._issued}"
        return make_pathfind_episode(self._ctx, self._bridge, self._text, key)


def make_subproblem_episode(
    ctx: Context,
    bridge: KGBridge,
    subproblem_text: str,
    key: str,
    *,
    max_walks: int = 2,
) -> Episode:
    """Build one SUBPROBLEM child Episode (one domain thread)."""
    return Episode(
        grain=SUBPROBLEM_GRAIN,
        key=key,
        source=SubproblemSource(ctx, bridge, subproblem_text, max_walks=max_walks),
    )


# --------------------------------------------------------------------------
# QuestionSource — the root: decompose the question into subproblem episodes
# --------------------------------------------------------------------------
class QuestionSource:
    """Yields one SUBPROBLEM episode per subproblem of the HLE question.

    The LLM decomposes the question into domain subproblems (a string task —
    the one model call this source makes). If it cannot decompose, it falls back
    to N parallel graph LOCATIONS (distinct seed communities), realizing the
    "parallelize in different places and enforce communication up" path. Either
    way, children communicate only upward through fan-up.
    """

    def __init__(
        self,
        ctx: Context,
        bridge: KGBridge,
        question: str,
        llm: Optional[LLMFn] = None,
        *,
        max_subproblems: int = 4,
        max_walks: int = 2,
    ) -> None:
        self._ctx = ctx
        self._bridge = bridge
        self._question = question
        self._llm = llm
        self._max = max_subproblems
        self._max_walks = max_walks
        self._subproblems: list[str] = []
        self._issued = 0
        self._planned = False

    def _decompose(self) -> None:
        if self._planned:
            return
        self._planned = True
        subs: list[str] = []
        if self._llm is not None:
            prompt = (
                "Break this exam question into 1-4 focused sub-problems, one per "
                "line, no numbering:\n\n" + self._question
            )
            try:
                raw = self._llm(prompt) or ""
                subs = [ln.strip("-* \t") for ln in raw.splitlines() if ln.strip()][: self._max]
            except Exception:  # noqa: BLE001
                subs = []
        if not subs:
            # NODECOMP fallback: N parallel graph locations from the question's seeds.
            try:
                _coarse, seeds = self._bridge.locate(self._question, m=self._max * 3)
            except Exception:  # noqa: BLE001
                seeds = []
            seen: list[str] = []
            for s in seeds:
                c = self._bridge.community_of(s)
                tag = self._bridge.community_label(c) if c is not None else s
                if tag not in seen:
                    seen.append(tag)
                    subs.append(f"{self._question}  [aspect near: {tag}]")
                if len(subs) >= self._max:
                    break
            if not subs:
                subs = [self._question]
        self._subproblems = subs

    def next(self, view: EpisodeView) -> Any:
        self._decompose()
        if self._issued >= len(self._subproblems):
            return None  # exhausted: all subproblems dispatched
        text = self._subproblems[self._issued]
        self._issued += 1
        key = f"sub{self._issued}"
        return make_subproblem_episode(
            self._ctx, self._bridge, text, key, max_walks=self._max_walks
        )
