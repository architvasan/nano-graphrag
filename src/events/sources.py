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
    "WebSearchSource",
    "make_pathfind_episode",
    "make_websearch_episode",
    "make_subproblem_episode",
]

# An LLM callable: prompt -> completion string. Injected so the surface is not
# bound to any one provider (rule: model is a string expert, swappable).
LLMFn = Callable[[str], str]


def _dedupe(items: list[str]) -> list[str]:
    """Drop near-duplicate subproblems (case/whitespace-insensitive), keep
    first-seen order. Guards against the LLM emitting the same aspect twice,
    which would waste a walk and inflate redundant fan-up."""
    seen: set[str] = set()
    out: list[str] = []
    for it in items:
        norm = " ".join(it.lower().split())
        if norm and norm not in seen:
            seen.add(norm)
            out.append(it)
    return out


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
    judgement (active=False) rather than a fake zero-yield. The fact's
    deterministic provenance confidence rides in the note as ``conf=<0-1>`` so
    the summary layer can aggregate an episode confidence without touching the
    frozen CreditResult schema."""
    ident = (extracted.identity or "").strip()
    if not ident:
        return CreditResult.disabled("hop reached no usable node identity")
    return CreditResult(credits=(ident,), note=f"conf={extracted.confidence:.4f}")


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
        # Traverse any overlay nodes a prior web-search walk linked into the
        # subgraph this run: they are now part of the graph structure the walk
        # sees ('linked back into the graph at the same time'). Recurrence
        # against the base frontier collapses in the numerical layer.
        seen = {f.identity for f in self._frontier}
        for e in getattr(self._bridge, "overlay", []):
            if e.web_identity not in seen:
                self._frontier.append(
                    GraphFact(
                        identity=e.web_identity,
                        text=e.text,
                        source="web_fill",
                        paper_id=e.paper_id,
                        linked_to=e.base_node,
                        link_score=e.score,
                    )
                )
                seen.add(e.web_identity)
        if not self._seed_node and self._frontier:
            self._seed_node = self._frontier[0].identity

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
# WebSearchSource — a spawnable sibling of the graph walk (same PATHFIND grain)
# --------------------------------------------------------------------------
class WebSearchSource:
    """Yields web-search hops for one PATHFIND episode, linking each result
    into the graph as it goes.

    This is the spawnable web-search child: at the same grain as the graph
    walk, so a subproblem can dispatch both. Each hit is embedded, attached by
    cosine to the nearest node of the subproblem's retrieved subgraph (an
    overlay edge on the bridge), and served as one hop. The overlay makes these
    nodes traversable by any subsequent graph walk in the same subproblem —
    'linked back into the graph structure at the same time' — without mutating
    the base KG.
    """

    def __init__(
        self,
        bridge: KGBridge,
        subproblem_text: str,
        *,
        limit: int = 4,
        anchor_k: int = 6,
        min_link_score: float = 0.0,
    ) -> None:
        self._bridge = bridge
        self._text = subproblem_text
        self._limit = limit
        self._anchor_k = anchor_k
        self._min_link_score = min_link_score
        self._facts: list[GraphFact] = []
        self._served = 0
        self._loaded = False

    def _load(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        # anchor the web results to the retrieved subgraph (nodes to link into)
        try:
            anchors = [f.identity for f in self._bridge.grounded_facts(self._text, k=self._anchor_k)]
        except Exception:  # noqa: BLE001
            anchors = []
        try:
            self._facts = self._bridge.web_facts(
                self._text,
                limit=self._limit,
                link_nodes=anchors,
                min_link_score=self._min_link_score,
            )
        except Exception:  # noqa: BLE001
            self._facts = []

    def next(self, view: EpisodeView) -> Any:
        self._load()
        if not self._facts:
            return SourceEnd(kind="source_failed", reason="web_search_no_results")
        if self._served >= len(self._facts):
            return None  # exhausted: consumed all web hits
        fact = self._facts[self._served]
        self._served += 1
        return _hop_leaf(fact, self._served)


def make_websearch_episode(
    ctx: Context,
    bridge: KGBridge,
    subproblem_text: str,
    key: str,
    **kwargs: Any,
) -> Episode:
    """Build one web-search PATHFIND child Episode (sibling of the graph walk)."""
    return Episode(
        grain=PATHFIND_GRAIN,
        key=key,
        source=WebSearchSource(bridge, subproblem_text, **kwargs),
    )


# --------------------------------------------------------------------------
# SubproblemSource — yields PATHFIND episodes for one domain thread
# --------------------------------------------------------------------------
class SubproblemSource:
    """Yields PATHFIND episodes for one subproblem.

    Locates the subproblem's community, then dispatches walks of two kinds at
    the same grain: a graph walk, and (optionally) a spawnable web-search walk.
    The web walk runs first when the graph frontier looks off-target so its
    results are linked into the overlay before the graph walk traverses it.
    Communication is UP only: each pathfind's reached identities fan up to this
    subproblem's credits.
    """

    def __init__(
        self,
        ctx: Context,
        bridge: KGBridge,
        subproblem_text: str,
        *,
        max_walks: int = 2,
        spawn_web: bool = True,
        web_relevance_gate: float = 0.30,
    ) -> None:
        self._ctx = ctx
        self._bridge = bridge
        self._text = subproblem_text
        self._max_walks = max_walks
        self._spawn_web = spawn_web
        self._web_gate = web_relevance_gate
        self._issued = 0
        self._coarse: Optional[int] = None
        self._located = False
        self._plan: list[str] = []  # ordered walk kinds: "web" | "graph"

    def _locate(self) -> None:
        if self._located:
            return
        self._located = True
        try:
            self._coarse, _seeds = self._bridge.locate(self._text)
        except Exception:  # noqa: BLE001
            self._coarse = None
        self._plan = self._make_plan()

    def _make_plan(self) -> list[str]:
        """Decide walk order. Web search is spawned as a child when the graph
        frontier is thin OR off-target (low top-fact relevance to the
        subproblem) — the relevance-gated trigger. Otherwise graph only."""
        kinds: list[str] = []
        if self._spawn_web and self._should_spawn_web():
            kinds.append("web")   # web first: seeds the overlay for the graph walk
        while len(kinds) < self._max_walks:
            kinds.append("graph")
        return kinds[: self._max_walks]

    def _should_spawn_web(self) -> bool:
        """Relevance gate: spawn web search if the graph frontier is empty or
        its best fact embeds far from the subproblem (off-target coverage)."""
        try:
            facts = self._bridge.grounded_facts(self._text, k=4)
        except Exception:  # noqa: BLE001
            return True  # no graph => web is the only hope
        if not facts:
            return True
        try:
            qv = self._bridge.embed(self._text)
            fv = self._bridge.embed(facts[0].text[:400])
            top = self._bridge._cosine(qv, fv)
            return top < self._web_gate  # off-target => spawn web
        except Exception:  # noqa: BLE001
            return False

    def next(self, view: EpisodeView) -> Any:
        self._locate()
        if self._issued >= len(self._plan):
            return None  # exhausted: no more walks planned for this subproblem
        kind = self._plan[self._issued]
        self._issued += 1
        key = f"{kind}{self._issued}"
        if kind == "web":
            return make_websearch_episode(self._ctx, self._bridge, self._text, key)
        return make_pathfind_episode(self._ctx, self._bridge, self._text, key)


def make_subproblem_episode(
    ctx: Context,
    bridge: KGBridge,
    subproblem_text: str,
    key: str,
    *,
    max_walks: int = 2,
    spawn_web: bool = True,
) -> Episode:
    """Build one SUBPROBLEM child Episode (one domain thread)."""
    return Episode(
        grain=SUBPROBLEM_GRAIN,
        key=key,
        source=SubproblemSource(
            ctx, bridge, subproblem_text, max_walks=max_walks, spawn_web=spawn_web
        ),
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
                lines = [ln.strip("-* \t") for ln in raw.splitlines() if ln.strip()]
                subs = self._community_dedupe(_dedupe(lines))[: self._max]
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

    def _community_dedupe(self, subs: list[str]) -> list[str]:
        """Semantic dedupe: drop subproblems whose seed community is already
        claimed by an earlier one. Catches distinctly-worded aspects that embed
        to the same graph region (which a string dedupe misses) so each walk
        starts in a fresh part of the graph. Subproblems that fail to locate a
        community are kept (fail-open — never silently drop unlocatable work)."""
        seen: set[int] = set()
        out: list[str] = []
        for s in subs:
            try:
                coarse, _seeds = self._bridge.locate(s)
            except Exception:  # noqa: BLE001
                coarse = None
            if coarse is None or coarse not in seen:
                if coarse is not None:
                    seen.add(coarse)
                out.append(s)
        return out or subs  # never return empty

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
