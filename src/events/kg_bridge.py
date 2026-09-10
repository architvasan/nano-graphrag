"""KGBridge — adapts the ragmosis KG tooling for the event-hierarchy surface.

This module is the ONLY point that touches the external KG stack (embeddings,
PPR graph retrieval, the community hierarchy, web-fill). Keeping it isolated
preserves the method_loop layering rule: the Episode core stays schema-agnostic
and never imports an application KG. See src/events/DESIGN.md.

The KG stack lives in a separate project (kg-memory-system). Paths are resolved
from env vars with sensible defaults, and every capability degrades gracefully:
if the tooling or artifacts are absent, the bridge reports it rather than
fabricating retrieval. Nothing here decides continue/stop (that is the numerical
controller's job); the bridge only supplies string/graph material.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Any, Optional

from .confidence import fact_confidence

__all__ = ["KGBridge", "GraphFact", "OverlayEdge", "Subgraph", "KGUnavailable"]

# Default locations of the KG project; override with env vars.
_KG_ROOT = os.environ.get(
    "KG_MEMORY_ROOT",
    os.path.expanduser("~/Desktop/Projects/kg-memory-system"),
)
_KG_SRC = os.path.join(_KG_ROOT, "src")
_KG_EXPT = os.path.join(_KG_ROOT, "experiments", "provenance_threshold")
_COMMUNITY_JSON = os.environ.get(
    "KG_COMMUNITY_GRAPH",
    os.path.join(_KG_EXPT, "results", "community_graph.json"),
)
_COARSE_POS = 3  # community-path position of the labeled top-domain communities


class KGUnavailable(RuntimeError):
    """Raised when a requested KG capability's tooling/artifacts are absent."""


def _is_stub_ev(ev: str) -> bool:
    """Real-evidence test mirroring GraphRetriever._is_stub: empty, id-stub, or
    too short to carry a mechanism. Used to keep subgraph facts contentful."""
    e = (ev or "").strip()
    return (not e) or len(e) < 25


#: co-mention/association predicates too generic to carry a mechanism. The base
#: retriever excludes co_occurs_with upstream; subgraph facts must drop the whole
#: family so a query-local slice never grounds reasoning on co-mention hubs.
_GENERIC_RELS = frozenset({"co_occurs_with", "co_occurs", "co-mention", "comention",
                           "mentioned_with", "associated_with", "related_to"})


@dataclass(frozen=True)
class GraphFact:
    """One retrieved claim: a typed edge with evidence, or a web-fill node."""

    identity: str          # opaque node/edge identity (what fans up as credit)
    text: str              # human-readable claim/triple/evidence
    source: str            # "graph" | "web_fill"
    paper_id: str = ""
    linked_to: str = ""    # for web_fill: the subgraph node it was attached to
    link_score: float = 0.0  # cosine of the web node to its linked subgraph node
    confidence: float = 0.0  # deterministic provenance-weighted 0-1 confidence


@dataclass
class OverlayEdge:
    """A run-local proposed edge: a web node attached to a base-graph node.

    Overlay edges live only for this run (never written to the base KG). They
    make web-discovered nodes traversable by later graph walks in the same
    subproblem, and together form the reviewable graph-addition proposal a
    completed run may emit (charter: acquisition never mutates the graph)."""

    web_identity: str
    base_node: str
    score: float
    text: str
    paper_id: str = ""

    def as_record(self) -> dict:
        return {
            "web_identity": self.web_identity,
            "base_node": self.base_node,
            "score": round(self.score, 4),
            "text": self.text,
            "paper_id": self.paper_id,
        }


@dataclass
class Subgraph:
    """A query-local slice of the megagraph (~budget nodes), extracted once per
    question so all downstream walks/reasoning run on a small, focused graph.

    Holds the induced node set, the adjacency among them, and the evidence
    side-table restricted to internal edges. The full megagraph is touched only
    to build this; everything after reads from here."""

    query: str
    nodes: tuple[str, ...]                    # PPR-ranked, best first
    adjacency: dict                           # node -> set(neighbor nodes) within the slice
    edge_ev: dict                             # frozenset({h,t}) -> [ {rel,ev,pid} ] (subset)
    node_rank: dict = field(default_factory=dict)  # node -> PPR rank (lower=closer)

    @property
    def n_nodes(self) -> int:
        return len(self.nodes)

    @property
    def n_edges(self) -> int:
        return len(self.edge_ev)

    def __contains__(self, node: str) -> bool:
        return node in self.adjacency

    def as_record(self) -> dict:
        return {"query": self.query[:80], "n_nodes": self.n_nodes,
                "n_edges": self.n_edges, "top_nodes": list(self.nodes[:10])}


@dataclass
class KGBridge:
    """Lazily-loaded adapter over embeddings + PPR retriever + community tree."""

    kg_src: str = _KG_SRC
    kg_expt: str = _KG_EXPT
    community_json: str = _COMMUNITY_JSON
    gate: str = "forman"
    pct: float = 0.3
    #: Injectable web backend: query -> list of {"url","title","description"}.
    #: Defaults to hermes_tools inside the Hermes runtime; inject a callable
    #: (e.g. ragmosis OpenAlex/Semantic-Scholar rescue) to run standalone.
    web_search_fn: Any = None
    #: chars of page body pulled per web source. Abstracts alone (~200 chars)
    #: lack quantitative rules; deeper text lets the reasoner ground on results.
    web_body_chars: int = 2500
    #: on-disk evidence-vector cache (content-hashed, 3072d text-embedding-3-large).
    evidence_cache_path: str = os.path.join(
        os.path.expanduser("~/.hermes/kg"), ".evidence_vectors_large.npz")

    _embed: Any = field(default=None, init=False, repr=False)
    _retriever: Any = field(default=None, init=False, repr=False)
    _hierarchy: Any = field(default=None, init=False, repr=False)
    _name2comm: dict = field(default_factory=dict, init=False, repr=False)
    _coarse_label: dict = field(default_factory=dict, init=False, repr=False)
    #: identity -> full evidence text (graph edge sentence / web abstract). The
    #: reasoner must ground on this, not on bare node/edge identities.
    evidence: dict = field(default_factory=dict, init=False, repr=False)
    #: parallel copy of edge_ev after a gated web-edge merge (base KG untouched).
    parallel_edge_ev: Any = field(default=None, init=False, repr=False)
    #: when set (by the orchestrator, once per question), grounded_facts reads
    #: from this ~500-node slice instead of the full megagraph — two-stage
    #: retrieval, so every walk runs on the small graph.
    active_subgraph: Any = field(default=None, init=False, repr=False)
    overlay: list = field(default_factory=list, init=False, repr=False)
    #: content-hashed evidence-vector cache: md5(ev_text) -> unit vec (3072d,
    #: text-embedding-3-large, matching the query space). Loaded lazily from
    #: evidence_cache_path; misses embed live and write back (self-healing).
    _ev_vecs: dict = field(default_factory=dict, init=False, repr=False)
    _ev_cache_loaded: bool = field(default=False, init=False, repr=False)
    _ev_cache_dirty: bool = field(default=False, init=False, repr=False)

    # -- path wiring -------------------------------------------------------
    def _ensure_path(self) -> None:
        for p in (self.kg_src, self.kg_expt):
            if p not in sys.path and os.path.isdir(p):
                sys.path.insert(0, p)

    # -- embeddings --------------------------------------------------------
    def embed(self, text: str):
        """Unit-normalized query embedding (argo:text-embedding-3-large, 3072d)."""
        if self._embed is None:
            self._ensure_path()
            try:
                from ragmosis.inference.argo_embed import embed as _e
            except Exception as exc:  # noqa: BLE001
                raise KGUnavailable(f"argo_embed not importable: {exc}") from exc
            self._embed = _e
        vecs = self._embed([text])
        return vecs[0] if hasattr(vecs, "__getitem__") else vecs

    def embed_many(self, texts: list[str]):
        """Batched embeddings — ONE argo call for a list, instead of N calls.
        Critical for subgraph facts: scoring ~500 edges one-embed-at-a-time was
        247 round-trips (~147s); batched it is a single call (~2s). Returns a
        list of vectors aligned with ``texts`` (empty input -> empty list)."""
        if not texts:
            return []
        if self._embed is None:
            self._ensure_path()
            try:
                from ragmosis.inference.argo_embed import embed as _e
            except Exception as exc:  # noqa: BLE001
                raise KGUnavailable(f"argo_embed not importable: {exc}") from exc
            self._embed = _e
        return self._embed(texts)

    # -- evidence-vector cache (content-hashed, build once, look up forever) ---
    @staticmethod
    def _ev_key(ev: str) -> str:
        import hashlib
        return hashlib.md5((ev or "").strip().encode()).hexdigest()

    def _load_ev_cache(self) -> None:
        """Lazy-load the on-disk evidence-vector cache once. Absent file is fine
        (misses self-heal into it)."""
        if self._ev_cache_loaded:
            return
        self._ev_cache_loaded = True
        try:
            import numpy as np
            if os.path.exists(self.evidence_cache_path):
                d = np.load(self.evidence_cache_path, allow_pickle=True)
                keys, vecs = d["keys"], d["vecs"]
                self._ev_vecs = {str(k): vecs[i] for i, k in enumerate(keys)}
        except Exception:  # noqa: BLE001  (corrupt/absent cache -> start empty)
            self._ev_vecs = {}

    def evidence_vectors(self, evs: list[str]):
        """Return unit vectors for evidence strings, aligned with ``evs``. Hits the
        content-hashed cache; MISSES are batch-embedded once (text-embedding-3-large,
        the query space) and written back. Duplicate strings share one vector."""
        import numpy as np
        self._load_ev_cache()
        keys = [self._ev_key(e) for e in evs]
        miss = [e for e, k in zip(evs, keys) if k not in self._ev_vecs]
        if miss:
            uniq = list(dict.fromkeys(miss))            # dedupe before embedding
            vecs = self.embed_many(uniq)
            for text, v in zip(uniq, vecs):
                arr = np.asarray(v, dtype="float32")
                n = float(np.linalg.norm(arr))
                self._ev_vecs[self._ev_key(text)] = arr / n if n else arr
            self._ev_cache_dirty = True
        return [self._ev_vecs.get(k) for k in keys]

    def flush_ev_cache(self) -> bool:
        """Persist the evidence-vector cache if it grew. Returns True if written."""
        if not self._ev_cache_dirty or not self._ev_vecs:
            return False
        import numpy as np
        os.makedirs(os.path.dirname(self.evidence_cache_path), exist_ok=True)
        keys = list(self._ev_vecs)
        vecs = np.stack([self._ev_vecs[k] for k in keys]).astype("float32")
        tmp = self.evidence_cache_path + f".tmp.{os.getpid()}.npz"
        np.savez(tmp, keys=np.array(keys), vecs=vecs)
        os.replace(tmp, self.evidence_cache_path)
        self._ev_cache_dirty = False
        return True

    # -- PPR + evidence retriever -----------------------------------------
    @property
    def retriever(self):
        if self._retriever is None:
            self._ensure_path()
            try:
                import graph_retriever as GRmod
            except Exception as exc:  # noqa: BLE001
                raise KGUnavailable(f"graph_retriever not importable: {exc}") from exc
            self._retriever = GRmod.GraphRetriever(gate=self.gate, pct=self.pct)
        return self._retriever

    def seeds(self, text: str, m: int = 12) -> list[str]:
        """Top-m seed node names for a query by embedding cosine."""
        r = self.retriever
        qv = r._qv(text) if hasattr(r, "_qv") else self.embed(text)
        return list(r._seeds(qv, m=m)) if hasattr(r, "_seeds") else []

    def grounded_facts(self, text: str, k: int = 8, mechanism_only: bool = True) -> list[GraphFact]:
        """PPR + typed-edge evidence for a query, as GraphFact claims, each with
        a deterministic provenance-weighted confidence (relevance x prov tier).

        Two-stage: when an active_subgraph is set (extracted once per question),
        read from that ~500-node slice instead of re-querying the full megagraph."""
        if self.active_subgraph is not None:
            return self.facts_from_subgraph(self.active_subgraph, text=text, k=k,
                                            mechanism_only=mechanism_only)
        r = self.retriever
        try:
            raw = r.retrieve_grounded(
                text, k=k, with_abstracts=False,
                mechanism_only=mechanism_only,
            )
        except Exception as exc:  # noqa: BLE001
            raise KGUnavailable(f"retrieve_grounded failed: {exc}") from exc
        # query embedding for the relevance signal (best-effort; 1.0 if unavailable)
        try:
            qv = self.embed(text)
        except Exception:  # noqa: BLE001
            qv = None
        out: list[GraphFact] = []
        for t, s in raw:
            claim = str(t)
            paper_id = str(s)
            # The claim identity is the triple/head of the text (before the em-dash
            # evidence separator). retrieve_grounded returns (claim_text, paper_id).
            identity = claim.split("  —  ")[0].strip()[:120] if claim else ""
            if not identity:
                continue
            pid = paper_id if paper_id and paper_id.lower() != "none" else ""
            rel = 1.0
            if qv is not None:
                try:
                    rel = max(0.0, self._cosine(qv, self.embed(identity)))
                except Exception:  # noqa: BLE001
                    rel = 1.0
            conf = fact_confidence(source="graph", paper_id=pid, relevance=rel)
            self.evidence[identity] = claim  # full edge sentence, not just the head
            out.append(
                GraphFact(
                    identity=identity,
                    text=claim,
                    source="graph",
                    paper_id=pid,
                    confidence=conf,
                )
            )
        return out

    # -- subgraph extraction (two-stage retrieval) ------------------------
    def extract_subgraph(self, text: str, budget: int = 500,
                         edge_budget: Optional[int] = None,
                         mode: str = "global") -> Optional[Subgraph]:
        """Stage 1 of two-stage retrieval: cut a query-local ~budget-node slice
        out of the megagraph via PPR, so every downstream walk/reason runs on the
        small graph instead of the full 95k-edge one.

        1. PPR-rank nodes for the query (reusing the retriever's gated PPR).
        2. take top nodes until the node budget is met.
        3. induce the subgraph: keep edges among those nodes, carrying the
           evidence side-table subset; cap edges at ``edge_budget`` (defaults to
           ``budget``), dropping the lowest-trust first so the most reliable edges
           survive the cut.
        Returns None if the retriever/graph is unavailable."""
        edge_budget = budget if edge_budget is None else edge_budget
        try:
            r = self.retriever
        except KGUnavailable:
            return None
        # 1-2. PPR-ranked nodes up to the budget
        try:
            hits = r.retrieve(text, k=budget, mode=mode)  # [(node, rank, name)]
        except Exception as exc:  # noqa: BLE001
            raise KGUnavailable(f"subgraph PPR failed: {exc}") from exc
        nodes = [h[0] for h in hits][:budget]
        node_rank = {h[0]: float(h[1]) for h in hits[:budget]}
        nodeset = set(nodes)
        if not nodeset:
            return None
        G = getattr(r, "G", None)
        base_ev = getattr(r, "edge_ev", {}) or {}
        edge_trust = getattr(r, "edge_trust", {}) or {}
        # 3. induce: internal edges only, carry evidence, cap by trust
        adjacency: dict = {n: set() for n in nodes}
        internal: list[tuple] = []  # (key, trust)
        for n in nodes:
            nbrs = (G[n] if (G is not None and n in G) else [])
            for nbr in nbrs:
                if nbr not in nodeset or nbr == n:
                    continue
                adjacency[n].add(nbr)
                key = frozenset((n, nbr))
                if key in base_ev:
                    tr = edge_trust.get((n, nbr), edge_trust.get((nbr, n), 0.0))
                    internal.append((key, float(tr)))
        # dedupe keys keeping best trust, then cap at budget (highest trust wins)
        best: dict = {}
        for key, tr in internal:
            if key not in best or tr > best[key]:
                best[key] = tr
        kept = sorted(best.items(), key=lambda kv: kv[1], reverse=True)[:edge_budget]
        sub_ev = {key: base_ev[key] for key, _ in kept}
        return Subgraph(query=text, nodes=tuple(nodes), adjacency=adjacency,
                        edge_ev=sub_ev, node_rank=node_rank)

    def facts_from_subgraph(self, sub: Subgraph, text: str = "",
                            k: int = 8, mechanism_only: bool = True) -> list[GraphFact]:
        """GraphFacts drawn ONLY from a subgraph's edges — the small-graph analog
        of grounded_facts. Applies the same generic/stub filter and deterministic
        confidence, and records full evidence text. Ranks edges by the PPR rank of
        their closest endpoint so the most query-relevant claims come first.

        Embeddings: the query embeds live (one call); evidence vectors come from
        the content-hashed cache (evidence_vectors) — a hit is zero network, a miss
        batch-embeds once and writes back. This is the whole point of two-stage
        retrieval: the graph narrows WHERE to look, the cosine ranks WHAT is
        actually relevant, and evidence is embedded at most once ever."""
        onto = getattr(self.retriever, "ONTOLOGY_RELS", frozenset())
        generic = onto | _GENERIC_RELS
        # pass 1: collect surviving edges (rank, identity, claim, pid, ev-snippet)
        rows: list[tuple] = []
        for key, entries in sub.edge_ev.items():
            pair = tuple(key)
            rank = min((sub.node_rank.get(n, 1e9) for n in pair), default=1e9)
            for e in entries or []:
                rel = e.get("rel", "")
                ev = e.get("ev", e.get("evidence", ""))
                if mechanism_only and (rel in generic or _is_stub_ev(ev)):
                    continue
                head = " / ".join(str(n) for n in pair)
                identity = f"{head} [{rel}]"[:120]
                claim = f"{head}  —  {ev}"
                pid = str(e.get("pid", "") or "")
                rows.append((rank, identity, claim, pid, ev[:200]))
        if not rows:
            return []
        # pass 2: query embeds live; evidence vectors come from the cache
        rel_scores = [1.0] * len(rows)
        if text:
            try:
                qv = self.embed(text)
                ev_vecs = self.evidence_vectors([r[4] for r in rows])
                rel_scores = [max(0.0, self._cosine(qv, ev)) if ev is not None else 1.0
                              for ev in ev_vecs]
            except Exception:  # noqa: BLE001
                rel_scores = [1.0] * len(rows)
        scored: list[tuple[float, GraphFact]] = []
        for (rank, identity, claim, pid, _ev), rel_score in zip(rows, rel_scores):
            conf = fact_confidence(source="graph", paper_id=pid, relevance=rel_score)
            self.evidence[identity] = claim
            scored.append((rank, GraphFact(identity=identity, text=claim,
                          source="graph", paper_id=pid, confidence=conf)))
        scored.sort(key=lambda rf: rf[0])  # lower PPR rank = closer to query
        return [f for _, f in scored[:k]]

    # -- community hierarchy ----------------------------------------------
    def _load_hierarchy(self) -> None:
        if self._hierarchy is not None:
            return
        import json
        if not os.path.exists(self.community_json):
            raise KGUnavailable(f"community graph absent: {self.community_json}")
        d = json.load(open(self.community_json))
        self._hierarchy = d
        self._name2comm = {n["name"]: n["comm"] for n in d.get("nodes", [])}
        meta = d.get("meta", {})
        self._coarse_label = {
            str(k): (v.get("label", str(k)) if isinstance(v, dict) else str(k))
            for k, v in meta.items()
        }

    def community_of(self, node_name: str) -> Optional[int]:
        """Coarse (top-domain) community id for a node, or None."""
        self._load_hierarchy()
        c = self._name2comm.get(node_name)
        return c[_COARSE_POS] if c and len(c) > _COARSE_POS else None

    def community_label(self, coarse_id: Any) -> str:
        self._load_hierarchy()
        return self._coarse_label.get(str(coarse_id), f"community {coarse_id}")

    def locate(self, text: str, m: int = 12) -> tuple[Optional[int], list[str]]:
        """Vote the dominant coarse community of a query's seeds. Returns
        (coarse_id, seeds). The numerical loop never sees this; it only shapes
        which region a pathfind starts from."""
        import collections
        seeds = self.seeds(text, m=m)
        votes = collections.Counter(
            self.community_of(s) for s in seeds if self.community_of(s) is not None
        )
        coarse = votes.most_common(1)[0][0] if votes else None
        return coarse, seeds

    # -- web-fill of missing nodes ----------------------------------------
    def web_fill(self, query: str, link_to: str = "") -> Optional[GraphFact]:
        """Search the web for a missing node, return it as a GraphFact LINKED to
        the retrieved subgraph. The model proposes the node string; a written
        cosine gate accepts the link. Returns None if nothing usable found.

        This is a string/graph task (rule 2): it never emits a verdict. The new
        node is a post-verdict proposal; it is NOT written into the base KG.
        """
        try:
            from hermes_tools import web_search, web_extract  # type: ignore
        except Exception:
            # Outside the Hermes runtime the surface must still run; report absence.
            return None
        try:
            hits = web_search(query, limit=3)
            results = (hits or {}).get("data", {}).get("web", [])
            if not results:
                return None
            top = results[0]
            url = top.get("url", "")
            title = top.get("title", "") or query
            body = ""
            if url:
                ex = web_extract([url], char_limit=2000)
                res = (ex or {}).get("results", [])
                if res:
                    body = (res[0].get("content") or "")[:800]
            text = f"{title}. {body}".strip()
            return GraphFact(
                identity=title[:120],
                text=text,
                source="web_fill",
                linked_to=link_to,
            )
        except Exception:  # noqa: BLE001
            return None

    def _web_hits(self, query: str, limit: int) -> list[dict]:
        """Return web hits [{"url","title","description"}] from the injected
        backend, or hermes_tools inside the Hermes runtime. Empty on absence —
        the surface degrades gracefully rather than fabricating results."""
        if self.web_search_fn is not None:
            try:
                hits = self.web_search_fn(query, limit) or []
                # normalize: accept either a list of dicts or the hermes shape
                if isinstance(hits, dict):
                    hits = hits.get("data", {}).get("web", [])
                return list(hits)
            except Exception:  # noqa: BLE001
                return []
        try:
            from hermes_tools import web_search  # type: ignore
        except Exception:  # noqa: BLE001
            return []
        try:
            hits = web_search(query, limit=limit)
            return (hits or {}).get("data", {}).get("web", [])
        except Exception:  # noqa: BLE001
            return []

    def _web_body(self, url: str, fallback: str = "") -> str:
        """Fetch page body via hermes_tools if available, else the fallback."""
        if not url:
            return fallback
        try:
            from hermes_tools import web_extract  # type: ignore
        except Exception:  # noqa: BLE001
            return fallback
        try:
            ex = web_extract([url], char_limit=max(2000, self.web_body_chars))
            res = (ex or {}).get("results", [])
            return (res[0].get("content") or "")[: self.web_body_chars] if res else fallback
        except Exception:  # noqa: BLE001
            return fallback

    def _cosine(self, a, b) -> float:
        """Cosine of two embedding vectors (both unit-norm already; dot = cosine)."""
        try:
            return float(sum(x * y for x, y in zip(a, b)))
        except Exception:  # noqa: BLE001
            return 0.0

    def web_facts(
        self,
        query: str,
        *,
        limit: int = 4,
        link_nodes: Optional[list[str]] = None,
        min_link_score: float = 0.0,
    ) -> list[GraphFact]:
        """Search the web and LINK each result into the graph at the same time.

        For each hit: embed its text, attach it to the nearest node in
        ``link_nodes`` (the retrieved subgraph) by cosine, record an
        :class:`OverlayEdge` on the run-local overlay, and return it as a
        GraphFact. The overlay makes these nodes traversable by later graph
        walks in the same subproblem WITHOUT mutating the base KG (charter:
        acquisition emits a reviewable proposal, never an in-run graph write).

        This is a string/graph task (rule 2): it never emits a verdict.
        """
        results = self._web_hits(query, limit)
        if not results:
            return []
        # Precompute embeddings of the subgraph anchor nodes for linking.
        anchors: list[tuple[str, Any]] = []
        for n in link_nodes or []:
            try:
                anchors.append((n, self.embed(n)))
            except Exception:  # noqa: BLE001
                break
        out: list[GraphFact] = []
        for hit in results:
            url = hit.get("url", "")
            title = hit.get("title", "") or query
            desc = hit.get("description", "") or ""
            body = self._web_body(url, fallback=desc)
            text = f"{title}. {body or desc}".strip()
            identity = title[:120]
            self.evidence[identity] = text  # full web abstract/body, not just title
            # link into the subgraph: nearest anchor by cosine
            best_node, best_score = "", 0.0
            if anchors:
                try:
                    wv = self.embed(text[:400])
                    for node, av in anchors:
                        sc = self._cosine(wv, av)
                        if sc > best_score:
                            best_node, best_score = node, sc
                except Exception:  # noqa: BLE001
                    pass
            if best_node and best_score < min_link_score:
                continue  # too weak a link to be trustworthy graph material
            self.overlay.append(
                OverlayEdge(
                    web_identity=identity,
                    base_node=best_node,
                    score=best_score,
                    text=text,
                    paper_id=url,
                )
            )
            out.append(
                GraphFact(
                    identity=identity,
                    text=text,
                    source="web_fill",
                    paper_id=url,
                    linked_to=best_node,
                    link_score=best_score,
                    confidence=fact_confidence(
                        source="web_fill",
                        paper_id=url,
                        relevance=max(0.0, best_score) if best_node else 1.0,
                        link_score=best_score,
                    ),
                )
            )
        return out

    def overlay_proposal(self) -> list[dict]:
        """The reviewable graph-addition proposal accumulated this run."""
        return [e.as_record() for e in self.overlay]

    # -- diagnostics -------------------------------------------------------
    def capabilities(self) -> dict:
        """Report which KG capabilities are live (no fabrication)."""
        caps = {"embed": False, "retriever": False, "community": False}
        try:
            self.embed("probe"); caps["embed"] = True
        except Exception:  # noqa: BLE001
            pass
        try:
            _ = self.retriever; caps["retriever"] = True
        except Exception:  # noqa: BLE001
            pass
        try:
            self._load_hierarchy(); caps["community"] = True
        except Exception:  # noqa: BLE001
            pass
        return caps
