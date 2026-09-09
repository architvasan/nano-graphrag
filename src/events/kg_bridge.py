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

__all__ = ["KGBridge", "GraphFact", "KGUnavailable"]

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


@dataclass(frozen=True)
class GraphFact:
    """One retrieved claim: a typed edge with evidence, or a web-fill node."""

    identity: str          # opaque node/edge identity (what fans up as credit)
    text: str              # human-readable claim/triple/evidence
    source: str            # "graph" | "web_fill"
    paper_id: str = ""
    linked_to: str = ""    # for web_fill: the subgraph node it was attached to


@dataclass
class KGBridge:
    """Lazily-loaded adapter over embeddings + PPR retriever + community tree."""

    kg_src: str = _KG_SRC
    kg_expt: str = _KG_EXPT
    community_json: str = _COMMUNITY_JSON
    gate: str = "forman"
    pct: float = 0.3

    _embed: Any = field(default=None, init=False, repr=False)
    _retriever: Any = field(default=None, init=False, repr=False)
    _hierarchy: Any = field(default=None, init=False, repr=False)
    _name2comm: dict = field(default_factory=dict, init=False, repr=False)
    _coarse_label: dict = field(default_factory=dict, init=False, repr=False)

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
        """PPR + typed-edge evidence for a query, as GraphFact claims."""
        r = self.retriever
        try:
            raw = r.retrieve_grounded(
                text, k=k, with_abstracts=False,
                mechanism_only=mechanism_only,
            )
        except Exception as exc:  # noqa: BLE001
            raise KGUnavailable(f"retrieve_grounded failed: {exc}") from exc
        out: list[GraphFact] = []
        for t, s in raw:
            text = str(t)
            paper_id = str(s)
            # The claim identity is the triple/head of the text (before the em-dash
            # evidence separator). retrieve_grounded returns (claim_text, paper_id).
            identity = text.split("  —  ")[0].strip()[:120] if text else ""
            if not identity:
                continue
            out.append(
                GraphFact(
                    identity=identity,
                    text=text,
                    source="graph",
                    paper_id=paper_id if paper_id and paper_id.lower() != "none" else "",
                )
            )
        return out

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
