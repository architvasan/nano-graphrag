"""Summary — the distilled-memory channel that fans up beside CreditResult.

method_loop's Contribution carries identities (CreditResult) for the numerical
verdict; that is the ONLY thing a child episode fans up by default. This module
adds a parallel, non-numerical channel: a distilled Summary produced as each
child unit completes (via the Episode.on_unit hook — the sanctioned extension
point, since Episode._contribution is FINAL) and aggregated up the tree.

Credit answers "how many distinct things did the child find" (numbers, for the
controller). Summary answers "what did the child actually establish" (memory,
for the parent's reasoning). Keeping them separate honors the charter: a model
never emits a count a branch consumes — the summary is string material only and
never feeds the verdict.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

__all__ = ["Summary", "SummaryBus", "distill_summary", "summary_from_record"]

# Optional distiller: (question/subproblem text, evidence lines) -> summary text.
DistillFn = Callable[[str, list[str]], str]


@dataclass
class Summary:
    """Distilled memory from one episode, mirroring the scope tree.

    ``text`` is the human-readable distillation; ``key_relations`` are the most
    salient edges/claims (verbatim, never invented); ``n_identities`` is the
    credit count carried alongside (diagnostic, not the verdict); ``children``
    are the nested summaries fanned up from this episode's child units.
    """

    scope_key: str
    kind: str  # "graph" | "web" | "subproblem" | "question"
    text: str = ""
    key_relations: list[str] = field(default_factory=list)
    n_identities: int = 0
    ended_by: str = ""
    children: list["Summary"] = field(default_factory=list)

    def as_record(self) -> dict:
        return {
            "scope_key": self.scope_key,
            "kind": self.kind,
            "text": self.text,
            "key_relations": self.key_relations,
            "n_identities": self.n_identities,
            "ended_by": self.ended_by,
            "children": [c.as_record() for c in self.children],
        }

    def flatten_relations(self, limit: int = 40) -> list[str]:
        """All key relations from this summary and its descendants, deduped."""
        seen: list[str] = []
        stack = [self]
        while stack and len(seen) < limit:
            s = stack.pop(0)
            for r in s.key_relations:
                if r not in seen:
                    seen.append(r)
            stack.extend(list(s.children))
        return seen[:limit]


@dataclass
class SummaryBus:
    """Per-episode accumulator of child summaries.

    One bus per episode instance; its ``on_unit`` is wired as the episode's hook
    so a Summary is captured incrementally as each child unit completes, rather
    than only in a final post-run reduction.
    """

    scope_key: str
    kind: str
    collected: list[Summary] = field(default_factory=list)

    def on_unit(self, unit: Any, contribution: Any, unit_record: Any) -> None:
        """Episode.on_unit hook. Capture one child's distilled summary.

        Reads only the Contribution/UnitRecord the kernel already built — never
        mutates them (charter: a source that mutates a record corrupts its own
        export). Excluded/failed children are recorded too, flagged by ended_by,
        so nothing is silently dropped."""
        child = getattr(contribution, "child", None)
        credits = list(getattr(contribution, "credit").credits) if getattr(contribution, "credit", None) else []
        if child is not None:
            # a nested child episode: summarize from its record
            self.collected.append(
                Summary(
                    scope_key=getattr(child, "scope_key", "?"),
                    kind=_infer_kind(getattr(child, "scope_key", "")),
                    key_relations=list(getattr(child, "distinct_identities", []))[:8],
                    n_identities=len(getattr(child, "distinct_identities", [])),
                    ended_by=getattr(child, "ended_by", ""),
                )
            )
        elif credits:
            # a leaf hop that carried credit: keep its relation as a mini-summary
            self.collected.append(
                Summary(
                    scope_key=getattr(unit_record, "unit_label", "hop"),
                    kind="hop",
                    key_relations=credits[:1],
                    n_identities=len(credits),
                )
            )

    def summary(self, text: str = "") -> Summary:
        """Roll the collected child summaries into this episode's Summary."""
        rels: list[str] = []
        for c in self.collected:
            for r in c.key_relations:
                if r not in rels:
                    rels.append(r)
        return Summary(
            scope_key=self.scope_key,
            kind=self.kind,
            text=text,
            key_relations=rels[:12],
            n_identities=sum(c.n_identities for c in self.collected),
            children=list(self.collected),
        )


def _infer_kind(scope_key: str) -> str:
    if scope_key.startswith("web"):
        return "web"
    if scope_key.startswith("graph") or scope_key.startswith("walk"):
        return "graph"
    if scope_key.startswith("sub"):
        return "subproblem"
    return "episode"


def distill_summary(
    text: str,
    relations: list[str],
    distill: Optional[DistillFn] = None,
) -> str:
    """Produce a summary string from evidence relations.

    With a distiller (LLM), ask for a 1-2 sentence distillation. Without one,
    return a deterministic digest of the top relations — never fabricated, never
    an invented conclusion."""
    if not relations:
        return "(no evidence reached)"
    if distill is None:
        head = "; ".join(relations[:4])
        return f"[digest] {head}"
    try:
        return (distill(text, relations) or "").strip() or "[distiller returned empty]"
    except Exception as exc:  # noqa: BLE001
        return f"[distill failed: {type(exc).__name__}]"


def summary_from_record(
    record: Any,
    question: str = "",
    distill: Optional[DistillFn] = None,
    _depth: int = 0,
) -> Summary:
    """Post-run reduction: build the full nested Summary tree from an
    EpisodeRecord. Complements the incremental on_unit capture — the record
    already holds the whole tree, so this reconstructs summaries at every scope
    with optional LLM distillation at the reasoning levels (question/subproblem).

    Distillation is applied only where it adds value (episodes with children);
    leaf-heavy walk levels keep their verbatim relations to avoid paraphrase
    drift on the actual evidence."""
    scope_key = getattr(record, "scope_key", "?")
    kind = _infer_kind(scope_key) if _depth else "question"
    children: list[Summary] = []
    for unit in getattr(record, "unit_records", []):
        child_rec = getattr(unit, "child", None)
        if child_rec is not None:
            children.append(summary_from_record(child_rec, question, distill, _depth + 1))
    relations = list(getattr(record, "distinct_identities", []))
    # distill only at reasoning levels that have children; keep walk relations raw
    text = ""
    if children and kind in ("question", "subproblem"):
        text = distill_summary(question or scope_key, relations[:12], distill)
    return Summary(
        scope_key=scope_key,
        kind=kind,
        text=text,
        key_relations=relations[:12],
        n_identities=len(relations),
        ended_by=getattr(record, "ended_by", ""),
        children=children,
    )
