"""`nanograph-events` — CLI for the event-hierarchy HLE orchestrator.

Composes with the KG tooling (ragmosis embeddings, the PPR graph retriever, and
the community hierarchy) via KGBridge. The LLM for string tasks (decompose,
compose answer) is pluggable; by default it wires the ragmosis ALCF client if
reachable, else runs in evidence-digest mode (no fabrication).

Subcommands:
  caps                 report which KG capabilities are live
  ask   "<question>"   answer one question through the event hierarchy
  batch <file.jsonl>   answer a JSONL of {"question": ...} rows -> results JSONL

Examples:
  nanograph-events caps
  nanograph-events ask "Which receptor does Nipah virus G bind?"
  nanograph-events batch questions.jsonl --out results.jsonl --max-subproblems 4
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Optional

from .kg_bridge import KGBridge
from .orchestrator import EventOrchestrator
from .sources import LLMFn


def _default_llm() -> Optional[LLMFn]:
    """Wire the ragmosis ALCF weak-LLM client if reachable; else None.

    Kept behind a factory so the CLI still runs (evidence-digest mode) when the
    KG project / model endpoint is not importable from this environment.
    """
    kg_src = os.environ.get(
        "KG_MEMORY_ROOT", os.path.expanduser("~/Desktop/Projects/kg-memory-system")
    )
    src = os.path.join(kg_src, "src")
    if src not in sys.path and os.path.isdir(src):
        sys.path.insert(0, src)
    try:
        from ragmosis.inference import alcf_client as LLM  # type: ignore

        model = os.environ.get("NANOGRAPH_LLM_MODEL", "meta-llama/Meta-Llama-3.1-8B-Instruct")
        cluster = os.environ.get("NANOGRAPH_LLM_CLUSTER", "sophia")

        def _call(prompt: str) -> str:
            return LLM.chat(
                [{"role": "user", "content": prompt}],
                model,
                cluster=cluster,
                temperature=0.0,
                max_tokens=500,
            )

        return _call
    except Exception:  # noqa: BLE001
        return None


def _default_web():
    """Wire a standalone web backend (ragmosis OpenAlex/Semantic-Scholar rescue)
    if reachable; else None so the bridge falls back to hermes_tools (inside the
    Hermes runtime) or degrades gracefully. Returns query,limit -> list of
    {"url","title","description"} hits.
    """
    kg_root = os.environ.get(
        "KG_MEMORY_ROOT", os.path.expanduser("~/Desktop/Projects/kg-memory-system")
    )
    expt = os.path.join(kg_root, "experiments", "provenance_threshold")
    if expt not in sys.path and os.path.isdir(expt):
        sys.path.insert(0, expt)
    try:
        from graph_web_rescue import openalex_search  # type: ignore

        def _web(query: str, limit: int = 4):
            hits = openalex_search(query, k=limit) or []
            return [
                {"url": url, "title": title, "description": abstr}
                for (title, abstr, _sid, url) in hits
            ]

        return _web
    except Exception:  # noqa: BLE001
        return None


def _make_orchestrator(args: argparse.Namespace) -> EventOrchestrator:
    bridge = KGBridge(gate=args.gate, pct=args.pct, web_search_fn=_default_web())
    llm = None if args.no_llm else _default_llm()
    return EventOrchestrator(
        bridge=bridge,
        llm=llm,
        max_subproblems=args.max_subproblems,
        max_walks=args.max_walks,
    )


def _cmd_caps(args: argparse.Namespace) -> int:
    bridge = KGBridge(gate=args.gate, pct=args.pct)
    caps = bridge.capabilities()
    llm = None if args.no_llm else _default_llm()
    caps["llm"] = llm is not None
    print(json.dumps(caps, indent=2))
    return 0


def _cmd_ask(args: argparse.Namespace) -> int:
    orch = _make_orchestrator(args)
    res = orch.answer_question(args.question)
    print(json.dumps(res.as_record(), indent=2)[:4000] if args.json else _fmt(res))
    return 0


def _cmd_batch(args: argparse.Namespace) -> int:
    orch = _make_orchestrator(args)
    out_path = args.out or (os.path.splitext(args.file)[0] + "_events_results.jsonl")
    n = 0
    with open(out_path, "w") as out:
        for line in open(args.file):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            q = row.get("question") or row.get("q")
            if not q:
                continue
            res = orch.answer_question(q)
            out.write(json.dumps(res.as_record()) + "\n")
            out.flush()
            n += 1
            print(f"[{n}] {res.ended_by} subs={res.n_subproblems} "
                  f"ids={res.n_distinct_identities} :: {res.answer[:80]}", flush=True)
    print(f"\nwrote {n} results -> {out_path}")
    return 0


def _fmt(res) -> str:
    lines = [
        f"Q: {res.question}",
        f"answer: {res.answer}",
        f"subproblems: {res.n_subproblems}  distinct_identities: "
        f"{res.n_distinct_identities}  ended_by: {res.ended_by}",
    ]
    for s in res.subproblem_summaries:
        lines.append(f"  - {s['subproblem']} [{s['ended_by']}] "
                     f"({s['n_identities']} nodes)")
    return "\n".join(lines)


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--gate", default="forman", help="curvature gate for the retriever")
    p.add_argument("--pct", type=float, default=0.3, help="curvature keep percentile")
    p.add_argument("--max-subproblems", type=int, default=4, dest="max_subproblems")
    p.add_argument("--max-walks", type=int, default=2, dest="max_walks")
    p.add_argument("--no-llm", action="store_true",
                   help="run in evidence-digest mode (no answer composition)")


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        prog="nanograph-events",
        description="Event-hierarchy HLE orchestrator over the method_loop Episode.",
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_caps = sub.add_parser("caps", help="report live KG capabilities")
    _add_common(p_caps)
    p_caps.set_defaults(func=_cmd_caps)

    p_ask = sub.add_parser("ask", help="answer one question")
    p_ask.add_argument("question")
    p_ask.add_argument("--json", action="store_true", help="emit full JSON record")
    _add_common(p_ask)
    p_ask.set_defaults(func=_cmd_ask)

    p_batch = sub.add_parser("batch", help="answer a JSONL of questions")
    p_batch.add_argument("file")
    p_batch.add_argument("--out", default=None)
    _add_common(p_batch)
    p_batch.set_defaults(func=_cmd_batch)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
