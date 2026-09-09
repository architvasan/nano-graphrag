# Event-Hierarchy HLE Orchestrator — Design

Status: design note written BEFORE build (per standing order). Strictly follows
`method_loop.Episode` (the method-loop-separation architecture). This is the
`src/events/` package: a new **surface binding** of the generic Episode method to
Humanity's-Last-Exam question answering, merging graph retrieval + graph
pathfinding + web-fill of missing nodes.

## How the repo's primitives realize the user's spec

| User's requirement                                   | method_loop primitive (followed, not reinvented)                     |
|------------------------------------------------------|----------------------------------------------------------------------|
| event hierarchical parent-child structure            | `Episode` nesting `Episode`/`Leaf`, path-addressed by `Context`      |
| children spawn their own new nodes                   | a child `Episode`'s `UnitSource.next` yields `Leaf`s for new nodes    |
| children communicate UP the chain only               | `Episode._contribution` fan-up: child distinct identities -> parent credits |
| memory passed upwards                                | `CreditResult` -> `Contribution` -> `UnitRecord` -> `EpisodeRecord`  |
| graph path-finding to reach an event end             | `PATHSTEP_GRAIN`: unit = one hop; ends `yield_stop`/`exhausted`       |
| web search fills missing nodes linked to subgraph    | `WebFillSource`: model proposes node, KG-retrieval links into subgraph|
| main orchestrator makes decisions (answers)          | root `Episode`; verdict is NUMERICAL (rarefaction), LLM only answers  |

## Charter invariants (from docs/ACQUISITION_LOOP.md) — strictly honored
1. No loop written on a surface: only `Episode.run` loops. Our sources only answer `next(view)`.
2. Decisions are numerical only. The continue/stop/switch edge is the rarefaction
   `ControllerVerdict`. The LLM (Llama-3.1-8B) is a string expert: decompose, extract,
   judge relevance, compose the final answer. A model NEVER emits a count/verdict a branch reads.
3. Credit has one owner: the numerical component, from accepted identities. Web/graph
   acceptance establishes a source chain; it does not itself assign method credit.
4. Every grain declared once (name, unit sentence, credit sentence, ControllerConfig).
5. Acquisition never mutates the base graph. New web-fill nodes are a post-verdict
   proposal (returned in the record), not written into the KG mid-run.

## Grain nesting (the event hierarchy)

```
QUESTION  (root Episode — orchestrator/answerer)
  unit   = one completed SUBPROBLEM episode
  credit = one distinct answer-bearing identity a subproblem surfaced
  source = QuestionSource: LLM decomposes the HLE question into subproblems
           (string task), yields one child SUBPROBLEM Episode per subproblem;
           if it cannot decompose -> yields N subproblems each seeded at a
           DIFFERENT graph location (the "parallelize in different places" path)
    │
    └── SUBPROBLEM  (child Episode — one domain thread)
          unit   = one completed PATHFIND episode (a walk toward an event-end)
          credit = one distinct node identity the pathfind reached
          source = SubproblemSource: locate the subproblem's community
                   (embed -> seeds -> community hierarchy), yield a PATHFIND
                   Episode from that seed toward the answer region
            │
            └── PATHFIND  (child Episode — graph path to an event end)
                  unit   = one hop step along a curvature/PPR-gated edge
                  credit = one distinct node identity encountered at this hop
                  source = FrontierSource: from the running EpisodeView, pick the
                           next best hop toward the target "event end" (answer
                           entity region). Two Leaf kinds it may yield:
                             (a) GraphHopLeaf  — an existing KG node on the frontier
                             (b) WebFillLeaf   — when the frontier is thin/missing,
                                 model proposes a node from web search, and the
                                 KG retriever LINKS it to the retrieved subgraph
                                 (cosine + edge attach). Node is new but enters
                                 the parent only as a credit identity (rule 5).
                  ends: yield_stop  = event-end reached (answer region hit)
                        exhausted    = frontier empty
                        (numerical verdict may stop the walk before either)
```

## Communication direction (strictly up the chain)
- A `PATHFIND` child's reached identities fan up to become `SUBPROBLEM` credits.
- A `SUBPROBLEM` child's reached identities fan up to become `QUESTION` credits.
- Siblings never talk directly; a subproblem sees another only via material the
  parent already folded in (matches "communicate up back to the parent node").
- Memory = the `EpisodeRecord` tree; the root reads its children's records to
  compose the answer. Nothing flows downward except the initial decomposition.

## The one place the model lives on each surface (rule 2)
- QuestionSource.next : decompose question (sample subproblem strings)
- FrontierSource.next : (web-fill only) propose a missing node string + relevance
- Leaf.extract        : read a node/page into observations
- Leaf.credit         : DETERMINISTIC projection accepted identities -> CreditResult
- final answer        : root composes answer string from the record tree
Everywhere else (verdict, which hop counts, when to stop) = numerical rule.

## KG bridge (composes with the ragmosis KG CLIs)
`events/kg_bridge.py` adapts the existing tooling WITHOUT importing it into the
method core (keeps method_loop schema-agnostic, per RUNTIME_INVARIANTS layering):
- embeddings     : ragmosis.inference.argo_embed.embed (argo:text-embedding-3-large, 3072d)
- graph + PPR    : the GraphRetriever (bioalchemy_retrieval_recall load path)
- community tree : community_graph.json (43 coarse labeled domains, fine->coarse)
- web-fill       : web_search + web_extract -> node text -> embed -> attach by cosine+edge

## Files
- events/grains.py       : the 3 Grain declarations + ControllerConfigs + ChannelSchemas
- events/kg_bridge.py    : KGBridge (embed, seeds, community-of, frontier, web-fill)
- events/sources.py      : QuestionSource, SubproblemSource, FrontierSource + Leaf builders
- events/orchestrator.py : wire the root Episode + Context; run; compose answer
- events/cli.py          : `nanograph-events` entry point; composes with KG CLIs
