"""Grain declarations for the event-hierarchy HLE orchestrator.

Every grain is declared once (charter rule 4): a name, one sentence for what a
unit is, one sentence for what a credit is, and its numerical ControllerConfig.
The grains carry NO task vocabulary and NO schema (the frozen ChannelSchema is
bound run-specifically by the Context). See src/events/DESIGN.md.

Nesting:  QUESTION  >  SUBPROBLEM  >  PATHFIND
          (root)       (domain)       (graph walk to an event-end)
"""
from __future__ import annotations

from method_loop import Grain
from rarefaction import ChannelSchema, ControllerConfig

__all__ = [
    "QUESTION_GRAIN",
    "SUBPROBLEM_GRAIN",
    "PATHFIND_GRAIN",
    "CHANNEL_SCHEMAS",
    "grain_order",
]

# --------------------------------------------------------------------------
# Numerical controller configs (the decision edge is numbers only, rule 2).
# gamma = expected-next-credit cutoff above the zero-yield floor; streak_length
# = how many consecutive sub-threshold units before the verdict says stop.
# Conservative defaults mirrored from question_pipeline/acquisition.py; these
# are calibration knobs, not task logic.
# --------------------------------------------------------------------------

#: PATHFIND: unit = one hop. A walk should keep hopping while hops still reach
#: new nodes; stop after a short streak of new-nothing hops.
PATHSTEP_CONTROL = ControllerConfig.uniform(
    ("overall",), gamma=0.06, rho=0.0, streak_length=3
)

#: SUBPROBLEM: unit = one completed pathfind. Give a domain a few walks before
#: declaring it locally saturated.
SUBPROBLEM_CONTROL = ControllerConfig.uniform(
    ("overall",), gamma=0.0, rho=0.0, streak_length=4
)

#: QUESTION (root): unit = one completed subproblem. The run converges when
#: further subproblems stop contributing distinct answer-bearing identities.
QUESTION_CONTROL = ControllerConfig.uniform(
    ("overall",), gamma=0.0, rho=0.0, streak_length=4
)


PATHFIND_GRAIN = Grain(
    name="pathfind",
    unit=(
        "one hop step along a curvature/PPR-gated edge from the current graph "
        "frontier toward the subproblem's target answer region; a web-fill node "
        "linked into the retrieved subgraph counts as one hop unit"
    ),
    credit=(
        "one distinct graph node identity encountered by this hop; the same "
        "identity reached again on a later hop is recurrence, not a new finding"
    ),
    control=PATHSTEP_CONTROL,
)

SUBPROBLEM_GRAIN = Grain(
    name="subproblem",
    unit="one completed pathfind episode (a walk toward an event-end) for this subproblem",
    credit=(
        "one distinct node identity that a pathfind contributed to this "
        "subproblem, counted once however many hops in that walk reached it"
    ),
    control=SUBPROBLEM_CONTROL,
)

QUESTION_GRAIN = Grain(
    name="question",
    unit="one completed subproblem episode of this question",
    credit=(
        "one distinct answer-bearing identity a subproblem surfaced, counted "
        "once however many of its pathfinds carried it"
    ),
    control=QUESTION_CONTROL,
)


#: Run-specific frozen channel schemas, one per grain (Context.enter fails
#: closed without them). This surface is single-channel: it counts opaque node
#: identities and stays schema-agnostic (it never knows what a result column is).
CHANNEL_SCHEMAS: dict[str, ChannelSchema] = {
    "question": ChannelSchema.single(),
    "subproblem": ChannelSchema.single(),
    "pathfind": ChannelSchema.single(),
}


def grain_order() -> tuple[Grain, ...]:
    """Declared nesting order, outermost first (Context validates against this)."""
    return (QUESTION_GRAIN, SUBPROBLEM_GRAIN, PATHFIND_GRAIN)
