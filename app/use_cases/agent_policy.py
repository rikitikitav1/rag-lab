from dataclasses import dataclass
from enum import StrEnum

# the ceiling a hop budget may name, on every door that takes one
MAX_HOPS = 10

TOOL_CALL_NUDGE = (
    "You described a tool call instead of issuing one. Call the tool for real now, "
    "with the arguments its schema lists, or answer without it."
)


class FallbackPolicy(StrEnum):
    corpus_first = "corpus_first"
    corpus_first_weak = "corpus_first_weak"
    agent_choice = "agent_choice"


class GateSignal(StrEnum):
    cross_encoder = "cross_encoder"
    distance = "distance"
    either = "either"


class Orchestrator(StrEnum):
    # gone, and the values stay queryable: 422 logs carry one arm and 222 the other
    handrolled = "agent"
    langgraph_middleware = "langgraph_middleware"
    langgraph_ported = "langgraph_ported"
    langgraph_idiomatic = "langgraph_idiomatic"
    # measured and refused: it filled a reasoning schema instead of calling a tool
    schema_guided = "schema_guided"


# retired implementations, declared here so a fourth retirement is one line, not three
GONE = frozenset(
    {Orchestrator.handrolled, Orchestrator.langgraph_middleware, Orchestrator.schema_guided}
)


class FallbackReason(StrEnum):
    none = "none"
    empty = "empty"
    weak = "weak"
    off_topic = "off_topic"
    # the grader threw out every chunk: the search found something, so this is not `empty`
    graded_out = "graded_out"


# which edge ended the graph: `final` meant three things, and the reader re-derived one of them
class FinishedBy(StrEnum):
    answer = "answer"
    hops_exhausted = "hops_exhausted"
    # the loop stopped without the model producing text, and the ceiling was not the reason
    no_answer = "no_answer"
    # a bare `create_agent` has no edge of ours; the emptiness is named rather than silent
    unrecorded = "unrecorded"


@dataclass
class Topic:
    threshold: float | None = None
    score: float | None = None


@dataclass
class Gate:
    signal: str = GateSignal.distance
    top: int | None = None
    threshold: float | None = None
    distance_threshold: float | None = None
    drop_weak_context: bool = False
    off_topic: bool = False
    announce: bool = False
    tool_signatures: str = ""


def weak_by_cross_encoder(sources: list, gate: Gate) -> bool:
    scores = [s.rerank_score for s in sources if s.rerank_score is not None]
    return bool(scores) and max(scores) < gate.threshold


def weak_by_distance(sources: list, gate: Gate) -> bool:
    distances = [s.vector_distance for s in sources if s.vector_distance is not None]
    return bool(distances) and min(distances) >= gate.distance_threshold


# the one reading of when the agent's gate calls the cross-encoder: `either` calls it as well
def gates_with_cross_encoder(policy, signal) -> bool:
    return (FallbackPolicy(policy) == FallbackPolicy.corpus_first_weak
            and GateSignal(signal) != GateSignal.distance)


def verdict(sources: list, gate: Gate) -> str | None:
    if not sources:
        return FallbackReason.empty
    if gate.threshold is None and gate.distance_threshold is None:
        return None
    checks = {
        GateSignal.cross_encoder: (weak_by_cross_encoder,),
        GateSignal.distance: (weak_by_distance,),
        GateSignal.either: (weak_by_cross_encoder, weak_by_distance),
    }[gate.signal]
    if any(check(sources, gate) for check in checks):
        return FallbackReason.weak
    return None


def signatures(tools) -> str:
    return "\n".join(
        f"- {tool.name}({', '.join(tool.parameters.get('required', []))}): "
        f"{' '.join(tool.description.split())[:160]}"
        for tool in tools
    )


def required_values(tool) -> str:
    props = tool.parameters.get("properties", {})
    return "; ".join(
        f"{name}: {' '.join(props.get(name, {}).get('description', name).split())}"
        for name in tool.parameters.get("required", [])
    )
