from dataclasses import dataclass
from enum import StrEnum

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
    handrolled = "agent"
    langgraph_ported = "langgraph_ported"
    langgraph_idiomatic = "langgraph_idiomatic"


class FallbackReason(StrEnum):
    none = "none"
    empty = "empty"
    weak = "weak"
    off_topic = "off_topic"


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
