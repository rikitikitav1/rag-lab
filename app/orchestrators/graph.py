import operator
from typing import Annotated, TypedDict

import agent_tools
import llm
import logging_setup
import outcomes
import prompt_repo
from langgraph.graph import END, StateGraph
from models.registry import Purpose
from use_cases import agent_policy as policy
from use_cases import chat

log = logging_setup.get_logger(__name__)


# reducers must be introspectable, so the builtins cannot be handed over directly
def _keep_max(old: int, new: int) -> int:
    return max(old, new)


def _merge(old: dict, new: dict) -> dict:
    return {**old, **new}


class State(TypedDict, total=False):
    messages: Annotated[list, operator.add]
    sources: Annotated[list, operator.add]
    dropped_sources: Annotated[list, operator.add]
    dropped_hits: Annotated[list, operator.add]
    prompt_tokens: Annotated[int, operator.add]
    completion_tokens: Annotated[int, operator.add]
    max_prompt_tokens: Annotated[int, _keep_max]
    hops: int
    nudges: int
    external: bool
    fallback_reason: str
    fallback_opened: bool
    fallback_announced: bool
    no_evidence_prompted: bool
    text: str
    finished: bool
    awaiting_tools: bool
    turn: object
    tool_errors: Annotated[dict, _merge]


def _ctx(config) -> dict:
    return config["configurable"]["run"]


def _tool_names(ctx) -> tuple:
    return (*ctx["remote"], agent_tools.CORPUS_TOOL)


def _schemas(ctx, external: bool) -> list:
    tools = agent_tools.schemas()
    if external:
        tools += [t.schema() for t in ctx["remote"].values()]
    return tools


def model_node(state: State, config) -> dict:
    ctx = _ctx(config)
    hop = state["hops"] + 1
    try:
        turn = ctx["chat"](
            state["messages"], tools=_schemas(ctx, state["external"]), role=ctx["role"],
            model=ctx["model"],
        )
    except RuntimeError as e:
        log.error("graph.hop_failed", hop=hop, error=str(e))
        return {"hops": hop, "finished": True}

    ctx["result"].note_prompt(turn.prompt_tokens)
    update = {
        "hops": hop,
        "prompt_tokens": turn.prompt_tokens,
        "completion_tokens": turn.completion_tokens,
        "max_prompt_tokens": turn.prompt_tokens,
        "turn": turn,
    }
    if turn.tool_calls:
        update["messages"] = [turn.message]
        update["awaiting_tools"] = True
        return update
    if state["nudges"] and outcomes.narrated_tool_call(turn.text, _tool_names(ctx)):
        log.info("graph.narrated_tool_call", hop=hop)
        update["nudges"] = state["nudges"] - 1
        update["awaiting_tools"] = False
        update["messages"] = [
            turn.message or {"role": "assistant", "content": turn.text},
            {"role": "user", "content": policy.TOOL_CALL_NUDGE},
        ]
        return update
    update["text"] = turn.text or ""
    update["finished"] = True
    return update


# one dispatch node instead of ToolNode: the verdict is computed before the messages are
# emitted, exactly as the hand-rolled loop does it, so no message has to be rewritten later
def _dispatch(state: State, ctx: dict) -> tuple[list, dict]:
    calls, errors_seen = [], {}
    for tc in state["turn"].tool_calls:
        log.info("graph.tool_call", tool=tc.function.name, arguments=tc.function.arguments)
        res = agent_tools.dispatch(
            tc.function.name,
            tc.function.arguments,
            extra=ctx["remote"] if state["external"] else None,
            k=ctx["k"],
            use_rerank=ctx["use_rerank"],
            gate_top=ctx["gate"].top,
        )
        if res.meta.get("error_kind"):
            errors_seen[tc.function.name] = res.meta["error_kind"]
        calls.append([tc, res, res.content, res.meta.get("sources", [])])
    return calls, errors_seen


def _drop_weak(corpus: list, hop: int) -> tuple[list, list]:
    log.info("graph.weak_context_dropped", hop=hop)
    dropped, hits = [], []
    for call in corpus:
        dropped.extend(s.source for s in call[3])
        hits.extend(call[3])
        call[2], call[3] = chat.NO_RESULTS, []
    return dropped, hits


def _announce(corpus: list, gate: policy.Gate) -> None:
    notice = prompt_repo.active_template(Purpose.agent_fallback)
    corpus[-1][2] = f"{corpus[-1][2]}\n\n{notice.replace('{tools}', gate.tool_signatures)}"


def tools_node(state: State, config) -> dict:
    ctx = _ctx(config)
    gate: policy.Gate = ctx["gate"]
    calls, errors_seen = _dispatch(state, ctx)
    corpus = [
        c for c in calls
        if c[0].function.name == agent_tools.CORPUS_TOOL and not c[1].meta.get("error_kind")
    ]
    verdict = policy.verdict([s for c in corpus for s in c[3]], gate) if corpus else None
    if gate.off_topic and corpus:
        verdict = policy.FallbackReason.off_topic

    dropped, dropped_hits = [], []
    if verdict in (policy.FallbackReason.weak, policy.FallbackReason.off_topic) and (
        gate.drop_weak_context
    ):
        dropped, dropped_hits = _drop_weak(corpus, state["hops"])

    # the loop recomputes announce per hop and it dies once external is open
    announced = bool(verdict and gate.announce and not state["external"] and corpus)
    if announced:
        _announce(corpus, gate)
    if verdict and gate.off_topic:
        verdict = policy.FallbackReason.off_topic

    update = {
        "messages": [
            {"role": "tool", "tool_call_id": tc.id, "content": content}
            for tc, _res, content, _sources in calls
        ],
        "sources": [s for c in calls for s in c[3]],
        "dropped_sources": dropped,
        "dropped_hits": dropped_hits,
        "tool_errors": errors_seen,
    }
    if announced:
        update["fallback_announced"] = True
    if verdict and state.get("fallback_reason") == policy.FallbackReason.none:
        update["fallback_reason"] = verdict
    if verdict and not state["external"] and ctx["remote"]:
        update["external"] = True
        update["fallback_opened"] = True
        log.info("graph.external_opened", hop=state["hops"], reason=verdict)
    return update


def final_node(state: State, config) -> dict:
    ctx = _ctx(config)
    # the loop forces a final turn only when no turn produced text at all
    if state.get("text"):
        return {}
    messages = list(state["messages"])
    update = {}
    if not state.get("sources"):
        messages = messages + [
            {"role": "user", "content": prompt_repo.active_template(Purpose.agent_no_evidence)}
        ]
        update["messages"] = [messages[-1]]
        update["no_evidence_prompted"] = True
    log.info("graph.forcing_final", hops=state["hops"], sources=len(state.get("sources", [])))
    try:
        final = ctx["chat"](messages, role=ctx["role"], model=ctx["model"])
    except RuntimeError as e:
        log.error("graph.final_failed", error=str(e))
        return update
    ctx["result"].note_prompt(final.prompt_tokens)
    update.update(
        hops=state["hops"] + 1,
        prompt_tokens=final.prompt_tokens,
        completion_tokens=final.completion_tokens,
        max_prompt_tokens=final.prompt_tokens,
        text=final.text or "",
    )
    if final.finish_reason == "length":
        log.warning("graph.truncated", hops=state["hops"] + 1)
    return update


def _after_model(state: State, config) -> str:
    if state.get("finished"):
        return "final"
    if state.get("awaiting_tools"):
        return "tools"
    # a nudge on the last hop must not buy an extra hop the loop would not take
    return "model" if state["hops"] < _ctx(config)["max_hops"] else "final"


def _after_tools(state: State, config) -> str:
    return "model" if state["hops"] < _ctx(config)["max_hops"] else "final"


def build():
    graph = StateGraph(State)
    graph.add_node("model", model_node)
    graph.add_node("tools", tools_node)
    graph.add_node("final", final_node)
    graph.set_entry_point("model")
    graph.add_conditional_edges("model", _after_model, {"tools": "tools", "final": "final", "model": "model"})
    graph.add_conditional_edges("tools", _after_tools, {"model": "model", "final": "final"})
    graph.add_edge("final", END)
    return graph.compile()


def _initial_state(question: str, system: str, external: bool) -> State:
    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": question},
        ],
        "sources": [],
        "dropped_sources": [],
        "dropped_hits": [],
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "max_prompt_tokens": 0,
        "hops": 0,
        "nudges": 1,
        "external": external,
        "fallback_reason": policy.FallbackReason.none,
        "fallback_opened": False,
        "fallback_announced": False,
        "no_evidence_prompted": False,
        "text": "",
        "finished": False,
        "awaiting_tools": False,
        "tool_errors": {},
    }


# the graph answers into the same AgentResult the hand-rolled loop fills, so logging, the judge
# and every metric downstream cannot tell which orchestrator ran
def invoke(question, system, ctx, result) -> None:
    ctx["result"] = result
    graph = build()
    state = graph.invoke(
        _initial_state(question, system, ctx["external"]),
        config={
            "configurable": {"run": ctx},
            "recursion_limit": 3 * ctx["max_hops"] + 6,
        },
    )
    result.messages.clear()
    result.messages.extend(state["messages"])
    result.sources = list(state["sources"])
    result.dropped_sources = list(state["dropped_sources"])
    result.dropped_hits = list(state["dropped_hits"])
    result.hops = state["hops"]
    result.prompt_tokens = state["prompt_tokens"]
    result.completion_tokens = state["completion_tokens"]
    result.max_prompt_tokens = state["max_prompt_tokens"]
    result.text = state.get("text") or ""
    result.fallback_reason = state.get("fallback_reason", policy.FallbackReason.none)
    result.fallback_opened = state.get("fallback_opened", False)
    result.fallback_announced = state.get("fallback_announced", False)
    result.no_evidence_prompted = state.get("no_evidence_prompted", False)
    result.tool_errors.update(state.get("tool_errors") or {})
    result.success = bool(result.text)


def versions() -> dict:
    from importlib.metadata import version

    out = {}
    for name in ("langgraph", "langchain-core", "langchain"):
        try:
            out[name] = version(name)
        except Exception:  # noqa: BLE001 - a missing package is not worth failing a run over
            out[name] = None
    return out


def context(**kwargs) -> dict:
    kwargs.setdefault("chat", llm.chat)
    return kwargs
