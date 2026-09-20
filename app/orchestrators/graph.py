import operator
import time
from typing import Annotated, TypedDict

import agent_tools
import llm
import logging_setup
import outcomes
import prompt_repo
import token_fields
from engines import answer_parsers
from errors import StandFault
from langgraph.graph import END, StateGraph
from models.registry import Purpose
from use_cases import agent_policy as policy
from use_cases import chat, grading

log = logging_setup.get_logger(__name__)


# reducers must be introspectable, so the builtins cannot be handed over directly
def _keep_max(old: int, new: int) -> int:
    return max(old, new)


def _merge(old: dict, new: dict) -> dict:
    return {**old, **new}


class State(TypedDict, total=False):
    messages: Annotated[list, operator.add]
    # which pieces came from which call: `contexts` is flat, and a replay cannot regroup it
    spans: Annotated[list, operator.add]
    finished_by: str
    sources: Annotated[list, operator.add]
    contexts: Annotated[list, operator.add]
    chunks: Annotated[list, operator.add]
    dropped_sources: Annotated[list, operator.add]
    dropped_hits: Annotated[list, operator.add]
    prompt_tokens: Annotated[int, operator.add]
    completion_tokens: Annotated[int, operator.add]
    max_prompt_tokens: Annotated[int, _keep_max]
    answer_parse: Annotated[list, operator.add]
    hops: int
    nudges: int
    external: bool
    fallback_reason: str
    fallback_opened: bool
    fallback_announced: bool
    announced_text: str
    no_evidence_prompted: bool
    text: str
    finished: bool
    awaiting_tools: bool
    turn: object
    # the toolbox each hop was handed: a replay cannot see it, the scripted chat ignores `tools`
    tools_offered: Annotated[list, operator.add]
    # what retrieval found before anything graded it: only `emit` turns this into messages
    pending: object
    coverage: str
    dropped_before_emit: dict
    # per call: the order the grader read the chunks in and which of them it threw out
    graded_before_emit: dict
    tool_errors: Annotated[dict, _merge]


def _ctx(config) -> dict:
    return config["configurable"]["run"]


def _tool_names(ctx) -> tuple:
    return (*ctx["remote"], agent_tools.CORPUS_TOOL)


# what the model was allowed to call on this hop: the schemas are built from exactly this
def _offered(ctx, external: bool) -> list:
    return [agent_tools.CORPUS_TOOL, *(sorted(ctx["remote"]) if external else ())]


def _schemas(ctx, external: bool) -> list:
    tools = agent_tools.schemas()
    if external:
        tools += [t.schema() for t in ctx["remote"].values()]
    return tools


def model_node(state: State, config) -> dict:
    ctx = _ctx(config)
    hop = state["hops"] + 1
    started = time.perf_counter()
    try:
        turn = ctx["chat"](
            state["messages"], tools=_schemas(ctx, state["external"]), role=ctx["role"],
            model=ctx["model"],
        )
    # the window guard refuses before the call, and a refusal it did not see wrote the row empty
    except (RuntimeError, llm.InputOverWindow) as e:
        log.error("graph.hop_failed", hop=hop, error=str(e))
        ctx["result"].failed = True
        ctx["result"].step("model", hop, failed=True)
        return {"hops": hop, "finished": True}

    ctx["result"].took("model", started)
    ctx["result"].step("model", hop, tool_calls=len(turn.tool_calls or ()), answered=bool(turn.text))
    ctx["result"].note_prompt(turn.prompt_tokens)
    update = {
        "hops": hop,
        "tools_offered": [_offered(ctx, state["external"])],
        # a server that answers without counts reduced into `0 + None` and failed the whole row
        "prompt_tokens": turn.prompt_tokens or 0,
        "completion_tokens": turn.completion_tokens or 0,
        "max_prompt_tokens": turn.prompt_tokens or 0,
        "answer_parse": [turn.parsed],
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
    # the answering turn was the one turn the row never kept, and a replay cannot invent it
    update["messages"] = [turn.message or {"role": "assistant", "content": turn.text or ""}]
    update["text"] = turn.text or ""
    update["finished"] = True
    if token_fields.cut(turn.finish_reason):
        log.warning("graph.truncated", hops=hop)
    return update


# our dispatch instead of ToolNode: the verdict is computed before any message is emitted
def _dispatch(state: State, ctx: dict) -> tuple[list, dict]:
    calls, errors_seen = [], {}
    for tc in state["turn"].tool_calls:
        log.info("graph.tool_call", tool=tc.function.name, arguments=tc.function.arguments)
        started = time.perf_counter()
        res = ctx["dispatch"](
            tc.function.name,
            tc.function.arguments,
            extra=ctx["remote"] if state["external"] else None,
            k=ctx["k"],
            use_rerank=ctx["use_rerank"],
            gate_top=ctx["gate"].top,
            variant=ctx["variant"],
        )
        ctx["result"].took(f"tool:{tc.function.name.split('__')[0]}", started)
        if res.meta.get("error_kind"):
            errors_seen[tc.function.name] = res.meta["error_kind"]
        if res.meta.get("ef_search") is not None:
            ctx["result"].ef_search = res.meta["ef_search"]
        calls.append([tc, res, res.content, res.meta.get("sources", [])])
    return calls, errors_seen


# the field list comes from the dataclass, so a new gate signal cannot be left out of the record
def _as_row(source) -> dict:
    from dataclasses import fields

    return {f.name: getattr(source, f.name, None) for f in fields(chat.Source)}


# what the call returned before this rewrote it: without it no replay can reach the gate branch
def _drop_weak(corpus: list, hop: int) -> tuple[list, list, dict]:
    log.info("graph.weak_context_dropped", hop=hop)
    dropped, hits, by_call = [], [], {}
    for call in corpus:
        dropped.extend(s.source for s in call[3])
        hits.extend(call[3])
        by_call[call[0].id] = {
            "sources": [_as_row(s) for s in call[3]],
            "pieces": len(agent_tools.context_pieces(call[1].meta, call[2])),
        }
        call[2], call[3] = chat.NO_RESULTS, []
    return dropped, hits, by_call


# returns what it appended: `announced` says the notice fired, and nothing said what it said
def _announce(corpus: list, gate: policy.Gate, template) -> str:
    notice = template(Purpose.agent_fallback).replace("{tools}", gate.tool_signatures)
    corpus[-1][2] = f"{corpus[-1][2]}\n\n{notice}"
    return notice


def _corpus_calls(calls: list) -> list:
    return [
        c for c in calls
        if c[0].function.name == agent_tools.CORPUS_TOOL and not c[1].meta.get("error_kind")
    ]


# search only: the results sit in the state, and no message for the model leaves this node
def retrieve_node(state: State, config) -> dict:
    ctx = _ctx(config)
    gate: policy.Gate = ctx["gate"]
    calls, errors_seen = _dispatch(state, ctx)
    corpus = _corpus_calls(calls)
    verdict = policy.verdict([s for c in corpus for s in c[3]], gate) if corpus else None
    if gate.off_topic and corpus:
        verdict = policy.FallbackReason.off_topic
    ctx["result"].step(
        "retrieve", state["hops"], calls=len(calls), corpus=len(corpus),
        sources=sum(len(c[3]) for c in corpus), verdict=str(verdict or policy.FallbackReason.none),
        errors=len(errors_seen or {}),
    )
    return {"pending": calls, "coverage": verdict or "", "tool_errors": errors_seen}


# reached only over the edge the verdict decides, so `coverage` here is never empty
def fallback_node(state: State, config) -> dict:
    ctx = _ctx(config)
    gate: policy.Gate = ctx["gate"]
    verdict, corpus = state["coverage"], _corpus_calls(state["pending"])
    update = {}
    if verdict in (policy.FallbackReason.weak, policy.FallbackReason.off_topic) and (
        gate.drop_weak_context
    ):
        dropped, hits, before = _drop_weak(corpus, state["hops"])
        update["dropped_sources"], update["dropped_hits"] = dropped, hits
        update["dropped_before_emit"] = before
    # the loop recomputes announce per hop and it dies once external is open
    graded_out = verdict == policy.FallbackReason.graded_out
    if gate.announce and not state["external"] and corpus and not graded_out:
        update["announced_text"] = _announce(corpus, gate, ctx["template"])
        update["fallback_announced"] = True
    if state.get("fallback_reason") == policy.FallbackReason.none:
        update["fallback_reason"] = verdict
    # a grader that emptied the corpus does not open the way outside, the row refuses instead
    if not state["external"] and ctx["remote"] and not graded_out:
        update["external"] = True
        update["fallback_opened"] = True
        log.info("graph.external_opened", hop=state["hops"], reason=verdict)
    ctx["result"].step(
        "fallback", state["hops"], verdict=str(verdict),
        dropped=len(update.get("dropped_sources") or ()),
        opened=bool(update.get("fallback_opened")), announced=bool(update.get("fallback_announced")),
    )
    return update


# llama3.1 renders tool schemas only in the last user message, and a tool answer buries them
def _restated(ctx, state) -> list:
    if not ctx.get("restate_tools"):
        return []
    lines = "\n".join(
        f"- {fn['name']}({', '.join(fn.get('parameters', {}).get('required', []))}): "
        f"{' '.join(fn.get('description', '').split())[:160]}"
        for fn in (s.get("function", s) for s in _schemas(ctx, state.get("external", False)))
    )
    if not lines:
        return []
    return [{"role": "user", "content": f"The tools you may still call:\n{lines}"}]


# sources are per file and deduplicated, pieces are per chunk: two chunks of one file align badly
def _surviving_sources(sources: list, chunks: list, kept: list) -> tuple[list, bool]:
    if len(chunks) == len(sources) and not any(c and c.get("source") for c in chunks):
        return [sources[i] for i in kept], True
    files = {chunks[i].get("source") for i in kept if i < len(chunks) and chunks[i]}
    if not files and kept:
        # nothing addressable to filter by, and dropping by position would name the wrong file
        return sources, False
    return [s for s in sources if s.source in files], True


def _keep_only(call, pieces: list, chunks: list, kept: list) -> bool:
    keep = [pieces[i] for i in kept]
    call[1].meta["contexts"] = keep
    if len(chunks) == len(pieces):
        call[1].meta["chunks"] = [chunks[i] for i in kept]
    call[3], filtered = _surviving_sources(call[3], chunks, kept)
    call[2] = "\n\n".join(keep) or chat.NO_RESULTS
    return filtered


# the verdict is computed in `retrieve_node` before this runs, so grading a doomed context is waste
def _already_doomed(state: State, ctx: dict) -> bool:
    verdict = state.get("coverage")
    return bool(verdict) and ctx["gate"].drop_weak_context and verdict in (
        policy.FallbackReason.weak, policy.FallbackReason.off_topic
    )


# a filter, not a signal: it drops chunks the grader read as foreign and leaves the rest standing
def grade_node(state: State, config) -> dict:
    ctx = _ctx(config)
    ask = ctx.get("grade_ask")
    corpus = _corpus_calls(state["pending"]) if state.get("pending") else []
    if not ask or not corpus or _already_doomed(state, ctx):
        return {}
    started = time.perf_counter()
    # one verdict per chunk for the whole row: a hop that searched again returned the same chunks
    memo, plan = ctx.setdefault("graded", {}), {}
    question, graded_all, kept_all, unreadable, asked, unaligned = ctx["question"], 0, 0, 0, 0, 0
    for call in corpus:
        pieces = agent_tools.context_pieces(call[1].meta, call[2])
        if not pieces:
            continue
        chunks = agent_tools.chunk_pieces(call[1].meta, call[2])
        graded = grading.grade_pieces(question, pieces, chunks, ctx["grade_system"], ask, memo)
        graded_all += len(pieces)
        kept_all += len(graded["kept"])
        unreadable += graded["unreadable"]
        asked += graded["asked"]
        plan[call[0].id] = {
            "order": graded["order"], "dropped": graded["dropped"],
            # the verdict of the gate was computed over these, and a replay must see the same list
            "sources": [_as_row(s) for s in call[3]],
        }
        unaligned += not _keep_only(call, pieces, chunks, graded["kept"])
    ctx["result"].took("grade", started)
    ctx["result"].step(
        "grade", state["hops"], graded=graded_all, kept=kept_all,
        dropped=graded_all - kept_all, unreadable=unreadable, asked=asked,
        # a call whose chunks carry no file: its sources are left whole rather than named wrongly
        sources_unaligned=unaligned or None,
    )
    # nothing survived: its own reason, so no report counts it as the gate firing
    if graded_all and not kept_all and not state.get("coverage"):
        return {"graded_before_emit": plan, "coverage": policy.FallbackReason.graded_out}
    return {"graded_before_emit": plan}


# the only node that speaks to the model, and it speaks after the verdict, never before
def emit_node(state: State, config) -> dict:
    ctx = _ctx(config)
    calls, before = state["pending"], state.get("dropped_before_emit") or {}
    graded = state.get("graded_before_emit") or {}
    return {
        "messages": [
            {"role": "tool", "tool_call_id": tc.id, "content": content}
            for tc, _res, content, _sources in calls
        ] + _restated(ctx, state),
        "sources": chat.stamped([s for c in calls for s in c[3]], state["hops"]),
        "contexts": [
            piece
            for _tc, res, content, _s in calls
            for piece in agent_tools.context_pieces(res.meta, content)
        ],
        "chunks": [
            piece
            for _tc, res, content, _s in calls
            for piece in agent_tools.chunk_pieces(res.meta, content)
        ],
        "spans": [
            {
                "tool_call_id": tc.id,
                "tool": tc.function.name,
                "hop": state["hops"],
                "pieces": len(agent_tools.context_pieces(res.meta, content)),
                # named, not counted: `_unique_sources` collapses a file seen on two hops
                "sources": [s.source for s in sources],
                **({"dropped": before[tc.id]} if tc.id in before else {}),
                **({"graded": graded[tc.id]} if tc.id in graded else {}),
            }
            for tc, res, content, sources in calls
        ],
        # consumed here: it has no reducer, and a later hop reusing a call id took this one's block
        "dropped_before_emit": {},
        "graded_before_emit": {},
    }


def final_node(state: State, config) -> dict:
    ctx = _ctx(config)
    # the loop forces a final turn only when no turn produced text at all
    if state.get("text"):
        ctx["result"].step("final", state["hops"], finished_by=str(policy.FinishedBy.answer))
        return {"finished_by": policy.FinishedBy.answer}
    messages = list(state["messages"])
    update = {}
    if not state.get("sources"):
        messages = messages + [
            {"role": "user", "content": ctx["template"](Purpose.agent_no_evidence)}
        ]
        update["messages"] = [messages[-1]]
        update["no_evidence_prompted"] = True
    # the forced final is reached two ways, and only one of them is the ceiling
    update["finished_by"] = (
        policy.FinishedBy.hops_exhausted
        if state["hops"] >= ctx["max_hops"]
        else policy.FinishedBy.no_answer
    )
    log.info("graph.forcing_final", hops=state["hops"], sources=len(state.get("sources", [])))
    started = time.perf_counter()
    try:
        final = ctx["chat"](messages, role=ctx["role"], model=ctx["model"])
    except (RuntimeError, llm.InputOverWindow) as e:
        log.error("graph.final_failed", error=str(e))
        return update
    ctx["result"].took("model", started)
    ctx["result"].note_prompt(final.prompt_tokens)
    update["messages"] = [
        *update.get("messages", []),
        final.message or {"role": "assistant", "content": final.text or ""},
    ]
    update["answer_parse"] = [*update.get("answer_parse", []), final.parsed]
    update.update(
        hops=state["hops"] + 1,
        prompt_tokens=final.prompt_tokens or 0,
        completion_tokens=final.completion_tokens or 0,
        max_prompt_tokens=final.prompt_tokens or 0,
        text=final.text or "",
    )
    if token_fields.cut(final.finish_reason):
        log.warning("graph.truncated", hops=state["hops"] + 1)
    return update


def _after_model(state: State, config) -> str:
    if state.get("finished"):
        return "final"
    if state.get("awaiting_tools"):
        return "retrieve"
    # a nudge on the last hop must not buy an extra hop the loop would not take
    return "model" if state["hops"] < _ctx(config)["max_hops"] else "final"


# the coverage verdict rides an edge, so a reader of the graph sees the branch the row took
def _after_grade(state: State, config) -> str:
    return "fallback" if state.get("coverage") else "emit"


def _after_tools(state: State, config) -> str:
    return "model" if state["hops"] < _ctx(config)["max_hops"] else "final"


# a hop is up to five super-steps plus the final turn; the slack guards a loop, not a long run
STEPS_PER_HOP = 6
GUARD_SLACK = 6


def recursion_limit(max_hops: int) -> int:
    return STEPS_PER_HOP * max_hops + GUARD_SLACK


def build():
    graph = StateGraph(State)
    graph.add_node("model", model_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("grade", grade_node)
    graph.add_node("fallback", fallback_node)
    graph.add_node("emit", emit_node)
    graph.add_node("final", final_node)
    graph.set_entry_point("model")
    graph.add_conditional_edges(
        "model", _after_model, {"retrieve": "retrieve", "final": "final", "model": "model"}
    )
    graph.add_edge("retrieve", "grade")
    graph.add_conditional_edges(
        "grade", _after_grade, {"fallback": "fallback", "emit": "emit"}
    )
    graph.add_edge("fallback", "emit")
    graph.add_conditional_edges("emit", _after_tools, {"model": "model", "final": "final"})
    graph.add_edge("final", END)
    return graph.compile()


def _initial_state(question: str, system: str, external: bool) -> State:
    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": question},
        ],
        "sources": [],
        "contexts": [],
        "chunks": [],
        "spans": [],
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
        "finished_by": "",
        "awaiting_tools": False,
        "tools_offered": [],
        "pending": [],
        "coverage": "",
        "dropped_before_emit": {},
        "graded_before_emit": {},
        "tool_errors": {},
    }


# every implementation fills the same AgentResult, so nothing downstream can tell them apart
def invoke(question, system, ctx, result) -> None:
    ctx["result"] = result
    # the grader grades against the question, and the state holds it only as the first user message
    ctx["question"] = question
    graph = build()
    try:
        state = graph.invoke(
            _initial_state(question, system, ctx["external"]),
            config={
                "configurable": {"run": ctx},
                "recursion_limit": recursion_limit(ctx["max_hops"]),
            },
        )
    except StandFault:
        raise
    except Exception as e:
        # a run that raises here writes no row at all, and a missing row breaks every pairing
        log.error("graph.failed", error=str(e))
        result.text = ""
        result.success = False
        result.failed = True
        return
    result.messages.clear()
    result.messages.extend(state["messages"])
    result.sources = list(state["sources"])
    result.contexts = list(state["contexts"])
    result.chunks = list(state["chunks"])
    result.spans = list(state["spans"])
    result.tools_offered = list(state.get("tools_offered") or [])
    result.dropped_sources = list(state["dropped_sources"])
    result.dropped_hits = list(state["dropped_hits"])
    result.hops = state["hops"]
    result.prompt_tokens = state["prompt_tokens"]
    result.completion_tokens = state["completion_tokens"]
    result.answer_parse = answer_parsers.summarize(state.get("answer_parse") or [])
    result.max_prompt_tokens = state["max_prompt_tokens"]
    result.text = state.get("text") or ""
    result.finished_by = str(state.get("finished_by") or policy.FinishedBy.answer)
    result.fallback_reason = state.get("fallback_reason", policy.FallbackReason.none)
    result.fallback_opened = state.get("fallback_opened", False)
    result.fallback_announced = state.get("fallback_announced", False)
    result.announced_text = state.get("announced_text") or ""
    result.no_evidence_prompted = state.get("no_evidence_prompted", False)
    result.tool_errors.update(state.get("tool_errors") or {})
    result.success = bool(result.text)


def versions() -> dict:
    from importlib.metadata import version

    out = {}
    for name in ("langgraph", "langchain-core", "langchain"):
        try:
            out[name] = version(name)
        except Exception:  # a missing package is not worth failing a run over
            out[name] = None
    return out


# every door the graph reaches outside itself, so a replay can hand it the row's own answers
def context(**kwargs) -> dict:
    kwargs.setdefault("chat", llm.chat)
    kwargs.setdefault("dispatch", agent_tools.dispatch)
    kwargs.setdefault("template", prompt_repo.active_template)
    return kwargs
