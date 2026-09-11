import json
import time

import agent_tools
import engines
import engines.ollama
import llm
import logging_setup
from errors import StandFault
from langchain_core.tools import StructuredTool
from use_cases import agent_policy as policy
from use_cases import chat

log = logging_setup.get_logger(__name__)


def truncated(message) -> bool:
    meta = getattr(message, "response_metadata", None) or {}
    return meta.get("finish_reason") == "length" or meta.get("done_reason") == "length"


# the third client in the tree, and the only one that used to read the address out of the config
def chat_model(role: str = "generation", model: str | None = None):
    from langchain_ollama import ChatOllama

    picked = llm.resolve_for(role, model)
    engines.ollama.refuse_unless_ollama(picked.engine, "the idiomatic orchestrator")
    sent = llm.sampler(role, picked.engine).sent
    return ChatOllama(
        base_url=engines.base_url(picked.engine),
        model=picked.name,
        temperature=sent.get("temperature"),
        num_predict=sent.get("max_tokens"),
        seed=sent.get("seed"),
        # ChatOllama has no retry of its own, so this is one attempt where our client takes two
        client_kwargs={"timeout": engines.LLM_TIMEOUT},
    )


# two-layer contract: content to the model, artifact to the pipeline. No room for error kinds
def as_tools(remote: dict, k=None, use_rerank=None, result=None, variant=None) -> list:
    def make(name: str, tool):
        def call(**kwargs) -> tuple[str, dict]:
            # the tool node is built once from every tool, so the gate has to refuse here
            started = time.perf_counter()
            res = agent_tools.dispatch(
                name, json.dumps(kwargs), extra=remote, k=k, use_rerank=use_rerank,
                gate_top=None, variant=variant,
            )
            if result is not None:
                result.took(f"tool:{name.split('__')[0]}", started)
                if res.meta.get("error_kind"):
                    result.tool_errors[name] = res.meta["error_kind"]
                if res.meta.get("ef_search") is not None:
                    result.ef_search = res.meta["ef_search"]
            # the whole meta: the chunks a hop read are on it, and this artifact is the only way out
            return res.content, res.meta

        return StructuredTool.from_function(
            func=call,
            name=name,
            description=tool.description,
            args_schema=tool.parameters,
            response_format="content_and_artifact",
        )

    tools = [make(t.name, t) for t in agent_tools.registry()]
    return tools + [make(name, tool) for name, tool in remote.items()]


# two steps per hop plus the answer, no final turn, so this limit IS the hop budget
BARE_STEPS_PER_HOP = 2
BARE_ANSWER_STEP = 1


def recursion_limit(max_hops: int) -> int:
    return BARE_STEPS_PER_HOP * max_hops + BARE_ANSWER_STEP


# langchain says `ai` and `human` where the rest of the stand says `assistant` and `user`
_ROLES = {"ai": "assistant", "human": "user", "system": "system", "tool": "tool"}


# one shape of message on both arms: the reader of a transcript must not learn two dialects
def _as_message(m) -> dict:
    kind = getattr(m, "type", None)
    out = {"role": _ROLES.get(kind, kind or "unknown"), "content": str(m.content)}
    calls = getattr(m, "tool_calls", None) or []
    if calls:
        out["tool_calls"] = [
            {"function": {"name": c.get("name"), "arguments": json.dumps(c.get("args") or {})}}
            for c in calls
        ]
    if getattr(m, "tool_call_id", None):
        out["tool_call_id"] = m.tool_call_id
    return out


def invoke(question: str, system: str, ctx: dict, result) -> None:
    from langchain.agents import create_agent
    from langgraph.errors import GraphRecursionError

    limit = recursion_limit(ctx["max_hops"])
    try:
        tools = as_tools(
            ctx["remote"], k=ctx["k"], use_rerank=ctx["use_rerank"], result=result,
            variant=ctx["variant"],
        )
        agent = create_agent(
            model=ctx.get("model_client") or chat_model(ctx["role"], ctx["model"]),
            tools=tools,
            system_prompt=system,
        )
        state = agent.invoke(
            {"messages": [{"role": "user", "content": question}]},
            config={"recursion_limit": limit},
        )
    except GraphRecursionError:
        log.error("react.recursion_limit", max_hops=ctx["max_hops"], limit=limit)
        result.hops = ctx["max_hops"] + 1
        result.text = ""
        result.success = False
        # for this arm the limit is the budget, so reaching it is exhaustion, not a failure
        result.failed = False
        return
    except StandFault:
        raise
    except Exception as e:
        log.error("react.client_failed", error=str(e))
        result.text = ""
        result.success = False
        result.failed = True
        return
    messages = state["messages"]
    replies = [m for m in messages if getattr(m, "type", None) == "ai"]
    # the hop a tool answered on is the number of replies before it
    hop, collected, contexts, chunks, spans = 0, [], [], [], []
    for message in messages:
        kind = getattr(message, "type", None)
        if kind == "ai":
            hop += 1
        elif kind == "tool":
            meta = getattr(message, "artifact", None) or {}
            pieces = agent_tools.context_pieces(meta, str(message.content))
            collected += chat.stamped(meta.get("sources") or [], hop)
            contexts += pieces
            chunks += agent_tools.chunk_pieces(meta, str(message.content))
            spans.append({
                "tool_call_id": getattr(message, "tool_call_id", None),
                "tool": getattr(message, "name", None),
                "hop": hop,
                "pieces": len(pieces),
                "sources": [s.source for s in (meta.get("sources") or [])],
            })
    result.messages.clear()
    result.messages.extend(_as_message(m) for m in messages)
    result.sources = list(collected)
    result.spans = spans
    result.contexts = contexts
    result.chunks = chunks
    result.hops = len(replies)
    # a bare `create_agent` has no edge of ours to record, and a silent gap reads as a fault
    result.finished_by = str(policy.FinishedBy.unrecorded)
    result.text = str(replies[-1].content) if replies else ""
    result.success = bool(result.text)
    for reply in replies:
        usage = getattr(reply, "usage_metadata", None) or {}
        # only this client reports where a call went: reading the prompt or writing the answer
        result.note_server_timings(getattr(reply, "response_metadata", None) or {})
        result.note_prompt(usage.get("input_tokens"))
        result.prompt_tokens += usage.get("input_tokens") or 0
        result.completion_tokens += usage.get("output_tokens") or 0
        result.max_prompt_tokens = max(result.max_prompt_tokens, usage.get("input_tokens") or 0)
    if replies and truncated(replies[-1]):
        log.warning("react.truncated", hops=result.hops)
    # without usage the arm reports zero context tokens and no trimming, which reads as a clean run
    if replies and not result.prompt_tokens:
        log.warning("react.no_token_usage", hops=result.hops)
    log.info("react.done", hops=result.hops, sources=len(result.sources))
