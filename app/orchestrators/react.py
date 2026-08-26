import json

import agent_tools
import config
import llm
import logging_setup
from langchain_core.tools import StructuredTool

log = logging_setup.get_logger(__name__)


def truncated(message) -> bool:
    meta = getattr(message, "response_metadata", None) or {}
    return meta.get("finish_reason") == "length" or meta.get("done_reason") == "length"


def chat_model(role: str = "generation", model: str | None = None):
    from langchain_ollama import ChatOllama

    opts = config.settings.llm.roles[role].options
    return ChatOllama(
        base_url=config.settings.llm.base_url,
        model=model or llm.resolve_name(role),
        temperature=opts.get("temperature"),
        num_predict=opts.get("max_tokens"),
        # our client gives up after this long, and an arm that waits forever is not comparable.
        # ChatOllama has no retry of its own, so the loop still gets one attempt more than this
        client_kwargs={"timeout": llm.LLM_TIMEOUT},
    )


# the standard two-layer tool contract: content goes to the model, artifact rides along on the
# ToolMessage for the pipeline. Our error kinds have no place in it, only success or error
def as_tools(remote: dict, k=None, use_rerank=None, run=None) -> list:
    def make(name: str, tool):
        def call(**kwargs) -> tuple[str, list]:
            # the tool node is built once from every tool, so a name the gate has not opened yet
            # still reaches dispatch: whoever runs a gate has to refuse it here
            extra = (run.remote if run.external else None) if run else remote
            res = agent_tools.dispatch(
                name, json.dumps(kwargs), extra=extra, k=k, use_rerank=use_rerank,
                gate_top=run.gate.top if run else None,
            )
            if run is not None and res.meta.get("error_kind"):
                run.tool_errors[name] = res.meta["error_kind"]
            return res.content, res.meta.get("sources", [])

        return StructuredTool.from_function(
            func=call,
            name=name,
            description=tool.description,
            args_schema=tool.parameters,
            response_format="content_and_artifact",
        )

    offered = run.remote if run is not None else remote
    tools = [make(t.name, t) for t in agent_tools.registry()]
    return tools + [make(name, tool) for name, tool in offered.items()]


def invoke(question: str, system: str, ctx: dict, result, middleware=None, run=None) -> None:
    from langchain.agents import create_agent
    from langgraph.errors import GraphRecursionError

    # a bare agent spends two super-steps per hop; the middleware hooks add their own
    limit = (12 * ctx["max_hops"] + 24) if middleware else (2 * ctx["max_hops"] + 1)
    try:
        tools = as_tools(ctx["remote"], k=ctx["k"], use_rerank=ctx["use_rerank"], run=run)
        agent = create_agent(
            model=ctx.get("model_client") or chat_model(ctx["role"], ctx["model"]),
            tools=tools,
            system_prompt=system,
            **({"middleware": middleware} if middleware else {}),
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
        # without middleware the limit IS the hop budget: the bare arm has no final turn, so
        # running out of hops there is exhaustion, not a guard that caught a loop
        result.failed = middleware is not None
        return
    except Exception as e:
        log.error("react.client_failed", error=str(e))
        result.text = ""
        result.success = False
        result.failed = True
        return
    messages = state["messages"]
    replies = [m for m in messages if getattr(m, "type", None) == "ai"]
    collected = [
        source
        for m in messages
        if getattr(m, "type", None) == "tool"
        for source in (getattr(m, "artifact", None) or [])
    ]
    result.messages.clear()
    result.messages.extend(
        {"role": getattr(m, "type", "unknown"), "content": str(m.content)} for m in messages
    )
    result.sources = list(run.sources if run is not None else collected)
    if run is not None:
        result.dropped_sources = list(run.dropped)
        result.dropped_hits = list(run.dropped_hits)
        result.fallback_reason = run.fallback_reason
        result.fallback_opened = run.fallback_opened
        result.fallback_announced = run.announced
        result.no_evidence_prompted = run.no_evidence_prompted
        result.tool_errors.update(run.tool_errors)
    result.hops = len(replies)
    result.text = str(replies[-1].content) if replies else ""
    result.success = bool(result.text)
    for reply in replies:
        usage = getattr(reply, "usage_metadata", None) or {}
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
