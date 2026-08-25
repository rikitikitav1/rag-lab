import agent_tools
import config
import llm
import logging_setup
from langchain_core.tools import StructuredTool
from orchestrators import middleware as orch_middleware

log = logging_setup.get_logger(__name__)


def chat_model(role: str = "generation", model: str | None = None):
    from langchain_ollama import ChatOllama

    opts = config.settings.llm.roles[role].options
    return ChatOllama(
        base_url=config.settings.llm.base_url,
        model=model or llm.resolve_name(role),
        temperature=opts.get("temperature"),
        num_predict=opts.get("max_tokens"),
    )


# the standard two-layer tool contract: content goes to the model, artifact rides along on the
# ToolMessage for the pipeline. Our error kinds have no place in it, only success or error
def as_tools(remote: dict, k=None, use_rerank=None) -> list:
    def make(name: str, tool):
        def call(**kwargs) -> tuple[str, list]:
            import json

            res = agent_tools.dispatch(
                name, json.dumps(kwargs), extra=remote, k=k, use_rerank=use_rerank
            )
            return res.content, res.meta.get("sources", [])

        return StructuredTool.from_function(
            func=call,
            name=name,
            description=tool.description,
            args_schema=tool.parameters,
            response_format="content_and_artifact",
        )

    tools = [make(t.name, t) for t in agent_tools.registry()]
    return tools + [make(name, tool) for name, tool in remote.items()]


def invoke(question: str, system: str, ctx: dict, result, middleware=None, run=None) -> None:
    from langchain.agents import create_agent

    tools = (
        orch_middleware.tools_for(run, k=ctx["k"], use_rerank=ctx["use_rerank"])
        if run is not None
        else as_tools(ctx["remote"], k=ctx["k"], use_rerank=ctx["use_rerank"])
    )
    agent = create_agent(
        model=ctx.get("model_client") or chat_model(ctx["role"], ctx["model"]),
        tools=tools,
        system_prompt=system,
        **({"middleware": middleware} if middleware else {}),
    )
    from langgraph.errors import GraphRecursionError

    try:
        # a bare agent spends two super-steps per hop; the middleware hooks add their own
        limit = (4 * ctx["max_hops"] + 8) if middleware else (2 * ctx["max_hops"] + 1)
        state = agent.invoke(
            {"messages": [{"role": "user", "content": question}]},
            config={"recursion_limit": limit},
        )
    except GraphRecursionError:
        # without this the question vanishes from the run instead of counting as exhausted
        log.warning("react.recursion_limit", max_hops=ctx["max_hops"])
        result.hops = ctx["max_hops"]
        result.text = ""
        result.success = False
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
        result.prompt_tokens += usage.get("input_tokens") or 0
        result.completion_tokens += usage.get("output_tokens") or 0
        result.max_prompt_tokens = max(result.max_prompt_tokens, usage.get("input_tokens") or 0)
    log.info("react.done", hops=result.hops, sources=len(result.sources))
