import agent_tools
import config
import llm
import logging_setup
from langchain_core.tools import StructuredTool
from use_cases import agent_policy as policy

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


def invoke(question: str, system: str, ctx: dict, result) -> None:
    from langchain.agents import create_agent

    tools = as_tools(ctx["remote"], k=ctx["k"], use_rerank=ctx["use_rerank"])
    agent = create_agent(
        model=ctx.get("model_client") or chat_model(ctx["role"], ctx["model"]),
        tools=tools,
        system_prompt=system,
    )
    state = agent.invoke(
        {"messages": [{"role": "user", "content": question}]},
        config={"recursion_limit": 2 * ctx["max_hops"] + 1},
    )
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
    result.sources = list(collected)
    result.hops = len(replies)
    result.text = str(replies[-1].content) if replies else ""
    result.success = bool(result.text)
    result.fallback_reason = policy.FallbackReason.none
    for reply in replies:
        usage = getattr(reply, "usage_metadata", None) or {}
        result.prompt_tokens += usage.get("input_tokens") or 0
        result.completion_tokens += usage.get("output_tokens") or 0
        result.max_prompt_tokens = max(result.max_prompt_tokens, usage.get("input_tokens") or 0)
    log.info("react.done", hops=result.hops, sources=len(result.sources))
