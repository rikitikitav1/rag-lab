import asyncio
import inspect
import json
from collections.abc import Callable
from dataclasses import dataclass, field

import config
import errors
import logging_setup
import mcp_client
from models.mcp_integration import McpIntegration, McpStatus
from orm.sync_db import Session
from sqlalchemy import select
from use_cases import chat

log = logging_setup.get_logger(__name__)


@dataclass
class ToolResult:
    content: str
    meta: dict = field(default_factory=dict)


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict
    run: Callable[..., ToolResult]

    def schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


CORPUS_TOOL = "search_corpus"

_REGISTRY: dict[str, Tool] = {}


def register(tool: Tool) -> None:
    _REGISTRY[tool.name] = tool


def schemas() -> list[dict]:
    return [t.schema() for t in _REGISTRY.values()]


def registry() -> list:
    return list(_REGISTRY.values())


def dispatch(
    name: str, arguments: str, extra: dict[str, Tool] | None = None, **runtime
) -> ToolResult:
    tool = (extra or {}).get(name) or _REGISTRY.get(name)
    if tool is None:
        return ToolResult(
            content=f"{errors.ERROR_PREFIX}unknown tool '{name}'", meta={"error_kind": "client"}
        )
    try:
        raw = json.loads(arguments or "{}")
    except json.JSONDecodeError:
        return ToolResult(
            content=f"{errors.ERROR_PREFIX}tool '{name}' got invalid arguments",
            meta={"error_kind": "client"},
        )
    allowed = tool.parameters.get("properties")
    if allowed is None:
        # remote schemas may keep args under $ref/anyOf; pass through untouched
        kwargs = dict(raw)
    else:
        kwargs = {k: v for k, v in raw.items() if k in allowed}
        dropped = [k for k in raw if k not in allowed]
        if dropped:
            log.warning("tool.dropped_args", tool=name, dropped=dropped)
        missing = [p for p in tool.parameters.get("required", []) if p not in kwargs]
        if missing:
            log.warning("tool.missing_args", tool=name, missing=missing)
            return ToolResult(
                content=(
                    f"{errors.ERROR_PREFIX}tool '{name}' is missing required arguments "
                    f"{missing}; it accepts: {', '.join(allowed)}"
                ),
                meta={"error_kind": "client"},
            )
    accepted = inspect.signature(tool.run).parameters
    kwargs.update({key: val for key, val in runtime.items() if val is not None and key in accepted})
    try:
        return tool.run(**kwargs)
    except Exception as e:
        log.error("tool.failed", tool=name, error=str(e))
        return ToolResult(
            content=f"{errors.ERROR_PREFIX}tool '{name}' failed", meta={"error_kind": "tool"}
        )


def _search_corpus(
    query: str,
    category: str | None = None,
    k: int | None = None,
    use_rerank: bool | None = None,
    gate_top: int | None = None,
    variant: str | None = None,
) -> ToolResult:
    # dispatch drops runtime values that are None, so the orchestrator always supplies this one
    content, sources = chat.search_chunks(
        query, category, k=k, use_rerank=use_rerank, gate_top=gate_top,
        variant=variant or config.settings.corpus.variant,
    )
    return ToolResult(content=content, meta={"sources": sources})


def _remote_run(integration, tool_name):
    def run(**kwargs) -> ToolResult:
        outcome = asyncio.run(mcp_client.call_tool(integration, tool_name, kwargs))
        text = outcome.text
        if not text.strip():
            return ToolResult(
                content=f"{errors.ERROR_PREFIX}tool '{tool_name}' returned nothing",
                meta={"error_kind": "empty"},
            )
        if outcome.error_kind or text.startswith(errors.ERROR_PREFIX):
            return ToolResult(content=text, meta={"error_kind": outcome.error_kind})
        source = chat.Source(
            source=f"mcp:{integration.name}__{tool_name}",
            vector_rank=None,
            keyword_rank=None,
            vector_distance=None,
            score=0.0,
        )
        return ToolResult(content=text, meta={"sources": [source]})

    return run


def remote_tools() -> list[Tool]:
    try:
        with Session() as session:
            stmt = select(McpIntegration).where(
                McpIntegration.status == McpStatus.active
            )
            mcp_integrations = session.scalars(stmt).all()
    except Exception as e:
        log.error("remote_tools.load_failed", error=str(e))
        return []

    result = []
    for mcp_int in mcp_integrations:
        schema_cache = mcp_int.tool_schemas
        for tool_name in mcp_int.allowed_tools:
            if tool_name not in schema_cache:
                log.warning(
                    "remote_tools.schema_missing",
                    integration=mcp_int.name,
                    tool=tool_name,
                )
                continue
            result.append(
                Tool(
                    name=f"{mcp_int.name}__{tool_name}",
                    description=schema_cache[tool_name]["description"],
                    parameters=schema_cache[tool_name]["parameters"],
                    run=_remote_run(mcp_int, tool_name),
                )
            )

    return result


register(
    Tool(
        name=CORPUS_TOOL,
        description=(
            f"Search {config.settings.corpus.description} and return the most relevant "
            "chunks with their [source] markers. Call this before answering."
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query, phrased for retrieval.",
                },
            },
            "required": ["query"],
        },
        run=_search_corpus,
    )
)
