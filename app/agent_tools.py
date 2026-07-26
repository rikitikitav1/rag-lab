import json
from collections.abc import Callable
from dataclasses import dataclass, field

import config
import logging_setup
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


_REGISTRY: dict[str, Tool] = {}


def register(tool: Tool) -> None:
    _REGISTRY[tool.name] = tool


def schemas() -> list[dict]:
    return [t.schema() for t in _REGISTRY.values()]


def dispatch(name: str, arguments: str) -> ToolResult:
    tool = _REGISTRY.get(name)
    if tool is None:
        return ToolResult(content=f"error: unknown tool '{name}'")
    try:
        kwargs = json.loads(arguments or "{}")
    except json.JSONDecodeError as e:
        return ToolResult(content=f"error: invalid arguments json: {e}")
    try:
        return tool.run(**kwargs)
    except Exception as e:
        log.error("tool.failed", tool=name, error=str(e))
        return ToolResult(content=f"error: tool '{name}' failed")


def _search_corpus(
    query: str,
    category: str | None = None,
) -> ToolResult:
    rows = chat._retrieve_rows(
        question=query,
        category=category,
        k=config.settings.retrieval.results_limit,
        rerank_enabled=config.settings.rerank.enabled,
    )

    if not rows:
        return ToolResult(
            content="No relevant documents found.",
            meta={"sources": []},
        )

    content = "\n\n".join(
        f"[{src}]\n{content}"
        for content, src, *_ in rows
        if not chat.is_ignored_source(src)
    )

    return ToolResult(
        content=content or "No relevant documents found.",
        meta={
            "sources": chat.take_sources(rows),
        },
    )


register(
    Tool(
        name="search_corpus",
        description=(
            "Search the technical knowledge corpus (interview banks, "
            "system-design-primer, redis docs) and return the most relevant "
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
