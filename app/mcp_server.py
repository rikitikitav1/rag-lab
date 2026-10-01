from typing import Annotated, Literal

import config
import logging_setup
from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from models.registry import Pipeline
from pydantic import BaseModel, Field
from search_scope import MAX_SOURCES
from sqlalchemy.exc import SQLAlchemyError
from use_cases import agent, card_wait, chat

import db

log = logging_setup.get_logger(__name__)

mcp = FastMCP("rag-lab", mask_error_details=True)

_MAX_QUERY_LEN = 2000


class AnswerResult(BaseModel):
    answer: str
    retrieved: bool
    sources: list[str]


def _check_text(value: str, field: str) -> None:
    if not value.strip():
        raise ToolError(f"{field} must not be empty")
    if len(value) > _MAX_QUERY_LEN:
        raise ToolError(f"{field} too long (max {_MAX_QUERY_LEN} chars)")


def _safe_category(category: str | None) -> str | None:
    try:
        db.refuse_bad_category(category)
    except ValueError as e:
        raise ToolError(str(e)) from e
    return category


def _safe_scope(category: str | None, sources: list[str] | None, version: str | None) -> db.Scope:
    scope = db.Scope.of(category, sources, version)
    try:
        db.refuse_bad_scope(scope)
    except ValueError as e:
        raise ToolError(str(e)) from e
    return scope


_SOURCES = Field(
    description="Optional source names; the search reads only them. The `sources` ops tool lists them.",
    max_length=MAX_SOURCES,
)
_VERSION = Field(
    description="Optional released version of the category named in `category` (e.g. '17' with 'postgresql'): its "
    "docs plus the category's sources of no version, such as books; without it a versioned category reads its newest."
)


_TOOL_DESC = {
    "search_corpus": (
        "Search the technical knowledge corpus (its sources: the ops tool `sources`, stage accepted) "
        "and return the most relevant chunks "
        "with their [source] markers. Optionally filter by category."
    ),
    "answer_question": (
        "Answer a question from the technical knowledge corpus. Returns "
        "{answer, retrieved, sources}: retrieved=true means context was found and "
        "fed to the model, false means it had nothing to ground on. retrieved does "
        "NOT certify the answer is correct or fully supported, so judge the answer "
        "and its sources yourself. sources lists the paths retrieved as context, "
        "not necessarily the ones the answer rests on. For raw chunks "
        "use search_corpus instead. The 'agent' pipeline reformulates and searches "
        "over multiple hops (better recall); 'single_shot' does one pass (faster)."
    ),
    "list_categories": (
        "List the corpus's categories by name with chunk counts; chunks of no "
        "category count under 'none'. With only_top=true the keys are groups and the counts their totals "
        "(cannot be combined with a category filter). The filter also takes a tag: list_tags shows them. "
        "Counts cover every version, while a search with no version reads the newest."
    ),
    "list_tags": (
        "List the corpus's tags (file paths' folders, a sheet's own category, a bank's topic) with chunk counts, "
        "most used first. Any of them is a label the category filter takes."
    ),
}


# the same guard as the REST chat: an MCP question reached ollama beside an awake judge
def _wait_for_the_card(*roles) -> None:
    try:
        card_wait.wait_for_the_card(*roles)
    except card_wait.CardBusy as e:
        raise ToolError(e.detail) from e


@mcp.tool(
    name="search_corpus",
    description=_TOOL_DESC["search_corpus"],
    annotations={"readOnlyHint": True},
)
def search_corpus(
    query: Annotated[str, Field(description="Search query, phrased for retrieval.")],
    category: Annotated[
        str | None,
        Field(
            description="Optional filter, a literal label: a category (e.g. 'redis'), a group "
            "of them (e.g. 'databases') or a tag. Call list_categories to discover valid labels."
        ),
    ] = None,
    sources: Annotated[list[str] | None, _SOURCES] = None,
    version: Annotated[str | None, _VERSION] = None,
) -> str:
    _check_text(query, "query")
    scope = _safe_scope(category, sources, version)
    _wait_for_the_card(*card_wait.retrieving_roles())
    try:
        content, _texts, _sources, _depth, _chunks = chat.search_chunks(
            query, scope, variant=config.settings.corpus.variant
        )
    except (db.ForeignVectors, db.ScopeRefused) as e:
        raise ToolError(str(e)) from e
    return content


@mcp.tool(
    name="answer_question",
    description=_TOOL_DESC["answer_question"],
    annotations={"readOnlyHint": True},
)
def answer_question(
    text: Annotated[str, Field(description="The question to answer.")],
    pipeline: Annotated[
        Pipeline,
        Field(description="'agent' (multi-hop, better recall) or 'single_shot' (faster)."),
    ] = Pipeline.agent,
    category: Annotated[
        str | None,
        Field(description="Optional literal category label; only with pipeline=single_shot."),
    ] = None,
    sources: Annotated[list[str] | None, _SOURCES] = None,
    version: Annotated[str | None, _VERSION] = None,
    language: Annotated[
        Literal["ru", "en"] | None,
        Field(description="Force answer language: 'ru' or 'en'."),
    ] = None,
) -> AnswerResult:
    _check_text(text, "text")
    scope = _safe_scope(category, sources, version)
    if pipeline == Pipeline.agent and scope.narrowed:
        raise ToolError("a category, source or version filter is only supported with pipeline=single_shot")
    _wait_for_the_card(*card_wait.answering_roles(agent=pipeline == Pipeline.agent))
    try:
        if pipeline == Pipeline.agent:
            res = agent.run(text, run_name="mcp", language=language)
        else:
            res = chat.answer(text, scope=scope, run_name="mcp", language=language)
    except (db.ForeignVectors, db.ScopeRefused) as e:
        raise ToolError(str(e)) from e
    except Exception as e:
        log.error("mcp.answer_failed", error=str(e))
        raise
    return AnswerResult(
        answer=res.text or "No answer generated.",
        retrieved=bool(res.success),
        sources=[s.source for s in res.sources],
    )


@mcp.tool(
    name="list_categories",
    description=_TOOL_DESC["list_categories"],
    annotations={"readOnlyHint": True},
)
def list_categories(
    category: Annotated[str | None, Field(description="Optional category or group to list under.")] = None,
    only_top: Annotated[bool, Field(description="If true, totals per group.")] = False,
) -> dict[str, int]:
    category = _safe_category(category)
    if only_top and category:
        raise ToolError("only_top cannot be combined with a category filter")
    try:
        rows = db.list_categories(only_top=only_top, category=category, variant=config.settings.corpus.variant)
    except SQLAlchemyError as e:
        log.error("mcp.list_categories_failed", error=str(e))
        raise
    return {name: n for name, _, n in rows}


@mcp.tool(
    name="list_tags",
    description=_TOOL_DESC["list_tags"],
    annotations={"readOnlyHint": True},
)
def list_tags(
    limit: Annotated[int, Field(ge=1, le=1000, description="How many tags, most used first.")] = 50,
) -> dict[str, int]:
    return {name: n for name, n in db.list_tags(limit, variant=config.settings.corpus.variant)}
