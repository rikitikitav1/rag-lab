import re
from typing import Annotated, Literal

import config
import logging_setup
from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from models.registry import Pipeline
from pydantic import BaseModel, Field
from sqlalchemy.exc import SQLAlchemyError
from use_cases import agent, chat

import db

log = logging_setup.get_logger(__name__)

mcp = FastMCP("rag-lab", mask_error_details=True)

_MAX_QUERY_LEN = 2000
_CATEGORY_RE = re.compile(r"^[a-zA-Z0-9_.-]+$")


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
    if category is not None and not _CATEGORY_RE.match(category):
        raise ToolError("invalid category filter")
    return category


_TOOL_DESC = {
    "search_corpus": (
        "Search the technical knowledge corpus (interview banks, "
        "system-design-primer, redis docs) and return the most relevant chunks "
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
        "List category paths present in the corpus with chunk counts. With "
        "only_top=true the counts are subtree totals per top-level category "
        "(cannot be combined with a category filter); otherwise they are exact "
        "per-path counts. Use to discover valid category values before filtering."
    ),
}


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
            description="Optional category filter, a literal label matched anywhere in "
            "the ltree path (e.g. 'redis' or 'databases.redis'). Call list_categories "
            "to discover valid labels."
        ),
    ] = None,
) -> str:
    _check_text(query, "query")
    category = _safe_category(category)
    content, _ = chat.search_chunks(
        query, category, variant=config.settings.corpus.variant
    )
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
    language: Annotated[
        Literal["ru", "en"] | None,
        Field(description="Force answer language: 'ru' or 'en'."),
    ] = None,
) -> AnswerResult:
    _check_text(text, "text")
    category = _safe_category(category)
    if pipeline == Pipeline.agent and category:
        raise ToolError("category filter is only supported with pipeline=single_shot")
    try:
        if pipeline == Pipeline.agent:
            res = agent.run(text, run_name="mcp", language=language)
        else:
            res = chat.answer(text, category=category, run_name="mcp", language=language)
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
    category: Annotated[
        str | None, Field(description="Optional literal label to list paths under.")
    ] = None,
    only_top: Annotated[
        bool, Field(description="If true, top-level categories with subtree totals.")
    ] = False,
) -> dict[str, int]:
    category = _safe_category(category)
    if only_top and category:
        raise ToolError("only_top cannot be combined with a category filter")
    try:
        rows = db.list_categories(only_top=only_top, category=category, variant=config.settings.corpus.variant)
    except SQLAlchemyError as e:
        log.error("mcp.list_categories_failed", error=str(e))
        raise
    return {row[0]: row[1] for row in rows}
