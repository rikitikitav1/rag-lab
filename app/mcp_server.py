from typing import Literal

import logging_setup
from fastmcp import FastMCP
from models.registry import Pipeline
from sqlalchemy.exc import SQLAlchemyError
from use_cases import agent, chat

import db

log = logging_setup.get_logger(__name__)

mcp = FastMCP("rag-lab", mask_error_details=True)

_TOOL_DESC = {
    "search_corpus": (
        "Search the technical knowledge corpus (interview banks, "
        "system-design-primer, redis docs) and return the most relevant "
        "chunks with their [source] markers. Optionally restrict the search to a "
        "category subtree (call list_categories first to discover valid paths)."
    ),
    "answer_question": (
        "Answer a question from the technical knowledge corpus: retrieves "
        "relevant context and returns {answer, sources} where sources is the list "
        "of source paths the answer drew on. Use this when you want a direct "
        "answer rather than raw chunks (for raw chunks use search_corpus instead). "
        "The 'agent' pipeline reformulates the query and searches over multiple "
        "hops (better recall); 'single_shot' does a single retrieval pass (faster)."
    ),
    "list_categories": (
        "List the category paths present in the corpus with their chunk counts "
        "(ltree paths, e.g. 'databases.redis'). Use this to discover valid "
        "category values before filtering a search, instead of guessing. Pass a "
        "category to list paths under it, or set only_top=true for just the "
        "top-level categories."
    ),
}


@mcp.tool(name="search_corpus", description=_TOOL_DESC["search_corpus"])
def search_corpus(query: str, category: str | None = None) -> str:
    try:
        content, _ = chat.search_chunks(query, category)
        return content
    except Exception as e:
        log.error("mcp.search_failed", error=str(e))
        return f"{chat.ERROR_PREFIX}search failed"


@mcp.tool(name="answer_question", description=_TOOL_DESC["answer_question"])
def answer_question(
    question: str,
    pipeline: Pipeline = Pipeline.agent,
    language: Literal["ru", "en"] | None = None,
) -> dict:
    try:
        if pipeline == Pipeline.agent:
            res = agent.run(question, run_name="mcp", language=language)
        else:
            res = chat.answer(question, run_name="mcp", language=language)
    except Exception as e:
        log.error("mcp.answer_failed", error=str(e))
        return {"answer": f"{chat.ERROR_PREFIX}answer generation failed", "sources": []}
    return {
        "answer": res.text or "No answer generated.",
        "sources": [s.source for s in res.sources],
    }


@mcp.tool(name="list_categories", description=_TOOL_DESC["list_categories"])
def list_categories(category: str | None = None, only_top: bool = False) -> dict[str, int]:
    try:
        rows = db.list_categories(only_top=only_top, category=category)
    except SQLAlchemyError as e:
        log.error("mcp.list_categories_failed", error=str(e))
        return {}
    return {row[0]: row[1] for row in rows}
