import asyncio
from types import SimpleNamespace

import mcp_server
import pytest
from fastmcp import Client
from fastmcp.exceptions import ToolError
from models.registry import Pipeline


def test_search_corpus_returns_content(monkeypatch):
    monkeypatch.setattr(mcp_server.chat, "search_chunks", lambda q, category=None: ("chunks", []))
    assert mcp_server.search_corpus("redis") == "chunks"


def test_search_corpus_forwards_category(monkeypatch):
    seen = {}

    def fake(q, category=None):
        seen["category"] = category
        return ("c", [])

    monkeypatch.setattr(mcp_server.chat, "search_chunks", fake)
    mcp_server.search_corpus("redis", category="databases.redis")
    assert seen["category"] == "databases.redis"


def test_search_corpus_empty_query_raises():
    with pytest.raises(ToolError):
        mcp_server.search_corpus("   ")


def test_search_corpus_too_long_query_raises():
    with pytest.raises(ToolError):
        mcp_server.search_corpus("x" * (mcp_server._MAX_QUERY_LEN + 1))


@pytest.mark.parametrize("bad", ["(", "*", "databases|interview", "!databases", "a b"])
def test_search_corpus_rejects_bad_category(bad):
    with pytest.raises(ToolError) as ei:
        mcp_server.search_corpus("x", category=bad)
    assert "invalid category" in str(ei.value)


def test_search_corpus_masks_unexpected_through_client(monkeypatch):
    def boom(q, category=None):
        raise RuntimeError("SELECT secret_sql FROM data_chunks; embedding=[0.1]")

    monkeypatch.setattr(mcp_server.chat, "search_chunks", boom)

    async def go():
        async with Client(mcp_server.mcp) as c:
            return await c.call_tool("search_corpus", {"query": "x"})

    with pytest.raises(ToolError) as ei:
        asyncio.run(go())
    assert "secret_sql" not in str(ei.value)


def test_answer_question_agent_pipeline(monkeypatch):
    seen = {}

    def fake_run(q, run_name=None, language=None):
        seen["run_name"] = run_name
        return SimpleNamespace(text="agent answer", success=True, sources=[])

    monkeypatch.setattr(mcp_server.agent, "run", fake_run)
    out = mcp_server.answer_question("q")
    assert out.answer == "agent answer"
    assert out.retrieved is True
    assert out.sources == []
    assert seen["run_name"] == "mcp"


def test_answer_question_retrieved_false_no_evidence(monkeypatch):
    monkeypatch.setattr(
        mcp_server.agent,
        "run",
        lambda q, run_name=None, language=None: SimpleNamespace(
            text="fabricated", success=False, sources=[]
        ),
    )
    out = mcp_server.answer_question("q")
    assert out.retrieved is False
    assert out.sources == []
    assert out.answer == "fabricated"


def test_answer_question_returns_sources(monkeypatch):
    srcs = [SimpleNamespace(source="a.md"), SimpleNamespace(source="b.md")]
    monkeypatch.setattr(
        mcp_server.agent,
        "run",
        lambda q, run_name=None, language=None: SimpleNamespace(
            text="ans", success=True, sources=srcs
        ),
    )
    assert mcp_server.answer_question("q").sources == ["a.md", "b.md"]


def test_answer_question_single_shot_forwards_category(monkeypatch):
    seen = {}

    def fake_answer(text, category=None, run_name=None, language=None):
        seen["category"] = category
        return SimpleNamespace(text="a", success=True, sources=[])

    monkeypatch.setattr(mcp_server.chat, "answer", fake_answer)
    mcp_server.answer_question("q", pipeline=Pipeline.single_shot, category="redis")
    assert seen["category"] == "redis"


def test_answer_question_empty_text_raises():
    with pytest.raises(ToolError):
        mcp_server.answer_question("  ")


def test_answer_question_agent_rejects_category():
    with pytest.raises(ToolError):
        mcp_server.answer_question("q", pipeline=Pipeline.agent, category="redis")


def test_answer_question_bad_category_raises():
    with pytest.raises(ToolError) as ei:
        mcp_server.answer_question("q", pipeline=Pipeline.single_shot, category="*")
    assert "invalid category" in str(ei.value)


def test_answer_question_empty_text_fallback(monkeypatch):
    monkeypatch.setattr(
        mcp_server.agent,
        "run",
        lambda q, run_name=None, language=None: SimpleNamespace(
            text="", success=False, sources=[]
        ),
    )
    assert mcp_server.answer_question("q").answer == "No answer generated."


def test_answer_question_error_masks_through_client(monkeypatch):
    def boom(q, run_name=None, language=None):
        raise RuntimeError("secret detail sql")

    monkeypatch.setattr(mcp_server.agent, "run", boom)

    async def go():
        async with Client(mcp_server.mcp) as c:
            return await c.call_tool("answer_question", {"text": "q"})

    with pytest.raises(ToolError) as ei:
        asyncio.run(go())
    assert "secret detail" not in str(ei.value)


def test_list_categories_maps_rows_with_counts(monkeypatch):
    monkeypatch.setattr(
        mcp_server.db,
        "list_categories",
        lambda only_top, category: [("databases", 5), ("llm", 3)],
    )
    assert mcp_server.list_categories() == {"databases": 5, "llm": 3}


def test_list_categories_bad_category_raises():
    with pytest.raises(ToolError):
        mcp_server.list_categories(category="(")


def test_list_categories_only_top_with_category_raises():
    with pytest.raises(ToolError):
        mcp_server.list_categories(category="databases", only_top=True)
