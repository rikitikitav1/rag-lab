import asyncio
from types import SimpleNamespace

import mcp_server
from fastmcp import Client
from models.registry import Pipeline
from sqlalchemy.exc import SQLAlchemyError
from use_cases import chat


def test_search_corpus_returns_content(monkeypatch):
    monkeypatch.setattr(mcp_server.chat, "search_chunks", lambda q, category=None: ("chunks", []))
    assert mcp_server.search_corpus("redis") == "chunks"


def test_search_corpus_forwards_category(monkeypatch):
    seen = {}

    def fake(q, category=None):
        seen["category"] = category
        return ("c", [])

    monkeypatch.setattr(mcp_server.chat, "search_chunks", fake)
    mcp_server.search_corpus("redis", category="databases")
    assert seen["category"] == "databases"


def test_search_corpus_error_is_generic(monkeypatch):
    def boom(q, category=None):
        raise RuntimeError("SELECT ... FROM data_chunks; embedding=[0.1] secret")

    monkeypatch.setattr(mcp_server.chat, "search_chunks", boom)
    out = mcp_server.search_corpus("x")
    assert out.startswith(chat.ERROR_PREFIX)
    assert "secret" not in out


def test_search_corpus_error_masked_through_client(monkeypatch):
    def boom(q, category=None):
        raise RuntimeError("SELECT secret_sql FROM data_chunks; embedding=[0.1]")

    monkeypatch.setattr(mcp_server.chat, "search_chunks", boom)

    async def go():
        async with Client(mcp_server.mcp) as c:
            return await c.call_tool("search_corpus", {"query": "x"})

    res = asyncio.run(go())
    text = res.content[0].text if res.content else ""
    assert "secret_sql" not in text


def test_answer_question_agent_pipeline(monkeypatch):
    seen = {}

    def fake_run(question, run_name=None, language=None):
        seen["arm"] = "agent"
        seen["run_name"] = run_name
        return SimpleNamespace(text="agent answer", sources=[])

    monkeypatch.setattr(mcp_server.agent, "run", fake_run)
    assert mcp_server.answer_question("q") == {"answer": "agent answer", "sources": []}
    assert seen["arm"] == "agent"
    assert seen["run_name"] == "mcp"


def test_answer_question_single_shot_pipeline(monkeypatch):
    seen = {}

    def fake_answer(question, run_name=None, language=None):
        seen["arm"] = "single"
        return SimpleNamespace(text="single answer", sources=[])

    monkeypatch.setattr(mcp_server.chat, "answer", fake_answer)
    out = mcp_server.answer_question("q", pipeline=Pipeline.single_shot)
    assert out == {"answer": "single answer", "sources": []}
    assert seen["arm"] == "single"


def test_answer_question_returns_sources(monkeypatch):
    srcs = [SimpleNamespace(source="a.md"), SimpleNamespace(source="b.md")]
    monkeypatch.setattr(
        mcp_server.agent,
        "run",
        lambda question, run_name=None, language=None: SimpleNamespace(text="ans", sources=srcs),
    )
    assert mcp_server.answer_question("q") == {"answer": "ans", "sources": ["a.md", "b.md"]}


def test_answer_question_empty_text_fallback(monkeypatch):
    monkeypatch.setattr(
        mcp_server.agent,
        "run",
        lambda question, run_name=None, language=None: SimpleNamespace(text="", sources=[]),
    )
    assert mcp_server.answer_question("q") == {"answer": "No answer generated.", "sources": []}


def test_answer_question_error_is_generic(monkeypatch):
    def boom(question, run_name=None, language=None):
        raise RuntimeError("upstream 500: secret detail")

    monkeypatch.setattr(mcp_server.agent, "run", boom)
    out = mcp_server.answer_question("q")
    assert out["answer"].startswith(chat.ERROR_PREFIX)
    assert "secret detail" not in out["answer"]
    assert out["sources"] == []


def test_list_categories_maps_rows_with_counts(monkeypatch):
    monkeypatch.setattr(
        mcp_server.db,
        "list_categories",
        lambda only_top, category: [("databases", 5), ("llm", 3)],
    )
    assert mcp_server.list_categories() == {"databases": 5, "llm": 3}


def test_list_categories_error_returns_empty(monkeypatch):
    def boom(only_top, category):
        raise SQLAlchemyError("bad lquery")

    monkeypatch.setattr(mcp_server.db, "list_categories", boom)
    assert mcp_server.list_categories(category="(") == {}
