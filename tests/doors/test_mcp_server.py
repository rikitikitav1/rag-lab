import asyncio
from types import SimpleNamespace

import mcp_server
import pytest
from fastmcp import Client
from fastmcp.exceptions import ToolError
from models.registry import Pipeline


@pytest.fixture(autouse=True)
def card_is_free(monkeypatch):
    monkeypatch.setattr(mcp_server.card_wait, "wait_for_the_card", lambda *roles: None)


def test_search_corpus_returns_content(monkeypatch):
    monkeypatch.setattr(
        mcp_server.chat,
        "search_chunks",
        lambda q, category=None, **kw: ("chunks", ["chunks"], [], 200, []),
    )
    assert mcp_server.search_corpus("redis") == "chunks"


def test_search_corpus_forwards_category(monkeypatch):
    seen = {}

    def fake(q, category=None, **kw):
        seen["category"] = category
        return ("c", ["c"], [], 200, [])

    monkeypatch.setattr(mcp_server.chat, "search_chunks", fake)
    mcp_server.search_corpus("redis", category="redis")
    assert seen["category"].label == "redis"


def test_the_old_dotted_category_path_is_refused_rather_than_finding_nothing():
    with pytest.raises(ToolError):
        mcp_server.search_corpus("redis", category="databases.redis")


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
    def boom(q, category=None, **kw):
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

    def fake_answer(text, scope=None, run_name=None, language=None):
        seen["category"] = scope.label
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
        lambda only_top, category, variant: [("none", "none", 3), ("redis", "databases", 5)],
    )
    assert mcp_server.list_categories() == {"none": 3, "redis": 5}


def test_list_categories_bad_category_raises():
    with pytest.raises(ToolError):
        mcp_server.list_categories(category="(")


def test_list_categories_only_top_with_category_raises():
    with pytest.raises(ToolError):
        mcp_server.list_categories(category="databases", only_top=True)


def test_the_mcp_tools_wait_for_the_card_like_the_rest_chat(monkeypatch):
    # only the REST doors had the guard, and an MCP question met ollama beside an awake judge
    asked = []

    def busy(*roles):
        asked.append(roles)
        raise mcp_server.card_wait.CardHeld("the card is held by the judge on vllm", 5)

    monkeypatch.setattr(mcp_server.card_wait, "wait_for_the_card", busy)
    with pytest.raises(ToolError, match="held by the judge"):
        mcp_server.search_corpus("redis")
    with pytest.raises(ToolError, match="held by the judge"):
        mcp_server.answer_question("what is redis")
    assert asked == [("embedding",), ("embedding", "generation")]


def test_a_version_needs_its_category_and_a_listed_version():
    with pytest.raises(ToolError, match="names no category"):
        mcp_server.search_corpus("x", version="17")
    with pytest.raises(ToolError, match="no versions declared"):
        mcp_server.search_corpus("x", category="redis", version="7")
    with pytest.raises(ToolError, match="has no version 9"):
        mcp_server.search_corpus("x", category="postgresql", version="9")
    with pytest.raises(ToolError, match="one category"):
        mcp_server.search_corpus("x", category="databases", version="17")


def test_without_a_version_a_versioned_category_reads_its_newest_and_a_rolling_source_reads_all():
    import db

    sql, params = db._scope_filter(db.Scope())
    assert "cardinality(versions) = 0" in sql
    assert "18" in params["newest_versions"] and "17" not in params["newest_versions"]
    sql, params = db._scope_filter(db.Scope(label="postgresql", version="17"))
    assert ":scope_version = ANY(versions) OR cardinality(versions) = 0" in sql and params["scope_version"] == "17"


# one bound on a scope's sources at every door: the MCP tools take no more than the chat and the queue do
def test_the_mcp_scope_takes_no_more_sources_than_the_other_doors(monkeypatch):
    from search_scope import MAX_SOURCES

    searched = []
    monkeypatch.setattr(mcp_server.db, "refuse_sources_out_of_search", lambda names: None)
    monkeypatch.setattr(mcp_server.chat, "search_chunks", lambda *a, **kw: searched.append(a) or [])

    async def go():
        async with Client(mcp_server.mcp) as c:
            return await c.call_tool("search_corpus", {"query": "x", "sources": ["s"] * (MAX_SOURCES + 1)})

    with pytest.raises(ToolError):
        asyncio.run(go())
    assert not searched
