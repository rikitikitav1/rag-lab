import agent_tools as at
from agent_tools import Tool, ToolResult


def test_schema_shape():
    tool = Tool(name="t", description="d", parameters={"type": "object"}, run=lambda: ToolResult(""))
    s = tool.schema()
    assert s["type"] == "function"
    assert s["function"] == {
        "name": "t",
        "description": "d",
        "parameters": {"type": "object"},
    }


def test_dispatch_unknown_tool():
    assert "unknown tool" in at.dispatch("nope", "{}").content


def test_dispatch_bad_json():
    assert "invalid arguments json" in at.dispatch("search_corpus", "{bad").content


def test_dispatch_catches_tool_exception():
    def boom(**kwargs):
        raise ValueError("kaboom")

    at.register(Tool(name="boom", description="", parameters={}, run=boom))
    try:
        result = at.dispatch("boom", "{}")
    finally:
        at._REGISTRY.pop("boom", None)
    assert result.content == "error: tool 'boom' failed"
    assert "kaboom" not in result.content


def test_dispatch_passes_parsed_args():
    seen = {}

    def echo(**kwargs):
        seen.update(kwargs)
        return ToolResult("ok")

    at.register(Tool(name="echo", description="", parameters={}, run=echo))
    try:
        at.dispatch("echo", '{"query": "hi", "category": "db"}')
    finally:
        at._REGISTRY.pop("echo", None)
    assert seen == {"query": "hi", "category": "db"}


def test_search_corpus_formats_content_and_sources(monkeypatch):
    from use_cases import chat

    rows = [("chunk one", "src/a.md"), ("chunk two", "src/b.md")]
    monkeypatch.setattr(chat, "_retrieve_rows", lambda *a, **k: rows)
    monkeypatch.setattr(chat, "is_ignored_source", lambda s: False)
    monkeypatch.setattr(chat, "take_sources", lambda r: ["S1", "S2"])

    result = at._search_corpus("q")
    assert "[src/a.md]\nchunk one" in result.content
    assert "[src/b.md]\nchunk two" in result.content
    assert result.meta["sources"] == ["S1", "S2"]


def test_search_corpus_empty(monkeypatch):
    from use_cases import chat

    monkeypatch.setattr(chat, "_retrieve_rows", lambda *a, **k: [])
    result = at._search_corpus("q")
    assert result.content == "No relevant documents found."
    assert result.meta["sources"] == []
