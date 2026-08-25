import agent_tools as at


def test_schema_shape():
    tool = at.Tool(
        name="t", description="d", parameters={"type": "object"}, run=lambda: at.ToolResult("")
    )
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
    assert "invalid arguments" in at.dispatch("search_corpus", "{bad").content


def test_dispatch_catches_tool_exception(monkeypatch):
    def boom(**kwargs):
        raise ValueError("kaboom")

    monkeypatch.setitem(
        at._REGISTRY, "boom", at.Tool(name="boom", description="", parameters={}, run=boom)
    )
    result = at.dispatch("boom", "{}")
    assert result.content == "error: tool 'boom' failed"
    assert "kaboom" not in result.content


def test_dispatch_drops_undeclared_args(monkeypatch):
    seen = {}

    def echo(**kwargs):
        seen.update(kwargs)
        return at.ToolResult("ok")

    schema = {"type": "object", "properties": {"query": {"type": "string"}}}
    monkeypatch.setitem(
        at._REGISTRY, "echo", at.Tool(name="echo", description="", parameters=schema, run=echo)
    )
    at.dispatch("echo", '{"query": "hi", "category": "db"}')
    assert seen == {"query": "hi"}


def test_search_corpus_formats_content_and_sources(monkeypatch):
    from use_cases import chat

    rows = [("chunk one", "src/a.md"), ("chunk two", "src/b.md")]
    monkeypatch.setattr(chat, "_retrieve_rows", lambda *a, **k: (rows, None))
    monkeypatch.setattr(chat, "is_ignored_source", lambda s: False)
    monkeypatch.setattr(chat, "take_sources", lambda r, scores=None: ["S1", "S2"])

    result = at._search_corpus("q")
    assert "[src/a.md]\nchunk one" in result.content
    assert "[src/b.md]\nchunk two" in result.content
    assert result.meta["sources"] == ["S1", "S2"]


def test_search_corpus_empty(monkeypatch):
    from use_cases import chat

    monkeypatch.setattr(chat, "_retrieve_rows", lambda *a, **k: ([], None))
    result = at._search_corpus("q")
    assert result.content == "No relevant documents found."
    assert result.meta["sources"] == []


def test_the_corpus_tool_describes_the_corpus_from_config():
    import config

    description = at._REGISTRY[at.CORPUS_TOOL].description
    assert config.settings.corpus.description in description
