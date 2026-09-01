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

    from db import Hit

    rows = [
        Hit("chunk one", "src/a.md", "cat", 0, 1, None, 0.1, 0.5, None),
        Hit("chunk two", "src/b.md", "cat", 0, 2, None, 0.2, 0.4, None),
    ]
    monkeypatch.setattr(chat, "_retrieve_rows", lambda *a, **k: (rows, None, 200))
    monkeypatch.setattr(chat, "take_sources", lambda r, scores=None, variant=None: ["S1", "S2"])

    result = at._search_corpus("q")
    assert "[src/a.md]\nchunk one" in result.content
    assert "[src/b.md]\nchunk two" in result.content
    assert result.meta["sources"] == ["S1", "S2"]


def test_search_corpus_empty(monkeypatch):
    from use_cases import chat

    monkeypatch.setattr(chat, "_retrieve_rows", lambda *a, **k: ([], None, 200))
    result = at._search_corpus("q")
    assert result.content == "No relevant documents found."
    assert result.meta["sources"] == []


def test_the_corpus_tool_describes_the_corpus_from_config():
    import config

    description = at._REGISTRY[at.CORPUS_TOOL].description
    assert config.settings.corpus.description in description


def test_the_corpus_tool_carries_the_depth_it_searched_at(monkeypatch):
    # resolving the depth again at logging time is a second answer, not the same one
    import agent_tools
    from use_cases import chat

    monkeypatch.setattr(
        chat, "search_chunks", lambda *a, **kw: ("content", ["chunk"], [], 137)
    )
    res = agent_tools._search_corpus("q", variant="baseline")
    assert res.meta["ef_search"] == 137


def test_the_topic_threshold_is_resolved_per_language():
    from config import AgentCfg

    cfg = AgentCfg(topic_threshold={"ru": 0.4962, "en": 0.4712})
    assert cfg.topic_threshold_for("ru") == 0.4962
    assert cfg.topic_threshold_for("en") == 0.4712


def test_an_unmeasured_language_gets_the_most_permissive_threshold():
    # we refuse only where refusing was shown not to cost a real question
    from config import AgentCfg

    cfg = AgentCfg(topic_threshold={"ru": 0.4962, "en": 0.4712})
    assert cfg.topic_threshold_for("de") == 0.4962
    assert cfg.topic_threshold_for(None) == 0.4962


def test_a_single_number_and_a_disabled_axis_still_work():
    from config import AgentCfg

    assert AgentCfg(topic_threshold=0.5).topic_threshold_for("en") == 0.5
    assert AgentCfg(topic_threshold=None).topic_threshold_for("en") is None
    assert AgentCfg(topic_threshold={}).topic_threshold_for("en") is None
