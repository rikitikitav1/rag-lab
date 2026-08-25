from types import SimpleNamespace

import agent_tools
import errors
import llm
import mcp_client
import outcomes
import pytest
from models.registry import Purpose
from use_cases import agent, chat

_REAL_DISPATCH = agent_tools.dispatch


def _turn(text=None, tool_calls=(), message=None):
    return llm.ChatTurn(
        text=text,
        tool_calls=list(tool_calls),
        message=message,
        prompt_tokens=0,
        completion_tokens=0,
    )


def _tool_call(call_id, name, arguments):
    return SimpleNamespace(
        id=call_id, function=SimpleNamespace(name=name, arguments=arguments)
    )


def test_apply_turn_final_answer_stops_loop():
    result = agent.AgentResult()
    messages = []
    done = agent._apply_turn(_turn(text="the answer"), messages, result)
    assert done is True
    assert result.success is True
    assert result.text == "the answer"
    assert messages == []


def test_apply_turn_coerces_none_text():
    result = agent.AgentResult()
    assert agent._apply_turn(_turn(text=None), [], result) is True
    assert result.text == ""
    assert result.success is False


def test_apply_turn_executes_tools_and_continues(monkeypatch):
    monkeypatch.setattr(
        agent_tools,
        "dispatch",
        lambda name, args, **kwargs: agent_tools.ToolResult(
            content="chunks", meta={"sources": ["S1"]}
        ),
    )
    tc = _tool_call("call_1", "search_corpus", '{"query": "x"}')
    turn = _turn(tool_calls=[tc], message={"role": "assistant"})
    messages = [{"role": "user", "content": "x"}]
    result = agent.AgentResult()

    done = agent._apply_turn(turn, messages, result)

    assert done is False
    assert result.success is False
    assert result.sources == ["S1"]
    assert messages[1] == {"role": "assistant"}
    assert messages[2] == {
        "role": "tool",
        "tool_call_id": "call_1",
        "content": "chunks",
    }


def test_context_from_messages_joins_only_tool_contents():
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "q"},
        SimpleNamespace(role="assistant", content=None),
        {"role": "tool", "tool_call_id": "a", "content": "chunk A"},
        {"role": "tool", "tool_call_id": "b", "content": "chunk B"},
    ]
    assert agent._context_from_messages(messages) == "chunk A\n\nchunk B"


def test_context_from_messages_excludes_sentinel_and_errors():
    from use_cases import chat

    messages = [
        {"role": "tool", "tool_call_id": "a", "content": "chunk A"},
        {"role": "tool", "tool_call_id": "b", "content": chat.NO_RESULTS},
        {"role": "tool", "tool_call_id": "c", "content": f"{errors.ERROR_PREFIX}boom"},
        {"role": "tool", "tool_call_id": "d", "content": "chunk B"},
    ]
    assert agent._context_from_messages(messages) == "chunk A\n\nchunk B"


def test_unique_sources_dedups_by_source():
    a1 = SimpleNamespace(source="a.md")
    a2 = SimpleNamespace(source="a.md")
    b = SimpleNamespace(source="b.md")
    out = agent._unique_sources([a1, a2, b])
    assert [s.source for s in out] == ["a.md", "b.md"]


def test_apply_turn_accumulates_across_multiple_calls(monkeypatch):
    monkeypatch.setattr(
        agent_tools,
        "dispatch",
        lambda name, args, **kwargs: agent_tools.ToolResult(content="c", meta={"sources": [name]}),
    )
    turn = _turn(
        tool_calls=[
            _tool_call("a", "search_corpus", "{}"),
            _tool_call("b", "search_corpus", "{}"),
        ],
        message={"role": "assistant"},
    )
    messages = []
    result = agent.AgentResult()

    agent._apply_turn(turn, messages, result)

    assert result.sources == ["search_corpus", "search_corpus"]
    assert [m.get("tool_call_id") for m in messages if m.get("role") == "tool"] == [
        "a",
        "b",
    ]


def _remote_tool(name="deepwiki__ask_question"):
    return agent_tools.Tool(
        name=name,
        description="remote",
        parameters={
            "type": "object",
            "required": ["q"],
            "properties": {"q": {"type": "string", "description": "the question"}},
        },
        run=lambda **kw: agent_tools.ToolResult(content="remote answer"),
    )


def _agent_harness(monkeypatch, turns, corpus_sources, seen_runtime=None):
    seen_tools, seen_extra = [], []

    def fake_chat(messages, tools=None, role=None, model=None):
        seen_tools.append([t["function"]["name"] for t in (tools or [])])
        return turns.pop(0)

    def fake_dispatch(name, arguments, extra=None, **runtime):
        seen_extra.append(extra)
        if seen_runtime is not None:
            seen_runtime.append(runtime)
        sources = (
            corpus_sources
            if name == agent_tools.CORPUS_TOOL
            else [SimpleNamespace(source="remote")]
        )
        return agent_tools.ToolResult(content="c", meta={"sources": sources})

    monkeypatch.setattr(agent.llm, "chat", fake_chat)
    monkeypatch.setattr(agent_tools, "dispatch", fake_dispatch)
    monkeypatch.setattr(agent_tools, "remote_tools", lambda: [_remote_tool()])
    monkeypatch.setattr(
        agent.llm, "ask", lambda system, user, **kw: SimpleNamespace(text="yes")
    )
    monkeypatch.setattr(agent.prompt_repo, "active_template", lambda purpose: f"tpl:{purpose}")
    monkeypatch.setattr(agent, "_log_answer", lambda *a, **kw: None)
    return seen_tools, seen_extra


def test_corpus_first_hides_remote_tools_until_search_comes_back_empty(monkeypatch):
    turns = [
        _turn(tool_calls=[_tool_call("a", "search_corpus", "{}")], message={"role": "assistant"}),
        _turn(text="final"),
    ]
    seen_tools, seen_extra = _agent_harness(monkeypatch, turns, corpus_sources=[])

    result = agent.run("q", max_hops=2)

    assert seen_tools == [["search_corpus"], ["search_corpus", "deepwiki__ask_question"]]
    assert seen_extra == [None]
    assert result.fallback_reason == agent.FallbackReason.empty


def test_corpus_first_keeps_remote_tools_hidden_while_corpus_answers(monkeypatch):
    turns = [
        _turn(tool_calls=[_tool_call("a", "search_corpus", "{}")], message={"role": "assistant"}),
        _turn(text="final"),
    ]
    seen_tools, _ = _agent_harness(monkeypatch, turns, corpus_sources=[SimpleNamespace(source="S1")])

    result = agent.run("q", max_hops=2)

    assert seen_tools == [["search_corpus"], ["search_corpus"]]
    assert result.fallback_reason == agent.FallbackReason.none


def test_the_notice_rides_in_the_tool_result_not_a_system_message(monkeypatch):
    turns = [
        _turn(tool_calls=[_tool_call("a", "search_corpus", "{}")], message={"role": "assistant"}),
        _turn(text="final"),
    ]
    _agent_harness(monkeypatch, turns, corpus_sources=[])

    result = agent.run("q", max_hops=2)

    tool_messages = [m for m in result.messages if m.get("role") == "tool"]
    assert len(tool_messages) == 1
    assert tool_messages[0]["content"].endswith(f"tpl:{Purpose.agent_fallback}")
    assert [m["role"] for m in result.messages].count("system") == 1
    assert result.fallback_announced is True


def test_no_notice_when_the_corpus_answered(monkeypatch):
    turns = [
        _turn(tool_calls=[_tool_call("a", "search_corpus", "{}")], message={"role": "assistant"}),
        _turn(text="final"),
    ]
    _agent_harness(monkeypatch, turns, corpus_sources=[SimpleNamespace(source="S1")])

    result = agent.run("q", max_hops=2)

    assert result.fallback_announced is False
    assert all(f"tpl:{Purpose.agent_fallback}" not in m.get("content", "") for m in result.messages)


def _scored(score, name="s.md"):
    return SimpleNamespace(source=name, rerank_score=score)


def test_verdict_reads_the_cross_encoder_not_the_hit_count():
    gate = agent.Gate(top=5, threshold=0.5)
    assert agent._verdict([], gate) == agent.FallbackReason.empty
    assert agent._verdict([_scored(0.02), _scored(0.4)], gate) == agent.FallbackReason.weak
    assert agent._verdict([_scored(0.02), _scored(0.91)], gate) is None
    assert agent._verdict([_scored(0.02)], agent.Gate()) is None


def test_weak_retrieval_opens_the_toolbox_and_drops_the_junk_context(monkeypatch):
    turns = [
        _turn(tool_calls=[_tool_call("a", "search_corpus", "{}")], message={"role": "assistant"}),
        _turn(text="final"),
    ]
    runtime = []
    seen_tools, _ = _agent_harness(
        monkeypatch, turns, corpus_sources=[_scored(0.02)], seen_runtime=runtime
    )

    result = agent.run("q", max_hops=2, fallback_policy="corpus_first_weak")

    assert result.fallback_reason == agent.FallbackReason.weak
    assert seen_tools[1] == ["search_corpus", "deepwiki__ask_question"]
    assert result.sources == []
    tool_content = [m["content"] for m in result.messages if m.get("role") == "tool"][0]
    assert tool_content.startswith(chat.NO_RESULTS)
    assert tool_content.endswith(f"tpl:{Purpose.agent_fallback}")
    assert runtime[0]["gate_top"] == 5


def test_strong_retrieval_keeps_the_context_and_the_gate_shut(monkeypatch):
    turns = [
        _turn(tool_calls=[_tool_call("a", "search_corpus", "{}")], message={"role": "assistant"}),
        _turn(text="final"),
    ]
    seen_tools, _ = _agent_harness(monkeypatch, turns, corpus_sources=[_scored(0.91)])

    result = agent.run("q", max_hops=2, fallback_policy="corpus_first_weak")

    assert result.fallback_reason == agent.FallbackReason.none
    assert seen_tools == [["search_corpus"], ["search_corpus"]]
    assert [s.rerank_score for s in result.sources] == [0.91]


def test_empty_rule_policy_does_not_score_the_gate(monkeypatch):
    turns = [
        _turn(tool_calls=[_tool_call("a", "search_corpus", "{}")], message={"role": "assistant"}),
        _turn(text="final"),
    ]
    runtime = []
    _agent_harness(
        monkeypatch, turns, corpus_sources=[_scored(0.02)], seen_runtime=runtime
    )

    result = agent.run("q", max_hops=2, fallback_policy="corpus_first")

    assert runtime[0].get("gate_top") is None
    assert result.fallback_reason == agent.FallbackReason.none


def test_agent_choice_exposes_remote_tools_from_the_first_hop(monkeypatch):
    turns = [_turn(text="final")]
    seen_tools, _ = _agent_harness(monkeypatch, turns, corpus_sources=[SimpleNamespace(source="S1")])

    agent.run("q", max_hops=2, fallback_policy="agent_choice")

    assert seen_tools == [["search_corpus", "deepwiki__ask_question"]]


def test_apply_turn_records_the_tool_error_kind(monkeypatch):
    monkeypatch.setattr(
        agent_tools,
        "dispatch",
        lambda name, args, **kw: agent_tools.ToolResult(
            content=f"{errors.ERROR_PREFIX}tool 'ask' failed (auth): HTTPStatusError",
            meta={"error_kind": "auth"},
        ),
    )
    result = agent.AgentResult()
    turn = _turn(
        tool_calls=[_tool_call("a", "deepwiki__ask_question", "{}")],
        message={"role": "assistant"},
    )

    agent._apply_turn(turn, [], result)

    assert result.tool_errors == {"deepwiki__ask_question": "auth"}
    assert result.sources == []
    assert result.fallback_reason == agent.FallbackReason.none


def test_dispatch_routes_extra_tools():
    calls = []
    tool = agent_tools.Tool(
        name="deepwiki__ask",
        description="d",
        parameters={"type": "object", "properties": {"q": {"type": "string"}}},
        run=lambda **kw: agent_tools.ToolResult(content=str(calls.append(kw) or "ok")),
    )
    res = agent_tools.dispatch("deepwiki__ask", '{"q": "hi"}', extra={tool.name: tool})
    assert res.content == "ok"
    assert calls == [{"q": "hi"}]


def test_dispatch_passthrough_without_properties():
    seen = {}
    tool = agent_tools.Tool(
        name="x__raw",
        description="d",
        parameters={},
        run=lambda **kw: agent_tools.ToolResult(content=str(seen.update(kw) or "ok")),
    )
    agent_tools.dispatch("x__raw", '{"a": 1, "b": 2}', extra={tool.name: tool})
    assert seen == {"a": 1, "b": 2}


def test_remote_run_success_adds_source_marker(monkeypatch):
    async def fake_call(integration, tool, args):
        return mcp_client.CallOutcome(text="answer text")

    monkeypatch.setattr(agent_tools.mcp_client, "call_tool", fake_call)
    integration = SimpleNamespace(name="deepwiki")
    run = agent_tools._remote_run(integration, "ask_question")
    res = run(question="q")
    assert res.content == "answer text"
    assert [s.source for s in res.meta["sources"]] == ["mcp:deepwiki__ask_question"]


def test_remote_run_error_carries_the_kind_and_no_sources(monkeypatch):
    async def fake_call(integration, tool, args):
        return mcp_client.CallOutcome(
            text=f"{errors.ERROR_PREFIX}tool 'ask' failed (timeout): ReadTimeout",
            error_kind="timeout",
        )

    monkeypatch.setattr(agent_tools.mcp_client, "call_tool", fake_call)
    run = agent_tools._remote_run(SimpleNamespace(name="deepwiki"), "ask")
    res = run(question="q")
    assert res.content.startswith(errors.ERROR_PREFIX)
    assert res.meta == {"error_kind": "timeout"}


def test_remote_run_closures_do_not_share_tool_name(monkeypatch):
    async def fake_call(integration, tool, args):
        return mcp_client.CallOutcome(text=f"called:{tool}")

    monkeypatch.setattr(agent_tools.mcp_client, "call_tool", fake_call)
    integration = SimpleNamespace(name="srv")
    runs = [agent_tools._remote_run(integration, name) for name in ("first", "second")]
    assert runs[0]().content == "called:first"
    assert runs[1]().content == "called:second"


def test_dispatch_tells_the_model_which_arguments_it_missed():
    tool = agent_tools.Tool(
        name="deepwiki__ask_question",
        description="d",
        parameters={
            "type": "object",
            "required": ["repoName", "question"],
            "properties": {"repoName": {"type": "string"}, "question": {"type": "string"}},
        },
        run=lambda **kw: agent_tools.ToolResult(content="never called"),
    )
    res = agent_tools.dispatch(
        "deepwiki__ask_question", '{"query": "pool limits in httpx"}', extra={tool.name: tool}
    )
    assert res.content.startswith(errors.ERROR_PREFIX)
    assert "repoName" in res.content and "question" in res.content
    assert res.meta == {"error_kind": "client"}


def test_a_narrated_tool_call_gets_one_nudge(monkeypatch):
    turns = [
        _turn(text='I will use deepwiki__ask_question(repoName="x")', message={"role": "assistant"}),
        _turn(tool_calls=[_tool_call("a", "deepwiki__ask_question", "{}")], message={"role": "assistant"}),
        _turn(text="real answer"),
    ]
    _agent_harness(monkeypatch, turns, corpus_sources=[SimpleNamespace(source="S1")])

    result = agent.run("q", max_hops=4, fallback_policy="agent_choice")

    assert agent.TOOL_CALL_NUDGE in [m.get("content") for m in result.messages]
    assert result.text == "real answer"


def test_plain_final_answer_is_not_nudged(monkeypatch):
    turns = [_turn(text="the corpus says hello")]
    _agent_harness(monkeypatch, turns, corpus_sources=[SimpleNamespace(source="S1")])

    result = agent.run("q", max_hops=4, fallback_policy="agent_choice")

    assert agent.TOOL_CALL_NUDGE not in [m.get("content") for m in result.messages]
    assert result.text == "the corpus says hello"


def test_tools_that_cannot_answer_are_never_offered(monkeypatch):
    turns = [
        _turn(tool_calls=[_tool_call("a", "search_corpus", "{}")], message={"role": "assistant"}),
        _turn(text="I cannot answer this from the available sources"),
    ]
    seen_tools, _ = _agent_harness(monkeypatch, turns, corpus_sources=[_scored(0.02)])
    monkeypatch.setattr(
        agent.llm, "ask", lambda system, user, **kw: SimpleNamespace(text="no")
    )

    result = agent.run("how do I cook carbonara", max_hops=2, fallback_policy="corpus_first_weak")

    assert seen_tools == [["search_corpus"], ["search_corpus"]]
    assert [s.rerank_score for s in result.sources] == [0.02]
    assert result.fallback_announced is False


def test_a_matching_tool_is_offered(monkeypatch):
    turns = [
        _turn(tool_calls=[_tool_call("a", "search_corpus", "{}")], message={"role": "assistant"}),
        _turn(text="final"),
    ]
    seen_tools, _ = _agent_harness(monkeypatch, turns, corpus_sources=[_scored(0.02)])
    monkeypatch.setattr(
        agent.llm, "ask", lambda system, user, **kw: SimpleNamespace(text="yes")
    )

    agent.run("in the repository x/y, what does z do", max_hops=2, fallback_policy="corpus_first_weak")

    assert seen_tools[1] == ["search_corpus", "deepwiki__ask_question"]


def test_agent_choice_skips_the_match_check(monkeypatch):
    turns = [_turn(text="final")]
    seen_tools, _ = _agent_harness(monkeypatch, turns, corpus_sources=[_scored(0.9)])
    monkeypatch.setattr(
        agent.llm, "ask", lambda *a, **kw: pytest.fail("baseline must stay untouched")
    )

    agent.run("q", max_hops=2, fallback_policy="agent_choice")

    assert seen_tools == [["search_corpus", "deepwiki__ask_question"]]


def test_a_tool_without_required_arguments_needs_no_check(monkeypatch):
    turns = [
        _turn(tool_calls=[_tool_call("a", "search_corpus", "{}")], message={"role": "assistant"}),
        _turn(text="final"),
    ]
    open_tool = agent_tools.Tool(
        name="srv__ping", description="d", parameters={"type": "object", "properties": {}},
        run=lambda **kw: agent_tools.ToolResult(content="pong"),
    )
    seen_tools, _ = _agent_harness(monkeypatch, turns, corpus_sources=[])
    monkeypatch.setattr(agent_tools, "remote_tools", lambda: [open_tool])
    monkeypatch.setattr(agent.llm, "ask", lambda *a, **kw: pytest.fail("nothing to check"))

    agent.run("q", max_hops=2, fallback_policy="corpus_first")

    assert seen_tools[1] == ["search_corpus", "srv__ping"]


def test_a_raw_tool_call_is_not_served_as_an_answer(monkeypatch):
    raw = '{"name": "yandex_search", "parameters": {"q": "carbonara"}}'
    turns = [_turn(text=raw), _turn(text=raw)]
    _agent_harness(monkeypatch, turns, corpus_sources=[])
    monkeypatch.setattr(agent_tools, "remote_tools", lambda: [])

    result = agent.run("как приготовить карбонару?", max_hops=1)

    assert result.text == chat.NO_RESULTS
    assert result.success is False


def test_outcome_separates_a_refusal_from_an_unsupported_answer(monkeypatch):
    cases = [
        ("I cannot answer this from the available sources", outcomes.Outcome.refused),
        ("Кликхаус это распределённая база данных", outcomes.Outcome.unsupported_answer),
        ('{"name": "deepwiki__ask_question", "parameters": {"q": "x"}}', outcomes.Outcome.narrated_call),
        ("assistant\n\nI will use deepwiki__ask_question(repoName=\"x\")", outcomes.Outcome.narrated_call),
        ("", outcomes.Outcome.exhausted),
    ]
    for text, expected in cases:
        # a narrated call burns the nudge, an empty answer burns the forced final turn
        turns = [_turn(text=text), _turn(text=text), _turn(text=text)]
        _agent_harness(monkeypatch, turns, corpus_sources=[])
        monkeypatch.setattr(agent_tools, "remote_tools", lambda: [_remote_tool()])

        result = agent.run("q", max_hops=2, fallback_policy="agent_choice")

        assert result.outcome == expected, text[:40]


def test_a_grounded_answer_is_answered(monkeypatch):
    turns = [
        _turn(tool_calls=[_tool_call("a", "search_corpus", "{}")], message={"role": "assistant"}),
        _turn(text="the corpus says hello"),
    ]
    _agent_harness(monkeypatch, turns, corpus_sources=[SimpleNamespace(source="S1")])

    result = agent.run("q", max_hops=2)

    assert result.outcome == outcomes.Outcome.answered


def test_a_failed_corpus_search_does_not_read_as_an_empty_corpus(monkeypatch):
    turns = [
        _turn(tool_calls=[_tool_call("a", "search_corpus", "{}")], message={"role": "assistant"}),
        _turn(text="final"),
    ]
    seen_tools, _ = _agent_harness(monkeypatch, turns, corpus_sources=[])
    monkeypatch.setattr(
        agent_tools,
        "dispatch",
        lambda name, args, **kw: agent_tools.ToolResult(
            content=f"{errors.ERROR_PREFIX}tool failed", meta={"error_kind": "tool"}
        ),
    )

    result = agent.run("q", max_hops=2)

    assert result.fallback_reason == agent.FallbackReason.none
    assert seen_tools == [["search_corpus"], ["search_corpus"]]


def test_a_refusal_counts_even_when_the_corpus_gave_chunks(monkeypatch):
    turns = [
        _turn(tool_calls=[_tool_call("a", "search_corpus", "{}")], message={"role": "assistant"}),
        _turn(text="I cannot answer this from the available sources"),
    ]
    _agent_harness(monkeypatch, turns, corpus_sources=[SimpleNamespace(source="junk.md")])

    result = agent.run("q", max_hops=2, fallback_policy="corpus_first")

    assert result.outcome == outcomes.Outcome.refused
    assert result.sources


def test_the_log_keeps_the_raw_text_of_a_narrated_call(monkeypatch):
    narration = 'assistant\n\n{"name": "deepwiki__ask_question", "parameters": {"q": "x"}}'
    logged = {}
    turns = [_turn(text=narration), _turn(text=narration)]
    _agent_harness(monkeypatch, turns, corpus_sources=[])
    monkeypatch.setattr(
        agent, "_log_answer", lambda question, result, *a, **kw: logged.update(
            text=result.text, outcome=result.outcome
        )
    )

    result = agent.run("q", max_hops=2, fallback_policy="agent_choice")

    assert logged["outcome"] == outcomes.Outcome.narrated_call
    assert logged["text"] == narration
    assert result.text == chat.NO_RESULTS


def test_the_toolbox_is_not_reported_open_when_there_is_nothing_to_open(monkeypatch):
    turns = [
        _turn(tool_calls=[_tool_call("a", "search_corpus", "{}")], message={"role": "assistant"}),
        _turn(text="I cannot answer this from the available sources"),
    ]
    _agent_harness(monkeypatch, turns, corpus_sources=[])
    monkeypatch.setattr(agent_tools, "remote_tools", lambda: [])

    result = agent.run("q", max_hops=2)

    assert result.fallback_reason == agent.FallbackReason.empty
    assert result.fallback_opened is False
    assert result.fallback_announced is False


def test_one_verdict_per_turn_even_with_two_searches(monkeypatch):
    turns = [
        _turn(
            tool_calls=[
                _tool_call("a", "search_corpus", '{"query": "redis"}'),
                _tool_call("b", "search_corpus", '{"query": "kafka"}'),
            ],
            message={"role": "assistant"},
        ),
        _turn(text="final"),
    ]
    hits = {"a": [SimpleNamespace(source="redis.md", rerank_score=0.9)], "b": []}
    _agent_harness(monkeypatch, turns, corpus_sources=[])
    monkeypatch.setattr(
        agent_tools,
        "dispatch",
        lambda name, args, **kw: agent_tools.ToolResult(
            content="c", meta={"sources": hits["a" if "redis" in args else "b"]}
        ),
    )

    result = agent.run("q", max_hops=2, fallback_policy="corpus_first")

    notices = [
        m for m in result.messages
        if m.get("role") == "tool" and f"tpl:{Purpose.agent_fallback}" in m["content"]
    ]
    assert notices == []
    assert result.fallback_reason == agent.FallbackReason.none


def test_broken_arguments_are_not_an_empty_corpus(monkeypatch):
    turns = [
        _turn(
            tool_calls=[_tool_call("a", "search_corpus", "{not json")],
            message={"role": "assistant"},
        ),
        _turn(text="final"),
    ]
    seen_tools, _ = _agent_harness(monkeypatch, turns, corpus_sources=[])
    monkeypatch.setattr(agent_tools, "dispatch", _REAL_DISPATCH)

    result = agent.run("q", max_hops=2)

    assert result.fallback_reason == agent.FallbackReason.none
    assert result.tool_errors == {"search_corpus": "client"}
    assert seen_tools == [["search_corpus"], ["search_corpus"]]


def test_the_nudge_also_covers_the_corpus_tool(monkeypatch):
    narration = 'I will search: search_corpus(query="redis persistence")'
    turns = [_turn(text=narration), _turn(text="real answer")]
    _agent_harness(monkeypatch, turns, corpus_sources=[])
    monkeypatch.setattr(agent_tools, "remote_tools", lambda: [])

    result = agent.run("q", max_hops=3)

    assert agent.TOOL_CALL_NUDGE in [m.get("content") for m in result.messages]
    assert result.text == "real answer"


def test_dropped_weak_chunks_still_count_as_retrieval(monkeypatch):
    logged = {}
    turns = [
        _turn(tool_calls=[_tool_call("a", "search_corpus", "{}")], message={"role": "assistant"}),
        _turn(text="final"),
    ]
    _agent_harness(monkeypatch, turns, corpus_sources=[_scored(0.02, "redis.md")])
    monkeypatch.setattr(
        agent, "_log_answer", lambda question, result, *a, **kw: logged.update(
            dropped=result.dropped_sources
        )
    )

    agent.run("q", max_hops=2, fallback_policy="corpus_first_weak")

    assert logged["dropped"] == ["redis.md"]


def test_a_run_with_no_evidence_is_asked_to_refuse_in_words(monkeypatch):
    turns = [
        _turn(tool_calls=[_tool_call("a", "search_corpus", "{}")], message={"role": "assistant"}),
        _turn(tool_calls=[_tool_call("b", "search_corpus", "{}")], message={"role": "assistant"}),
        _turn(text="I cannot answer this from the available sources"),
    ]
    _agent_harness(monkeypatch, turns, corpus_sources=[])
    monkeypatch.setattr(agent_tools, "remote_tools", lambda: [])

    result = agent.run("как приготовить карбонару?", max_hops=2)

    assert result.no_evidence_prompted is True
    assert result.messages[-1]["content"] == f"tpl:{Purpose.agent_no_evidence}"
    assert result.outcome == outcomes.Outcome.refused
    assert result.text == "I cannot answer this from the available sources"


def test_a_run_with_sources_is_not_asked_to_refuse(monkeypatch):
    turns = [
        _turn(tool_calls=[_tool_call("a", "search_corpus", "{}")], message={"role": "assistant"}),
        _turn(tool_calls=[_tool_call("b", "search_corpus", "{}")], message={"role": "assistant"}),
        _turn(text="the corpus says hello"),
    ]
    _agent_harness(monkeypatch, turns, corpus_sources=[SimpleNamespace(source="S1")])

    result = agent.run("q", max_hops=2)

    assert result.no_evidence_prompted is False
    assert result.outcome == outcomes.Outcome.answered


def _run_with_signal(monkeypatch, signal, source):
    turns = [
        _turn(tool_calls=[_tool_call("a", "search_corpus", "{}")], message={"role": "assistant"}),
        _turn(text="final"),
    ]
    runtime = []
    _agent_harness(monkeypatch, turns, corpus_sources=[source], seen_runtime=runtime)
    result = agent.run("q", max_hops=2, fallback_policy="corpus_first_weak", gate_signal=signal)
    return result, runtime


def _hit(rerank_score=None, vector_distance=None):
    return SimpleNamespace(source="s.md", rerank_score=rerank_score, vector_distance=vector_distance)


def test_distance_signal_flags_a_far_chunk_without_the_cross_encoder(monkeypatch):
    result, runtime = _run_with_signal(monkeypatch, "distance", _hit(vector_distance=0.48))
    assert result.fallback_reason == agent.FallbackReason.weak
    assert runtime[0].get("gate_top") is None


def test_distance_signal_lets_a_near_chunk_through(monkeypatch):
    result, _ = _run_with_signal(monkeypatch, "distance", _hit(vector_distance=0.20))
    assert result.fallback_reason == agent.FallbackReason.none


def test_either_flags_when_only_one_signal_fires(monkeypatch):
    far_but_relevant = _hit(rerank_score=0.9, vector_distance=0.48)
    result, _ = _run_with_signal(monkeypatch, "either", far_but_relevant)
    assert result.fallback_reason == agent.FallbackReason.weak

    close_but_irrelevant = _hit(rerank_score=0.02, vector_distance=0.20)
    result, _ = _run_with_signal(monkeypatch, "either", close_but_irrelevant)
    assert result.fallback_reason == agent.FallbackReason.weak


def test_cross_encoder_signal_ignores_the_distance(monkeypatch):
    result, _ = _run_with_signal(monkeypatch, "cross_encoder", _hit(rerank_score=0.9, vector_distance=0.48))
    assert result.fallback_reason == agent.FallbackReason.none


def _run_off_topic(monkeypatch, topic_score, corpus_sources, threshold=0.5):
    turns = [
        _turn(tool_calls=[_tool_call("a", "search_corpus", "{}")], message={"role": "assistant"}),
        _turn(text="I cannot answer this from the available sources"),
    ]
    seen_tools, _ = _agent_harness(monkeypatch, turns, corpus_sources=corpus_sources)
    monkeypatch.setattr(agent, "_topic_score", lambda question: topic_score)
    result = agent.run(
        "q", max_hops=2, fallback_policy="corpus_first_weak", topic_threshold=threshold
    )
    return result, seen_tools


def test_an_off_topic_question_never_sees_an_external_tool(monkeypatch):
    weak = SimpleNamespace(source="junk.md", rerank_score=0.02, vector_distance=0.52)
    result, seen_tools = _run_off_topic(monkeypatch, topic_score=0.78, corpus_sources=[weak])

    assert result.fallback_reason == agent.FallbackReason.off_topic
    assert seen_tools == [["search_corpus"], ["search_corpus"]]
    assert result.sources == []
    assert result.outcome == outcomes.Outcome.refused


def test_a_question_on_topic_still_reaches_the_toolbox(monkeypatch):
    weak = SimpleNamespace(source="junk.md", rerank_score=0.02, vector_distance=0.42)
    result, seen_tools = _run_off_topic(monkeypatch, topic_score=0.44, corpus_sources=[weak])

    assert result.fallback_reason == agent.FallbackReason.weak
    assert seen_tools[1] == ["search_corpus", "deepwiki__ask_question"]


def test_the_topic_axis_is_off_unless_a_threshold_is_given(monkeypatch):
    turns = [
        _turn(tool_calls=[_tool_call("a", "search_corpus", "{}")], message={"role": "assistant"}),
        _turn(text="final"),
    ]
    _agent_harness(monkeypatch, turns, corpus_sources=[_scored(0.9)])
    monkeypatch.setattr(
        agent, "_topic_score", lambda question: pytest.fail("topic must not be scored")
    )

    agent.run("q", max_hops=2, fallback_policy="corpus_first_weak")
