import json
from pathlib import Path
from types import SimpleNamespace

import agent_tools
import errors
import llm
import mcp_client
import outcomes
import pytest
from models.registry import Purpose
from use_cases import agent, agent_policy, chat

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
        # the real tool returns the join of its own chunks, or the harness cannot see them drift
        contexts = [f"[{s.source}]\nc" for s in sources]
        return agent_tools.ToolResult(
            content="\n\n".join(contexts),
            meta={"sources": sources, "contexts": contexts},
        )

    monkeypatch.setattr(agent.llm, "chat", fake_chat)
    monkeypatch.setattr(agent_tools, "dispatch", fake_dispatch)
    monkeypatch.setattr(agent_tools, "remote_tools", lambda: [_remote_tool()])
    monkeypatch.setattr(
        agent.llm, "ask", lambda system, user, **kw: SimpleNamespace(text="yes")
    )
    monkeypatch.setattr(agent.prompt_repo, "active_template", lambda purpose: f"tpl:{purpose}")
    monkeypatch.setattr(agent, "_log_answer", lambda *a, **kw: None)
    # otherwise the axis embeds against a live corpus and the suite depends on the stand
    monkeypatch.setattr(agent, "_topic_score", lambda question, variant: None)
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
    # the flag says it fired; without the text no reader can tell what the model was told
    assert result.announced_text.endswith(f"tpl:{Purpose.agent_fallback}")
    assert result.announced_text in tool_messages[0]["content"]


def test_no_notice_when_the_corpus_answered(monkeypatch):
    turns = [
        _turn(tool_calls=[_tool_call("a", "search_corpus", "{}")], message={"role": "assistant"}),
        _turn(text="final"),
    ]
    _agent_harness(monkeypatch, turns, corpus_sources=[SimpleNamespace(source="S1")])

    result = agent.run("q", max_hops=2)

    assert result.fallback_announced is False
    assert result.announced_text == ""
    assert all(f"tpl:{Purpose.agent_fallback}" not in m.get("content", "") for m in result.messages)


def _scored(score, name="s.md", distance=None):
    return SimpleNamespace(source=name, rerank_score=score, vector_distance=distance)


def _weak_hit(name="s.md"):
    return _scored(0.02, name, distance=0.52)


def _strong_hit(name="s.md"):
    return _scored(0.91, name, distance=0.20)


def test_verdict_reads_the_cross_encoder_not_the_hit_count():
    gate = agent.Gate(signal=agent.GateSignal.cross_encoder, top=5, threshold=0.5)
    assert agent_policy.verdict([], gate) == agent.FallbackReason.empty
    assert agent_policy.verdict([_scored(0.02), _scored(0.4)], gate) == agent.FallbackReason.weak
    assert agent_policy.verdict([_scored(0.02), _scored(0.91)], gate) is None
    assert agent_policy.verdict([_scored(0.02)], agent.Gate(signal=agent.GateSignal.cross_encoder)) is None


def test_weak_retrieval_opens_the_toolbox_and_drops_the_junk_context(monkeypatch):
    turns = [
        _turn(tool_calls=[_tool_call("a", "search_corpus", "{}")], message={"role": "assistant"}),
        _turn(text="final"),
    ]
    runtime = []
    seen_tools, _ = _agent_harness(
        monkeypatch, turns, corpus_sources=[_weak_hit()], seen_runtime=runtime
    )

    result = agent.run(
        "q", max_hops=2, fallback_policy="corpus_first_weak", gate_signal="cross_encoder"
    )

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
    seen_tools, _ = _agent_harness(monkeypatch, turns, corpus_sources=[_strong_hit()])

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

    assert agent_policy.TOOL_CALL_NUDGE in [m.get("content") for m in result.messages]
    assert result.text == "real answer"


def test_plain_final_answer_is_not_nudged(monkeypatch):
    turns = [_turn(text="the corpus says hello")]
    _agent_harness(monkeypatch, turns, corpus_sources=[SimpleNamespace(source="S1")])

    result = agent.run("q", max_hops=4, fallback_policy="agent_choice")

    assert agent_policy.TOOL_CALL_NUDGE not in [m.get("content") for m in result.messages]
    assert result.text == "the corpus says hello"


def test_tools_that_cannot_answer_are_never_offered(monkeypatch):
    turns = [
        _turn(tool_calls=[_tool_call("a", "search_corpus", "{}")], message={"role": "assistant"}),
        _turn(text="I cannot answer this from the available sources"),
    ]
    seen_tools, _ = _agent_harness(monkeypatch, turns, corpus_sources=[_weak_hit()])
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
    seen_tools, _ = _agent_harness(monkeypatch, turns, corpus_sources=[_weak_hit()])
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

    result = agent.run("how do I cook carbonara?", max_hops=1)

    assert result.text == chat.NO_RESULTS
    assert result.success is False


def test_outcome_separates_a_refusal_from_an_unsupported_answer(monkeypatch):
    cases = [
        ("I cannot answer this from the available sources", outcomes.Outcome.refused),
        ("ClickHouse is a distributed database", outcomes.Outcome.unsupported_answer),
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

    assert agent_policy.TOOL_CALL_NUDGE in [m.get("content") for m in result.messages]
    assert result.text == "real answer"


def test_dropped_weak_chunks_still_count_as_retrieval(monkeypatch):
    logged = {}
    turns = [
        _turn(tool_calls=[_tool_call("a", "search_corpus", "{}")], message={"role": "assistant"}),
        _turn(text="final"),
    ]
    _agent_harness(monkeypatch, turns, corpus_sources=[_weak_hit("redis.md")])
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

    result = agent.run("how do I cook carbonara?", max_hops=2)

    assert result.no_evidence_prompted is True
    # the prompt, then the turn it produced: the answering turn is kept now, so it comes last
    assert result.messages[-2]["content"] == f"tpl:{Purpose.agent_no_evidence}"
    assert result.messages[-1]["content"] == "I cannot answer this from the available sources"
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


def test_the_snapshot_keeps_what_the_gate_saw_after_the_drop(monkeypatch):
    logged = {}
    turns = [
        _turn(tool_calls=[_tool_call("a", "search_corpus", "{}")], message={"role": "assistant"}),
        _turn(text="final"),
    ]
    _agent_harness(monkeypatch, turns, corpus_sources=[_weak_hit("junk.md")])
    monkeypatch.setattr(
        agent, "_log_answer", lambda question, result, *a, **kw: logged.update(
            snapshot=agent._retrieval_snapshot(
                result.sources, result.dropped_hits, result.dropped_sources
            )
        )
    )

    agent.run("in the repository x/y, what does z do", max_hops=2, fallback_policy="corpus_first_weak")

    assert logged["snapshot"]["results_count"] == 0
    assert logged["snapshot"]["min_distance"] == 0.52
    assert logged["snapshot"]["dropped_sources"] == ["junk.md"]


def test_a_run_can_override_the_distance_threshold(monkeypatch):
    turns = [
        _turn(tool_calls=[_tool_call("a", "search_corpus", "{}")], message={"role": "assistant"}),
        _turn(text="final"),
    ]
    _agent_harness(monkeypatch, turns, corpus_sources=[_hit(vector_distance=0.40)])

    strict = agent.run("q", max_hops=2, fallback_policy="corpus_first_weak", weak_distance=0.30)

    turns.extend([
        _turn(tool_calls=[_tool_call("a", "search_corpus", "{}")], message={"role": "assistant"}),
        _turn(text="final"),
    ])
    loose = agent.run("q", max_hops=2, fallback_policy="corpus_first_weak", weak_distance=0.45)

    assert strict.fallback_reason == agent.FallbackReason.weak
    assert loose.fallback_reason == agent.FallbackReason.none


def test_the_default_signal_is_the_distance(monkeypatch):
    turns = [
        _turn(tool_calls=[_tool_call("a", "search_corpus", "{}")], message={"role": "assistant"}),
        _turn(text="final"),
    ]
    runtime = []
    far_but_well_scored = _hit(rerank_score=0.9, vector_distance=0.48)
    _agent_harness(monkeypatch, turns, corpus_sources=[far_but_well_scored], seen_runtime=runtime)

    result = agent.run("q", max_hops=2, fallback_policy="corpus_first_weak")

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
    monkeypatch.setattr(agent, "_topic_score", lambda question, variant: topic_score)
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


def test_a_zero_threshold_switches_the_topic_axis_off(monkeypatch):
    turns = [
        _turn(tool_calls=[_tool_call("a", "search_corpus", "{}")], message={"role": "assistant"}),
        _turn(text="final"),
    ]
    _agent_harness(monkeypatch, turns, corpus_sources=[_strong_hit()])
    monkeypatch.setattr(
        agent, "_topic_score", lambda question, variant: pytest.fail("topic must not be scored")
    )

    agent.run("q", max_hops=2, fallback_policy="corpus_first_weak", topic_threshold=0)


def test_the_configured_threshold_scores_the_topic_without_being_asked(monkeypatch):
    turns = [
        _turn(tool_calls=[_tool_call("a", "search_corpus", "{}")], message={"role": "assistant"}),
        _turn(text="final"),
    ]
    _agent_harness(monkeypatch, turns, corpus_sources=[_strong_hit()])
    seen = []
    monkeypatch.setattr(agent, "_topic_score", lambda question, variant: seen.append(question) or 0.1)

    agent.run("q", max_hops=2, fallback_policy="corpus_first_weak")

    assert seen == ["q"]


def test_an_off_topic_question_loses_its_context_even_when_the_gate_disagrees(monkeypatch):
    turns = [
        _turn(tool_calls=[_tool_call("a", "search_corpus", "{}")], message={"role": "assistant"}),
        _turn(text="final"),
    ]
    near_hit = _hit(rerank_score=0.9, vector_distance=0.20)
    _agent_harness(monkeypatch, turns, corpus_sources=[near_hit])
    monkeypatch.setattr(agent, "_topic_score", lambda question, variant: 0.61)

    result = agent.run(
        "who was the first president of France",
        max_hops=2,
        fallback_policy="corpus_first_weak",
        topic_threshold=0.5,
    )

    assert result.fallback_reason == agent.FallbackReason.off_topic
    assert result.sources == []
    assert result.dropped_sources == ["s.md"]


def test_stages_add_up_per_call():
    result = agent.AgentResult()
    import time as _time

    start = _time.perf_counter()
    result.took("model", start)
    result.took("model", start)
    result.took("topic", start)

    assert result.stages["model"]["calls"] == 2
    assert result.stages["topic"]["calls"] == 1
    assert result.stages["model"]["ms"] >= 0


def test_server_timings_are_taken_only_when_reported():
    result = agent.AgentResult()
    result.note_server_timings({"prompt_eval_duration": 31_282_000, "eval_duration": 49_192_000})
    result.note_server_timings({})

    assert result.stages["prefill"] == {"ms": 31, "calls": 1}
    assert result.stages["decode"] == {"ms": 49, "calls": 1}


def test_the_row_says_which_edge_ended_the_graph():
    # `final` meant three things, and `pools.outcome` re-derived one of them from `hops >= ceiling`
    from orchestrators import graph
    from use_cases.agent import AgentResult
    from use_cases.agent_policy import FinishedBy

    result = AgentResult()
    ctx = {"max_hops": 4, "result": result, "role": "generation", "model": None}
    answered = graph.final_node({"text": "an answer", "hops": 2, "sources": ["a"]},
                                {"configurable": {"run": ctx}})
    assert answered == {"finished_by": FinishedBy.answer}
    # the same edge, written where a report can read it without asking the graph again
    assert result.trace == [{"node": "final", "hop": 2, "finished_by": str(FinishedBy.answer)}]


def test_a_reader_trusts_the_recorded_edge_over_the_ceiling_it_would_guess():
    from evals.pools import _exhausted
    from use_cases.agent_policy import FinishedBy

    ceiling = {"max_hops": 4}
    # the row spent its hops and answered anyway: the old rule called that exhausted
    said_answer = {"hops": 4, "finished_by": str(FinishedBy.answer)}
    assert _exhausted(said_answer, ceiling) is False
    assert _exhausted({"hops": 4}, ceiling) is True

    said_out = {"hops": 4, "finished_by": str(FinishedBy.hops_exhausted)}
    assert _exhausted(said_out, ceiling) is True
    assert _exhausted({**said_out, "failed": True}, ceiling) is False


def test_the_idiomatic_arm_names_why_the_field_is_empty():
    # a gap without a reason reads as a fault a month later
    from use_cases.agent_policy import FinishedBy

    assert FinishedBy.unrecorded == "unrecorded"
    source = Path(__file__).resolve().parent.parent / "app" / "orchestrators" / "react.py"
    assert "FinishedBy.unrecorded" in source.read_text()


def test_the_transcript_keeps_the_turns_and_leaves_the_tool_results_alone():
    # `contexts` already holds every chunk; repeating them here would double the row
    from use_cases.agent import transcript_of

    chunk = "[src/a.md]\nthe whole of a retrieved chunk, a kilobyte of it"
    messages = [
        {"role": "system", "content": "you are"},
        {"role": "user", "content": "how do I log a request"},
        SimpleNamespace(
            role="assistant", content="",
            tool_calls=[SimpleNamespace(
                function=SimpleNamespace(name="search_corpus", arguments='{"query": "logging"}')
            )],
        ),
        {"role": "tool", "tool_call_id": "1", "content": chunk},
        {"role": "assistant", "content": "use a middleware"},
    ]
    out = transcript_of(messages)

    assert [t["role"] for t in out] == ["system", "user", "assistant", "tool", "assistant"]
    assert out[2]["tool_calls"] == [{"name": "search_corpus", "arguments": '{"query": "logging"}'}]
    assert "content" not in out[3], "a tool result belongs to `contexts`, not to the transcript"
    assert chunk not in json.dumps(out, ensure_ascii=False)
    assert out[4]["content"] == "use a middleware"


def test_the_row_carries_the_transcript_as_its_own_column():
    from models.eval import QuestionLog

    assert "transcript" in QuestionLog.__table__.columns
    source = Path(__file__).resolve().parent.parent / "app" / "use_cases" / "agent.py"
    assert "transcript=transcript_of(result.messages)" in source.read_text()


def test_the_row_says_which_pieces_came_from_which_call():
    # `contexts` is flat across hops and calls, and a replay cannot regroup a flat list
    from orchestrators import graph

    assert "spans" in graph.State.__annotations__
    source = Path(__file__).resolve().parent.parent / "app"
    for arm in ("orchestrators/graph.py", "orchestrators/react.py"):
        text = (source / arm).read_text()
        assert '"pieces": len(' in text, f"{arm} records no span"
        assert '"hop":' in text and '"tool_call_id":' in text
    assert '"spans": result.spans or None' in (source / "use_cases" / "agent.py").read_text()


def test_a_forced_final_is_not_always_the_ceiling():
    # `hops_exhausted` on an empty first hop was the same lie the field was added to end
    from orchestrators import graph
    from use_cases.agent_policy import FinishedBy

    ctx = {"max_hops": 4, "result": SimpleNamespace(took=lambda *a: None, note_prompt=lambda t: None),
           "role": "generation", "model": None,
           "chat": lambda *a, **kw: llm.ChatTurn(text="t", tool_calls=[], message=None,
                                                 prompt_tokens=0, completion_tokens=0)}
    early = graph.final_node({"text": "", "hops": 1, "sources": ["a"], "messages": []},
                             {"configurable": {"run": ctx}})
    assert early["finished_by"] == FinishedBy.no_answer

    late = graph.final_node({"text": "", "hops": 4, "sources": ["a"], "messages": []},
                            {"configurable": {"run": ctx}})
    assert late["finished_by"] == FinishedBy.hops_exhausted


def test_the_tools_are_said_again_after_a_tool_answer_only_when_asked(monkeypatch):
    # llama3.1 renders tool schemas in the last user message, and a tool answer buries them
    from orchestrators import graph

    state = {"external": False, "pending": [], "hops": 1}
    ctx = graph.context(remote={}, gate=None, external=False)
    assert graph._restated(ctx, state) == [], "it must not fire unless the run asked"

    asked = graph.context(remote={}, gate=None, external=False, restate_tools=True)
    said = graph._restated(asked, state)
    assert len(said) == 1 and said[0]["role"] == "user"
    assert "search_corpus" in said[0]["content"]


def test_an_answer_whose_gate_scored_with_the_cross_encoder_names_the_reranker(monkeypatch):
    # the gate can call the reranker without `use_rerank`
    from conftest import FakeSession

    session, snapped = FakeSession(), []
    monkeypatch.setattr(agent, "Session", lambda: session)
    monkeypatch.setattr(agent.chat, "_find_or_create_question",
                        lambda s, text, lang: SimpleNamespace(id=1, original_text=text,
                                                              reference_answer=None))
    monkeypatch.setattr(agent.chat, "resolve_rerank", lambda asked: bool(asked))
    monkeypatch.setattr(agent.prompt_repo, "active_versions", lambda purposes: {})
    monkeypatch.setattr(agent.llm, "resolve_name", lambda role: f"{role}-model")
    monkeypatch.setattr(agent.run_snapshot, "of_run", lambda **kw: snapped.append(kw) or {})

    for signal, named in (("cross_encoder", True), ("distance", False)):
        gate = agent_policy.Gate(signal=signal)
        agent._log_answer("q", agent.AgentResult(), "run", use_rerank=False,
                          fallback_policy="corpus_first_weak", gate=gate, variant="baseline")
        assert ("reranking" in session.added[-1].models) is named, signal
        assert snapped[-1]["cross_encoder_used"] is named, signal
    # no reranker seated: the row is still written
    def unseated(role):
        if role == "reranking":
            raise agent.engines.Unnamed("no model assigned to role reranking")
        return f"{role}-model"

    monkeypatch.setattr(agent.llm, "resolve_name", unseated)
    agent._log_answer("q", agent.AgentResult(), "run", use_rerank=False,
                      fallback_policy="corpus_first_weak", gate=agent_policy.Gate(signal="either"),
                      variant="baseline")
    assert session.added[-1].models["reranking"] is None, "the row is written, the name is unknown"
    monkeypatch.setattr(agent.llm, "resolve_name", lambda role: f"{role}-model")
    monkeypatch.setattr(agent.config.settings.agent.gate, "signal", "cross_encoder")
    agent._log_answer("q", agent.AgentResult(), "run", use_rerank=False,
                      fallback_policy="corpus_first_weak", gate=None, variant="baseline")
    assert "reranking" not in session.added[-1].models, "the idiomatic arm runs no gate"
