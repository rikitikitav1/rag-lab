import json
from types import SimpleNamespace

import pytest
from test_agent import _agent_harness, _hit, _tool_call, _turn, _weak_hit
from use_cases import agent, agent_policy


def _scenario(monkeypatch, turns, corpus_sources, **kwargs):
    seen_tools, _ = _agent_harness(monkeypatch, list(turns), corpus_sources)
    kwargs.setdefault("max_hops", 2)
    result = agent.run("q", **kwargs)
    return result, seen_tools


def _both(monkeypatch_factory, turns, corpus_sources, **kwargs):
    out = []
    for orchestrator in (None, agent_policy.Orchestrator.langgraph_ported):
        with monkeypatch_factory() as monkeypatch:
            result, seen_tools = _scenario(
                monkeypatch, turns, corpus_sources, orchestrator=orchestrator, **kwargs
            )
            out.append((result, seen_tools))
    return out


def _idiomatic_run(monkeypatch, script, corpus_sources, **kwargs):
    from fake_chat import ScriptedChatModel
    from orchestrators import react

    _agent_harness(monkeypatch, [], corpus_sources)
    monkeypatch.setattr(agent, "_topic_score", lambda question: None)
    model = ScriptedChatModel(script)
    monkeypatch.setattr(react, "chat_model", lambda role=None, model_name=None: model)
    kwargs.setdefault("max_hops", 2)
    return agent.run("q", orchestrator=agent_policy.Orchestrator.langgraph_idiomatic, **kwargs)


# the middleware arm talks to a chat model, not to our client, so the same script is replayed
# through a scripted model instead of a patched llm.chat
def _middleware_run(monkeypatch, script, corpus_sources, dispatch=None, topic_score=None, **kwargs):
    from fake_chat import ScriptedChatModel
    from orchestrators import react

    _agent_harness(monkeypatch, [], corpus_sources)
    monkeypatch.setattr(agent, "_topic_score", lambda question: topic_score)
    if dispatch is not None:
        import agent_tools

        monkeypatch.setattr(agent_tools, "dispatch", dispatch)
    model = ScriptedChatModel(script)
    monkeypatch.setattr(react, "chat_model", lambda role=None, model_name=None: model)
    kwargs.setdefault("max_hops", 2)
    return agent.run(
        "q", orchestrator=agent_policy.Orchestrator.langgraph_middleware, **kwargs
    )


@pytest.fixture
def monkeypatch_factory():
    from _pytest.monkeypatch import MonkeyPatch

    class _Ctx:
        def __enter__(self):
            self.mp = MonkeyPatch()
            return self.mp

        def __exit__(self, *a):
            self.mp.undo()
            return False

    return _Ctx


# the create_agent arms keep their own message format, so equivalence is compared on everything
# the pipeline downstream actually reads
def _core(result):
    shape = _shape(result)
    for key in ("roles", "contents"):
        shape.pop(key, None)
    return shape


def turns_for(script):
    out = []
    for step in script:
        calls = [
            _tool_call(call.get("id", f"call_{i}"), call["name"], json.dumps(call.get("args", {})))
            for i, call in enumerate(step.get("tool_calls") or [])
        ]
        out.append(
            _turn(text=step.get("text"), tool_calls=calls, message={"role": "assistant"})
            if calls
            else _turn(text=step.get("text"), message={"role": "assistant", "content": step.get("text")})
        )
    return out


def _shape(result):
    return {
        "outcome": str(result.outcome),
        "text": result.text,
        "hops": result.hops,
        "sources": [s.source for s in result.sources],
        "dropped": sorted(result.dropped_sources),
        "fallback_reason": str(result.fallback_reason),
        "fallback_opened": result.fallback_opened,
        "fallback_announced": result.fallback_announced,
        "no_evidence_prompted": result.no_evidence_prompted,
        "tool_errors": result.tool_errors,
        "roles": [m.get("role") for m in result.messages],
        "contents": [m.get("content") for m in result.messages],
    }


def test_a_plain_corpus_answer_is_identical(monkeypatch_factory):
    turns = [
        _turn(tool_calls=[_tool_call("a", "search_corpus", "{}")], message={"role": "assistant"}),
        _turn(text="final"),
    ]
    (loop, loop_tools), (graph, graph_tools) = _both(
        monkeypatch_factory, turns, [_hit(rerank_score=0.9, vector_distance=0.2)]
    )
    assert _shape(loop) == _shape(graph)
    assert loop_tools == graph_tools


def test_weak_retrieval_opens_the_toolbox_the_same_way(monkeypatch_factory):
    turns = [
        _turn(tool_calls=[_tool_call("a", "search_corpus", "{}")], message={"role": "assistant"}),
        _turn(text="final"),
    ]
    (loop, loop_tools), (graph, graph_tools) = _both(
        monkeypatch_factory, turns, [_weak_hit()], fallback_policy="corpus_first_weak"
    )
    assert _shape(loop) == _shape(graph)
    assert loop_tools == graph_tools
    assert loop.fallback_reason == agent.FallbackReason.weak


def test_an_off_topic_question_is_refused_the_same_way(monkeypatch_factory):
    turns = [
        _turn(tool_calls=[_tool_call("a", "search_corpus", "{}")], message={"role": "assistant"}),
        _turn(text="I cannot answer this from the available sources"),
    ]
    results = []
    for orchestrator in (None, agent_policy.Orchestrator.langgraph_ported):
        with monkeypatch_factory() as monkeypatch:
            _agent_harness(monkeypatch, list(turns), [_hit(rerank_score=0.9, vector_distance=0.2)])
            monkeypatch.setattr(agent, "_topic_score", lambda question: 0.61)
            results.append(
                agent.run(
                    "how do I cook carbonara",
                    max_hops=2,
                    fallback_policy="corpus_first_weak",
                    topic_threshold=0.5,
                    orchestrator=orchestrator,
                )
            )
    assert _shape(results[0]) == _shape(results[1])
    assert results[0].fallback_reason == agent.FallbackReason.off_topic
    assert results[0].sources == []


def test_an_empty_corpus_ends_in_the_same_no_evidence_turn(monkeypatch_factory):
    turns = [
        _turn(tool_calls=[_tool_call("a", "search_corpus", "{}")], message={"role": "assistant"}),
        _turn(tool_calls=[_tool_call("b", "search_corpus", "{}")], message={"role": "assistant"}),
        _turn(text="nothing here covers it"),
    ]
    (loop, _), (graph, _) = _both(monkeypatch_factory, turns, [])
    assert _shape(loop) == _shape(graph)
    assert loop.no_evidence_prompted


def test_a_narrated_call_is_nudged_the_same_way(monkeypatch_factory):
    narration = '{"name": "deepwiki__ask_question", "parameters": {"q": "x"}}'
    turns = [
        _turn(text=narration, message={"role": "assistant", "content": narration}),
        _turn(tool_calls=[_tool_call("a", "search_corpus", "{}")], message={"role": "assistant"}),
        _turn(text="final"),
    ]
    (loop, _), (graph, _) = _both(
        monkeypatch_factory, turns, [_hit(rerank_score=0.9, vector_distance=0.2)], max_hops=3
    )
    assert _shape(loop) == _shape(graph)
    assert agent_policy.TOOL_CALL_NUDGE in [m.get("content") for m in loop.messages]


def test_a_failing_tool_is_reported_the_same_way(monkeypatch_factory):
    import agent_tools
    import errors

    turns = [
        _turn(tool_calls=[_tool_call("a", "search_corpus", "{}")], message={"role": "assistant"}),
        _turn(text="final"),
    ]
    results = []
    for orchestrator in (None, agent_policy.Orchestrator.langgraph_ported):
        with monkeypatch_factory() as monkeypatch:
            _agent_harness(monkeypatch, list(turns), [])
            monkeypatch.setattr(
                agent_tools,
                "dispatch",
                lambda name, args, **kw: agent_tools.ToolResult(
                    content=f"{errors.ERROR_PREFIX}tool failed", meta={"error_kind": "timeout"}
                ),
            )
            results.append(agent.run("q", max_hops=2, orchestrator=orchestrator))
    assert _shape(results[0]) == _shape(results[1])
    assert results[0].tool_errors == {"search_corpus": "timeout"}


def test_the_graph_reports_the_same_hop_count_when_it_runs_out(monkeypatch_factory):
    turns = [
        _turn(tool_calls=[_tool_call(str(i), "search_corpus", "{}")], message={"role": "assistant"})
        for i in range(4)
    ] + [_turn(text="late answer")]
    (loop, _), (graph, _) = _both(
        monkeypatch_factory, turns, [_hit(rerank_score=0.9, vector_distance=0.2)]
    )
    assert _shape(loop) == _shape(graph)
    assert loop.hops == graph.hops


def test_the_graph_leaves_a_context_for_the_judge(monkeypatch_factory):
    turns = [
        _turn(tool_calls=[_tool_call("a", "search_corpus", "{}")], message={"role": "assistant"}),
        _turn(text="final"),
    ]
    (loop, _), (graph, _) = _both(
        monkeypatch_factory, turns, [_hit(rerank_score=0.9, vector_distance=0.2)]
    )
    # a run whose context comes back empty is silently skipped by the judge
    assert agent._context_from_messages(graph.messages)
    assert agent._context_from_messages(graph.messages) == agent._context_from_messages(
        loop.messages
    )



def test_the_notice_is_not_repeated_once_the_toolbox_is_open(monkeypatch_factory):
    turns = [
        _turn(tool_calls=[_tool_call("a", "search_corpus", "{}")], message={"role": "assistant"}),
        _turn(tool_calls=[_tool_call("b", "search_corpus", "{}")], message={"role": "assistant"}),
        _turn(text="final"),
    ]
    (loop, _), (graph, _) = _both(
        monkeypatch_factory, turns, [_weak_hit()],
        fallback_policy="corpus_first_weak", max_hops=3,
    )
    assert _shape(loop) == _shape(graph)
    notices = [m for m in graph.messages if "tpl:agent.fallback" in str(m.get("content"))]
    assert len(notices) == 1


def test_a_nudge_on_the_last_hop_does_not_buy_another_hop(monkeypatch_factory):
    narration = '{"name": "deepwiki__ask_question", "parameters": {"q": "x"}}'
    turns = [
        _turn(tool_calls=[_tool_call("a", "search_corpus", "{}")], message={"role": "assistant"}),
        _turn(text=narration, message={"role": "assistant", "content": narration}),
        _turn(text="final after the nudge"),
    ]
    (loop, _), (graph, _) = _both(
        monkeypatch_factory, turns, [_hit(rerank_score=0.9, vector_distance=0.2)], max_hops=2
    )
    assert _shape(loop) == _shape(graph)


def test_the_middleware_arm_answers_from_the_corpus_like_the_loop(monkeypatch_factory):
    script = [{"tool_calls": [{"name": "search_corpus", "args": {"query": "q"}}]}, {"text": "final"}]
    with monkeypatch_factory() as monkeypatch:
        result = _middleware_run(
            monkeypatch, script, [_hit(rerank_score=0.9, vector_distance=0.2)]
        )
    (loop, _), _ = _both(
        monkeypatch_factory, turns_for(script), [_hit(rerank_score=0.9, vector_distance=0.2)]
    )
    assert _core(loop) == _core(result)


def test_the_middleware_arm_drops_weak_context_and_opens_the_toolbox(monkeypatch_factory):
    script = [{"tool_calls": [{"name": "search_corpus", "args": {"query": "q"}}]}, {"text": "final"}]
    with monkeypatch_factory() as monkeypatch:
        result = _middleware_run(
            monkeypatch, script, [_weak_hit()], fallback_policy="corpus_first_weak"
        )
    (loop, _), _ = _both(
        monkeypatch_factory, turns_for(script), [_weak_hit()], fallback_policy="corpus_first_weak"
    )
    assert _core(loop) == _core(result)
    assert result.dropped_sources == ["s.md"]


def test_the_middleware_arm_nudges_a_narrated_call_once(monkeypatch_factory):
    narration = '{"name": "deepwiki__ask_question", "parameters": {"q": "x"}}'
    script = [
        {"text": narration},
        {"tool_calls": [{"name": "search_corpus", "args": {"query": "q"}}]},
        {"text": "final"},
    ]
    with monkeypatch_factory() as monkeypatch:
        result = _middleware_run(
            monkeypatch, script, [_hit(rerank_score=0.9, vector_distance=0.2)], max_hops=3
        )
    assert result.text == "final"
    assert any(agent_policy.TOOL_CALL_NUDGE in str(m.get("content")) for m in result.messages)


def test_a_failing_corpus_tool_does_not_read_as_an_empty_corpus(monkeypatch_factory):
    import agent_tools
    import errors

    script = [{"tool_calls": [{"name": "search_corpus", "args": {"query": "q"}}]}, {"text": "final"}]
    def failing(name, args, **kw):
        return agent_tools.ToolResult(
            content=f"{errors.ERROR_PREFIX}tool failed", meta={"error_kind": "timeout"}
        )

    with monkeypatch_factory() as monkeypatch:
        result = _middleware_run(
            monkeypatch, script, [], dispatch=failing, fallback_policy="corpus_first_weak"
        )
    assert str(result.fallback_reason) == "none"
    assert result.fallback_opened is False



def _remote_hit():
    return SimpleNamespace(
        source="mcp:deepwiki__ask_question", rerank_score=None, vector_distance=None
    )


def test_an_external_answer_keeps_its_source(monkeypatch_factory):
    turns = [
        _turn(tool_calls=[_tool_call("a", "search_corpus", "{}")], message={"role": "assistant"}),
        _turn(
            tool_calls=[_tool_call("b", "deepwiki__ask_question", '{"q": "x"}')],
            message={"role": "assistant"},
        ),
        _turn(text="answered from the tool"),
    ]
    (loop, _), (graph, _) = _both(
        monkeypatch_factory, turns, [_weak_hit()],
        fallback_policy="corpus_first_weak", max_hops=3,
    )
    assert _shape(loop) == _shape(graph)
    assert [s.source for s in graph.sources] == ["remote"]
    assert str(graph.outcome) == "answered"


def test_the_middleware_arm_keeps_an_external_source_too(monkeypatch_factory):
    script = [
        {"tool_calls": [{"name": "search_corpus", "args": {"query": "q"}}]},
        {"tool_calls": [{"name": "deepwiki__ask_question", "args": {"q": "x"}}]},
        {"text": "answered from the tool"},
    ]

    def dispatch(name, args, **kw):
        import agent_tools

        if name == "search_corpus":
            return agent_tools.ToolResult(content="weak", meta={"sources": [_weak_hit()]})
        return agent_tools.ToolResult(content="remote answer", meta={"sources": [_remote_hit()]})

    with monkeypatch_factory() as monkeypatch:
        result = _middleware_run(
            monkeypatch, script, [_weak_hit()], dispatch=dispatch,
            fallback_policy="corpus_first_weak", max_hops=3,
        )
    assert [s.source for s in result.sources] == ["mcp:deepwiki__ask_question"]
    assert str(result.outcome) == "answered"
    assert result.fallback_opened is True


def test_a_model_that_only_calls_tools_stops_at_the_hop_cap(monkeypatch_factory):
    script = [{"tool_calls": [{"name": "search_corpus", "args": {"query": "q"}}]}] * 8
    with monkeypatch_factory() as monkeypatch:
        result = _middleware_run(
            monkeypatch, script, [_hit(rerank_score=0.9, vector_distance=0.2)], max_hops=3
        )
    # the cap lives in the middleware, so the guard must never be the thing that stops a run
    assert str(result.outcome) != "error"
    assert result.hops <= 4


def test_the_loop_and_the_graph_stop_at_the_same_hop_cap(monkeypatch_factory):
    turns = [
        _turn(tool_calls=[_tool_call(str(i), "search_corpus", "{}")], message={"role": "assistant"})
        for i in range(8)
    ]
    (loop, _), (graph, _) = _both(
        monkeypatch_factory, turns, [_hit(rerank_score=0.9, vector_distance=0.2)], max_hops=3
    )
    assert _shape(loop) == _shape(graph)
    assert loop.hops == 4


def test_the_middleware_arm_refuses_an_off_topic_question_like_the_loop(monkeypatch_factory):
    script = [
        {"tool_calls": [{"name": "search_corpus", "args": {"query": "q"}}]},
        {"text": "I cannot answer this from the available sources"},
    ]
    strong = _hit(rerank_score=0.9, vector_distance=0.2)
    with monkeypatch_factory() as monkeypatch:
        result = _middleware_run(
            monkeypatch,
            script,
            [strong],
            topic_score=0.61,
            fallback_policy="corpus_first_weak",
            topic_threshold=0.5,
        )
    # a strong retrieval must not rescue a question the axis has already ruled out
    assert str(result.fallback_reason) == "off_topic"
    assert result.sources == []
    assert result.dropped_sources == ["s.md"]


def test_the_bare_arm_answers_and_fills_the_same_result(monkeypatch_factory):
    script = [
        {"tool_calls": [{"name": "search_corpus", "args": {"query": "q"}}]},
        {"text": "the corpus says hello"},
    ]
    hit = _hit(rerank_score=0.9, vector_distance=0.2)
    with monkeypatch_factory() as monkeypatch:
        result = _idiomatic_run(monkeypatch, script, [hit])

    # every number the grid reads off this arm is assembled by hand in react.invoke
    assert result.text == "the corpus says hello"
    assert str(result.outcome) == "answered"
    assert [s.source for s in result.sources] == [hit.source]
    assert result.hops == 2
    assert result.success is True
    assert result.failed is False


def test_the_bare_arm_reports_no_evidence_when_the_corpus_is_empty(monkeypatch_factory):
    script = [
        {"tool_calls": [{"name": "search_corpus", "args": {"query": "q"}}]},
        {"text": "I cannot answer this from the available sources"},
    ]
    with monkeypatch_factory() as monkeypatch:
        result = _idiomatic_run(monkeypatch, script, [])

    assert str(result.outcome) == "refused"
    assert result.sources == []


def test_the_last_turn_prompt_reaches_the_transcript_not_only_the_model(monkeypatch_factory):
    script = [
        {"tool_calls": [{"name": "search_corpus", "args": {"query": "q"}}]},
        {"tool_calls": [{"name": "search_corpus", "args": {"query": "q"}}]},
        {"text": "nothing here covers it"},
    ]
    with monkeypatch_factory() as monkeypatch:
        result = _middleware_run(monkeypatch, script, [], max_hops=2)

    assert result.no_evidence_prompted is True
    contents = [str(m.get("content")) for m in result.messages]
    assert any("agent.no_evidence" in c for c in contents)


def test_a_remote_call_before_the_gate_opens_is_refused_by_dispatch(monkeypatch_factory):
    import agent_tools

    seen = []

    def spying_dispatch(name, arguments, extra=None, **runtime):
        seen.append((name, extra))
        return agent_tools.ToolResult(content="c", meta={"sources": []})

    script = [
        {"tool_calls": [{"name": "deepwiki__ask_question", "args": {"question": "q"}}]},
        {"text": "final"},
    ]
    with monkeypatch_factory() as monkeypatch:
        _middleware_run(monkeypatch, script, [], dispatch=spying_dispatch, max_hops=2)

    # the model can name a tool the gate has not opened, and dispatch is the thing that says no
    assert seen == [("deepwiki__ask_question", None)]


def _run_into_the_recursion_limit(monkeypatch_factory, orchestrator):
    from langgraph.errors import GraphRecursionError

    class Looping:
        def invoke(self, *args, **kwargs):
            raise GraphRecursionError("limit")

    with monkeypatch_factory() as monkeypatch:
        import langchain.agents as agents_module
        from orchestrators import react

        _agent_harness(monkeypatch, [], [])
        monkeypatch.setattr(react, "chat_model", lambda role=None, model=None: object())
        monkeypatch.setattr(agents_module, "create_agent", lambda **kw: Looping())
        return agent.run("q", max_hops=2, orchestrator=orchestrator)


def test_the_bare_arm_runs_out_of_hops_at_the_limit_rather_than_breaking(monkeypatch_factory):
    result = _run_into_the_recursion_limit(
        monkeypatch_factory, agent_policy.Orchestrator.langgraph_idiomatic
    )

    # the bare arm has no budget of its own, so the limit is what ends a run that keeps calling
    assert result.failed is False
    assert str(result.outcome) == "exhausted"


def test_the_recursion_guard_is_an_error_where_the_budget_lives_elsewhere(monkeypatch_factory):
    result = _run_into_the_recursion_limit(
        monkeypatch_factory, agent_policy.Orchestrator.langgraph_middleware
    )

    assert result.failed is True
    assert str(result.outcome) == "error"


def test_a_client_failure_is_logged_as_an_error_not_a_missing_row(monkeypatch_factory):
    class Broken:
        def invoke(self, *args, **kwargs):
            raise RuntimeError("connection reset")

    with monkeypatch_factory() as monkeypatch:
        import langchain.agents as agents_module
        from orchestrators import react

        _agent_harness(monkeypatch, [], [])
        monkeypatch.setattr(react, "chat_model", lambda role=None, model=None: object())
        monkeypatch.setattr(agents_module, "create_agent", lambda **kw: Broken())
        result = agent.run(
            "q", max_hops=2, orchestrator=agent_policy.Orchestrator.langgraph_idiomatic
        )

    # the loop writes a row for a hop that blew up, and the arms have to write one too
    assert result.failed is True
    assert str(result.outcome) == "error"


def test_an_empty_turn_gets_the_same_last_answer_the_loop_asks_for(monkeypatch_factory):
    script = [
        {"tool_calls": [{"name": "search_corpus", "args": {"query": "q"}}]},
        {"text": ""},
        {"text": "the corpus says hello"},
    ]
    with monkeypatch_factory() as monkeypatch:
        result = _middleware_run(
            monkeypatch, script, [_hit(rerank_score=0.9, vector_distance=0.2)], max_hops=4
        )

    assert result.text == "the corpus says hello"
    assert result.success is True


def test_a_tool_error_from_the_standard_node_is_not_read_as_an_empty_corpus(monkeypatch_factory):
    from langchain_core.messages import ToolMessage
    from orchestrators import middleware as mw

    failed = ToolMessage(content="Error: boom", tool_call_id="a", name="search_corpus")
    assert mw._corpus_results([failed]) == []
    ours = ToolMessage(content="error: timeout", tool_call_id="b", name="search_corpus")
    assert mw._corpus_results([ours]) == []
