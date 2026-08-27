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


def script_from_turns(turns) -> list:
    out = []
    for turn in turns:
        calls = [
            {
                "name": tc.function.name,
                "args": json.loads(tc.function.arguments or "{}"),
                "id": tc.id,
            }
            for tc in (turn.tool_calls or [])
        ]
        out.append({"text": turn.text, "tool_calls": calls} if calls else {"text": turn.text})
    return out


# the middleware arm is gone (owner, 27.08: the graph is the implementation), so what was
# an equivalence harness is now one arm. The assertions about the graph are the point and
# they stay; what disappeared is the second side of the comparison
def _graph_run(monkeypatch_factory, turns, corpus_sources, **kwargs):
    with monkeypatch_factory() as monkeypatch:
        return _scenario(
            monkeypatch, turns, corpus_sources,
            orchestrator=agent_policy.Orchestrator.langgraph_ported, **kwargs
        )


def _idiomatic_run(monkeypatch, script, corpus_sources, **kwargs):
    from fake_chat import ScriptedChatModel
    from orchestrators import react

    _agent_harness(monkeypatch, [], corpus_sources)
    monkeypatch.setattr(agent, "_topic_score", lambda question, variant: None)
    model = ScriptedChatModel(script)
    monkeypatch.setattr(react, "chat_model", lambda role=None, model_name=None: model)
    kwargs.setdefault("max_hops", 2)
    return agent.run("q", orchestrator=agent_policy.Orchestrator.langgraph_idiomatic, **kwargs)


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


# the create_agent arms keep their own message format, so transcripts are left out
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
    graph, graph_tools = _graph_run(
        monkeypatch_factory, turns, [_hit(rerank_score=0.9, vector_distance=0.2)]
    )
    # one toolbox per hop, and the corpus tool is the only thing offered while it holds
    assert graph_tools == [["search_corpus"], ["search_corpus"]]
    assert graph.fallback_reason == agent.FallbackReason.none


def test_weak_retrieval_opens_the_toolbox_the_same_way(monkeypatch_factory):
    turns = [
        _turn(tool_calls=[_tool_call("a", "search_corpus", "{}")], message={"role": "assistant"}),
        _turn(text="final"),
    ]
    graph, graph_tools = _graph_run(
        monkeypatch_factory, turns, [_weak_hit()], fallback_policy="corpus_first_weak"
    )
    assert graph.fallback_reason == agent.FallbackReason.weak


def test_an_off_topic_question_is_refused_the_same_way(monkeypatch_factory):
    turns = [
        _turn(tool_calls=[_tool_call("a", "search_corpus", "{}")], message={"role": "assistant"}),
        _turn(text="I cannot answer this from the available sources"),
    ]
    hit = _hit(rerank_score=0.9, vector_distance=0.2)
    with monkeypatch_factory() as monkeypatch:
        _agent_harness(monkeypatch, list(turns), [hit])
        monkeypatch.setattr(agent, "_topic_score", lambda question, variant: 0.61)
        graph = agent.run(
            "how do I cook carbonara",
            max_hops=2,
            fallback_policy="corpus_first_weak",
            topic_threshold=0.5,
            orchestrator=agent_policy.Orchestrator.langgraph_ported,
        )
    assert graph.fallback_reason == agent.FallbackReason.off_topic
    assert graph.sources == []


def test_an_empty_corpus_ends_in_the_same_no_evidence_turn(monkeypatch_factory):
    turns = [
        _turn(tool_calls=[_tool_call("a", "search_corpus", "{}")], message={"role": "assistant"}),
        _turn(tool_calls=[_tool_call("b", "search_corpus", "{}")], message={"role": "assistant"}),
        _turn(text="nothing here covers it"),
    ]
    graph, _ = _graph_run(monkeypatch_factory, turns, [])
    assert graph.no_evidence_prompted


def test_a_narrated_call_is_nudged_the_same_way(monkeypatch_factory):
    narration = '{"name": "deepwiki__ask_question", "parameters": {"q": "x"}}'
    turns = [
        _turn(text=narration, message={"role": "assistant", "content": narration}),
        _turn(tool_calls=[_tool_call("a", "search_corpus", "{}")], message={"role": "assistant"}),
        _turn(text="final"),
    ]
    graph, _ = _graph_run(
        monkeypatch_factory, turns, [_hit(rerank_score=0.9, vector_distance=0.2)], max_hops=3
    )
    assert agent_policy.TOOL_CALL_NUDGE in [m.get("content") for m in graph.messages]


def test_a_failing_tool_is_reported_the_same_way(monkeypatch_factory):
    import agent_tools
    import errors

    turns = [
        _turn(tool_calls=[_tool_call("a", "search_corpus", "{}")], message={"role": "assistant"}),
        _turn(text="final"),
    ]
    failing = lambda name, args, **kw: agent_tools.ToolResult(  # noqa: E731
        content=f"{errors.ERROR_PREFIX}tool failed", meta={"error_kind": "timeout"}
    )
    with monkeypatch_factory() as monkeypatch:
        _agent_harness(monkeypatch, list(turns), [])
        monkeypatch.setattr(agent_tools, "dispatch", failing)
        graph = agent.run(
            "q", max_hops=2, orchestrator=agent_policy.Orchestrator.langgraph_ported
        )
    assert graph.tool_errors == {"search_corpus": "timeout"}


def test_the_graph_reports_the_same_hop_count_when_it_runs_out(monkeypatch_factory):
    turns = [
        _turn(tool_calls=[_tool_call(str(i), "search_corpus", "{}")], message={"role": "assistant"})
        for i in range(4)
    ] + [_turn(text="late answer")]
    graph, _ = _graph_run(
        monkeypatch_factory, turns, [_hit(rerank_score=0.9, vector_distance=0.2)]
    )
    assert graph.hops == 3


def test_the_graph_leaves_a_context_for_the_judge(monkeypatch_factory):
    turns = [
        _turn(tool_calls=[_tool_call("a", "search_corpus", "{}")], message={"role": "assistant"}),
        _turn(text="final"),
    ]
    graph, _ = _graph_run(
        monkeypatch_factory, turns, [_hit(rerank_score=0.9, vector_distance=0.2)]
    )
    # a run whose context comes back empty is silently skipped by the judge
    assert agent._context_from_messages(graph.messages)


def test_the_notice_is_not_repeated_once_the_toolbox_is_open(monkeypatch_factory):
    turns = [
        _turn(tool_calls=[_tool_call("a", "search_corpus", "{}")], message={"role": "assistant"}),
        _turn(tool_calls=[_tool_call("b", "search_corpus", "{}")], message={"role": "assistant"}),
        _turn(text="final"),
    ]
    graph, _ = _graph_run(
        monkeypatch_factory, turns, [_weak_hit()],
        fallback_policy="corpus_first_weak", max_hops=3,
    )
    notices = [m for m in graph.messages if "tpl:agent.fallback" in str(m.get("content"))]
    assert len(notices) == 1


def test_a_nudge_on_the_last_hop_does_not_buy_another_hop(monkeypatch_factory):
    narration = '{"name": "deepwiki__ask_question", "parameters": {"q": "x"}}'
    turns = [
        _turn(tool_calls=[_tool_call("a", "search_corpus", "{}")], message={"role": "assistant"}),
        _turn(text=narration, message={"role": "assistant", "content": narration}),
        _turn(text="final after the nudge"),
    ]
    graph, _ = _graph_run(
        monkeypatch_factory, turns, [_hit(rerank_score=0.9, vector_distance=0.2)], max_hops=2
    )


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
    graph, _ = _graph_run(
        monkeypatch_factory, turns, [_weak_hit()],
        fallback_policy="corpus_first_weak", max_hops=3,
    )
    assert [s.source for s in graph.sources] == ["remote"]
    assert str(graph.outcome) == "answered"


def test_the_loop_and_the_graph_stop_at_the_same_hop_cap(monkeypatch_factory):
    turns = [
        _turn(tool_calls=[_tool_call(str(i), "search_corpus", "{}")], message={"role": "assistant"})
        for i in range(8)
    ]
    graph, _ = _graph_run(
        monkeypatch_factory, turns, [_hit(rerank_score=0.9, vector_distance=0.2)], max_hops=3
    )
    assert graph.hops == 4


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


# a cross-check misses a mistake both sides make, so the plain path is pinned to a literal
def test_the_plain_corpus_path_matches_a_written_down_shape(monkeypatch_factory):
    turns = [
        _turn(tool_calls=[_tool_call("a", "search_corpus", "{}")], message={"role": "assistant"}),
        _turn(text="final"),
    ]
    graph, offered = _graph_run(
        monkeypatch_factory, turns, [_hit(rerank_score=0.9, vector_distance=0.2)]
    )

    assert _core(graph) == {
        "outcome": "answered",
        "text": "final",
        "hops": 2,
        "sources": ["s.md"],
        "dropped": [],
        "fallback_reason": "none",
        "fallback_opened": False,
        "fallback_announced": False,
        "no_evidence_prompted": False,
        "tool_errors": {},
    }
    assert offered == [["search_corpus"], ["search_corpus"]]


# two corpus calls in one turn: one verdict, one notice on the last result (fix from CRAG-2)
def test_two_corpus_calls_in_one_turn_get_one_verdict_and_one_notice(monkeypatch_factory):
    turns = [
        _turn(
            tool_calls=[
                _tool_call("a", "search_corpus", '{"query": "x"}'),
                _tool_call("b", "search_corpus", '{"query": "y"}'),
            ],
            message={"role": "assistant"},
        ),
        _turn(text="final"),
    ]
    graph, _ = _graph_run(
        monkeypatch_factory, turns, [_weak_hit()], fallback_policy="corpus_first_weak"
    )

    assert str(graph.fallback_reason) == "weak"
    assert graph.fallback_opened is True
    notices = [m for m in graph.messages if "agent.fallback" in str(m.get("content"))]
    assert len(notices) == 1, "the notice rides on one tool result, not on every one of them"
    tool_ids = [m.get("tool_call_id") for m in graph.messages if m.get("role") == "tool"]
    assert tool_ids == ["a", "b"]


# the last hop's results must reach the verdict before anything decides there is no evidence
def _empty_then_hit(hit):
    import agent_tools

    calls = {"n": 0}

    def dispatch(name, arguments, extra=None, **runtime):
        calls["n"] += 1
        sources = [hit] if calls["n"] > 1 else []
        return agent_tools.ToolResult(content="c", meta={"sources": sources})

    return dispatch


def test_an_empty_first_call_on_the_last_hop_still_opens_the_toolbox(monkeypatch_factory):
    narration = '{"name": "search_corpus", "parameters": {"query": "x"}}'
    turns = [
        _turn(text=narration, message={"role": "assistant", "content": narration}),
        _turn(tool_calls=[_tool_call("a", "search_corpus", "{}")], message={"role": "assistant"}),
        _turn(text="nothing here covers it"),
    ]
    graph, _ = _graph_run(
        monkeypatch_factory, turns, [], max_hops=2, fallback_policy="corpus_first_weak"
    )

    assert str(graph.fallback_reason) == "empty"
    assert graph.fallback_opened is True


# an empty turn before the cap still ends in the no-evidence turn when nothing was found