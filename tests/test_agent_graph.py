
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
