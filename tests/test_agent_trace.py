from types import SimpleNamespace as NS

import pytest


def _row(id, trace, outcome="answered"):
    return NS(id=id, metrics={"trace": trace, "outcome": outcome})


def test_the_ask_door_records_every_verdict_with_its_stage(monkeypatch):
    # a replay that asked the model again would compare two runs, not one
    import llm
    from use_cases import agent

    monkeypatch.setattr(llm, "ask", lambda **kw: NS(text=" YES "))
    result = agent.AgentResult()
    ask = agent.ask_door(result, "generation", None)
    assert ask("tool_match", "deepwiki__read", "system", "user") == "YES"
    assert result.asks == [{"stage": "tool_match", "key": "deepwiki__read", "text": "YES"}]
    assert result.stages["tool_match"]["calls"] == 1


def test_admission_asks_through_the_door_and_only_for_tools_that_need_a_value(monkeypatch):
    import prompt_repo
    from use_cases import agent

    monkeypatch.setattr(prompt_repo, "active_template", lambda purpose: "system")
    result = agent.AgentResult()
    asked = []

    def ask(stage, key, system, user):
        asked.append((stage, key))
        return "no"

    tools = {
        "needs_value": NS(parameters={"required": ["repo"], "properties": {"repo": {}}}),
        "free": NS(parameters={"required": [], "properties": {}}),
    }
    admitted = agent._admissible("q", tools, result, ask)
    # a tool with nothing to fill in is admitted without a call; the other one asked and was refused
    assert sorted(admitted) == ["free"]
    assert asked == [("tool_match", "needs_value")]


def test_the_graph_context_hands_no_live_ask_of_its_own():
    # a node that asks without a door must fail loudly, not reach the model behind the replay's back
    from orchestrators import graph

    ctx = graph.context(remote={}, gate=None, role="generation", model=None)
    assert "chat" in ctx and "dispatch" in ctx
    with pytest.raises(KeyError):
        ctx["ask"]


def test_the_trace_report_counts_hops_nodes_and_what_the_gate_said():
    from evals import trace

    rows = [
        _row(1, [
            {"node": "model", "hop": 1, "tool_calls": 1},
            {"node": "retrieve", "hop": 1, "calls": 1, "verdict": "none", "sources": 5},
            {"node": "model", "hop": 2, "answered": True},
            {"node": "final", "hop": 2, "finished_by": "answer"},
        ]),
        _row(2, [
            {"node": "model", "hop": 1, "tool_calls": 1},
            {"node": "retrieve", "hop": 1, "calls": 1, "verdict": "weak", "sources": 5},
            {"node": "fallback", "hop": 1, "verdict": "weak", "dropped": 5, "opened": True},
            {"node": "model", "hop": 2, "answered": True},
        ], outcome="answered_ungrounded"),
        NS(id=3, metrics={"outcome": "answered"}),
    ]
    got = trace.report(rows)
    assert got["rows"] == 3 and got["rows_traced"] == 2
    assert got["rows_without_trace"] == [3]
    assert got["hops"] == {2: 2}
    assert got["nodes"]["model"] == {"rows": 2, "steps": 4}
    assert got["nodes"]["retrieve"] == {"rows": 2, "steps": 2}
    assert got["verdicts"] == {"none": 1, "weak": 1}
    assert got["fallback"] == {"opened": 1, "dropped_context": 1, "announced": 0}
    assert got["outcomes_by_hops"] == {2: {"answered": 1, "answered_ungrounded": 1}}


def test_a_failed_hop_is_counted_and_named():
    from evals import trace

    got = trace.report([_row(4, [{"node": "model", "hop": 1, "failed": True}], outcome="error")])
    assert got["failed_hops"] == 1 and got["outcomes_by_hops"] == {1: {"error": 1}}


def test_a_probe_refused_over_the_window_fails_its_tool_not_the_whole_answer(monkeypatch):
    # the refusal is a ValueError, so the door answered 500 and the question got no row at all
    import llm
    import prompt_repo
    from use_cases import agent

    monkeypatch.setattr(prompt_repo, "active_template", lambda purpose: "system")
    result = agent.AgentResult()

    def refused(stage, key, system, user):
        raise llm.InputOverWindow("the input is at least 9000 tokens against the 8192-token window")

    tools = {"needs_value": NS(parameters={"required": ["repo"], "properties": {"repo": {}}})}
    assert agent._admissible("q", tools, result, refused) == {}
    assert result.tool_errors == {"needs_value": "tool_match"}
