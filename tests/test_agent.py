from types import SimpleNamespace

import agent_tools
import llm
from use_cases import agent


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
        lambda name, args: agent_tools.ToolResult(
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
        {"role": "tool", "tool_call_id": "c", "content": f"{chat.ERROR_PREFIX}boom"},
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
        lambda name, args: agent_tools.ToolResult(content="c", meta={"sources": [name]}),
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
