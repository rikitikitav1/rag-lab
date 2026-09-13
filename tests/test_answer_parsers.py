import json
from pathlib import Path
from types import SimpleNamespace

import engines
import pytest
from engines import answer_parsers
from models.registry import EngineKind, Placement, Role

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "cloud"
CLOUD = engines.EngineSpec(8, "gonka", EngineKind.openai_compatible, "GONKA", Placement.remote)


def _content(name: str) -> str | None:
    return json.loads((FIXTURES / f"{name}.json").read_text())["message"]["content"]


def test_a_closed_thinking_block_is_cut_and_counted():
    got = answer_parsers.parse("think_tags", _content("minimax_plain"))
    assert got.text.startswith("A Docker image is") and "<think>" not in got.text
    assert got.reasoning_chars > 100 and not got.reasoning_unclosed and got.leftover_markers == ()


def test_an_open_thinking_block_before_a_call_leaves_no_text_and_says_so():
    # MiniMax does not close its thinking before a call, even with room left: 69 tokens, `tool_calls`
    got = answer_parsers.parse("think_tags", _content("minimax_tool_long"))
    assert got.text == "" and got.reasoning_unclosed and got.leftover_markers == ()


def test_deepseek_call_markup_is_cut_and_the_markup_is_noted():
    got = answer_parsers.parse("deepseek_tools", _content("deepseek_tool"))
    assert got.text == "" and got.call_markup_in_content and got.leftover_markers == ()
    plain = _content("deepseek_plain")
    assert answer_parsers.parse("deepseek_tools", plain).text == plain, "a clean answer stays byte for byte"


def test_the_wrong_parser_leaves_markers_and_none_touches_nothing():
    raw = _content("minimax_plain")
    none = answer_parsers.parse("none", raw)
    assert none.text == raw and "<think>" in none.leftover_markers
    assert answer_parsers.parse("deepseek_tools", _content("deepseek_tool")).leftover_markers == ()
    assert answer_parsers.parse("think_tags", _content("deepseek_tool")).leftover_markers != ()


def test_parsers_combine_in_a_fixed_order_and_carry_versions():
    assert answer_parsers.label("think_tags+minimax_tools") == "minimax_tools@1+think_tags@1"
    with pytest.raises(ValueError, match="unknown answer parser bogus"):
        answer_parsers.refuse_unknown("think_tags+bogus")
    # the call comes out first, or an open thinking block would swallow it
    got = answer_parsers.parse("think_tags+minimax_tools", "<think>plan <tool_call>[{}]</tool_call> rest")
    assert got.call_markup_in_content and got.reasoning_unclosed and got.text == ""


def test_ask_hands_on_the_cut_text_and_what_was_cut(monkeypatch):
    import llm

    reply = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=_content("minimax_plain")), finish_reason="stop")],
        usage=SimpleNamespace(prompt_tokens=47, completion_tokens=157),
    )
    monkeypatch.setattr(llm, "resolve_for", lambda role, model=None: engines.Resolved("MiniMaxAI/MiniMax-M2.7", CLOUD, "think_tags"))
    monkeypatch.setattr(llm, "_params", lambda *a, **kw: {})
    monkeypatch.setattr(llm, "_complete", lambda *a, **kw: reply)
    got = llm.ask("s", "u", role="judging")
    assert got.text.startswith("A Docker image is") and got.parsed.reasoning_chars > 100


def _probe(monkeypatch, parser: str, fixture: str | None = None, raises: Exception | None = None):
    from use_cases import model_acceptance

    reply = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=fixture and _content(fixture)))])

    def create(**kw):
        if raises is not None:
            raise raises
        return reply

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    monkeypatch.setattr(engines, "spec_of_id", lambda engine_id: CLOUD)
    monkeypatch.setattr(engines, "find_model", lambda name, engine_id=None: engines.Resolved(name, CLOUD, parser))
    monkeypatch.setattr(engines, "client_for", lambda spec: client)
    return lambda: model_acceptance.refuse_unfit_model(Role.judging, "MiniMaxAI/MiniMax-M2.7", CLOUD.id)


def test_acceptance_refuses_a_cloud_model_whose_parser_leaves_markup(monkeypatch):
    # with `none` by default, MiniMax's thinking would reach the judge and read as a short answer
    with pytest.raises(ValueError, match="<think>"):
        _probe(monkeypatch, "none", "minimax_tool_long")()
    _probe(monkeypatch, "think_tags", "minimax_tool_long")()


def _refusal(status: int):
    import httpx
    import openai

    response = httpx.Response(status, request=httpx.Request("POST", "https://b.example/v1/chat/completions"))
    kind = {401: openai.AuthenticationError, 403: openai.PermissionDeniedError}.get(status, openai.APIStatusError)
    return kind("no", response=response, body=None)


@pytest.mark.parametrize("raised", [engines.Unconfigured("engine gonka: key is not configured"), 401, 403])
def test_acceptance_refuses_a_cloud_row_no_call_can_reach(monkeypatch, raised):
    # a row without its key sat on the role with one warning, and every call of the run failed
    error = _refusal(raised) if isinstance(raised, int) else raised
    with pytest.raises(ValueError, match="cannot be called"):
        _probe(monkeypatch, "none", raises=error)()


@pytest.mark.parametrize("status", [429, 503])
def test_acceptance_forgives_a_probe_that_did_not_land(monkeypatch, status):
    assert _probe(monkeypatch, "none", raises=_refusal(status))() is None


def test_the_door_takes_only_a_parser_the_stand_has():
    from api.v1.llm_model import ModelPatchRequest
    from pydantic import ValidationError

    assert ModelPatchRequest(answer_parser="think_tags+minimax_tools").answer_parser
    with pytest.raises(ValidationError):
        ModelPatchRequest(answer_parser="bogus")


def test_a_row_keeps_the_cut_only_when_something_was_cut():
    assert answer_parsers.record(answer_parsers.parse("none", "plain answer")) is None
    kept = answer_parsers.record(answer_parsers.parse("think_tags", _content("minimax_plain")))
    assert kept["reasoning_chars"] > 100 and kept["leftover_markers"] == []


def test_an_open_thinking_block_is_cut_by_length_only_when_the_limit_ended_it():
    # MiniMax leaves it open before every call; only `length` means the answer itself was lost
    assert answer_parsers.parse("think_tags", "<think>still thinking", finish_reason="length").reasoning_cut_by_length
    assert not answer_parsers.parse("think_tags", "<think>about to call", finish_reason="tool_calls").reasoning_cut_by_length


def test_an_agent_row_keeps_the_cut_of_every_hop():
    hops = [answer_parsers.parse("think_tags", "<think>ab</think> x"), None,
            answer_parsers.parse("think_tags", "<think>cde", finish_reason="tool_calls")]
    got = answer_parsers.summarize(hops)
    assert got["reasoning_chars"] == 5 and got["reasoning_unclosed"] and not got["reasoning_cut_by_length"]
    assert answer_parsers.summarize([None]) is None


def test_a_verdict_carries_the_parser_its_answer_was_cut_by(monkeypatch):
    import llm
    from job_handlers import judging
    from use_cases import judge

    cut = answer_parsers.parse("think_tags", '<think>weighing</think>{"reason": "fine", "score": 8}')
    monkeypatch.setattr(judge.llm, "ask", lambda **kw: llm.Completion(
        text=cut.text, prompt_tokens=10, completion_tokens=5, parsed=cut, parser="think_tags@1"))
    verdict = judge.judge("s", "u", model="MiniMaxAI/MiniMax-M2.7")
    assert (verdict.score, verdict.parser) == (8, "think_tags@1")
    assert verdict.answer_parse["reasoning_chars"] == len("weighing")
    stamp = judging._axis_metric(verdict)
    assert stamp["judge_parser"] == "think_tags@1" and stamp["judge_answer_parse"]["reasoning_chars"] == 8
    assert (stamp["judge_prompt_tokens"], stamp["judge_completion_tokens"]) == (10, 5)


def test_compare_refuses_arms_whose_judge_answers_were_cut_by_different_parsers():
    from evals import compare

    said = compare._what_to_read_first(True, True, True, True, True, one_parser=False)
    assert said and "different parsers" in said
    agreed = compare._what_to_read_first(True, True, True, True, True, one_parser=True)
    assert "different parsers" not in (agreed or "")


def test_the_run_snapshot_names_the_parser_of_each_answering_role(monkeypatch):
    from models.registry import Role
    from use_cases import run_snapshot

    local = engines.EngineSpec(1, "ollama", EngineKind.ollama, "OLLAMA", Placement.gpu)
    monkeypatch.setattr(run_snapshot, "model_of", lambda role: engines.Resolved("bge-m3", local))
    monkeypatch.setattr(run_snapshot.card, "model_on_card", lambda spec, name: None)
    monkeypatch.setattr(run_snapshot.llm, "sampler", lambda role, spec: SimpleNamespace(dropped={}))
    *_, parsers = run_snapshot._by_role(engines.Resolved("MiniMaxAI/MiniMax-M2.7", CLOUD, "think_tags+minimax_tools"))
    assert parsers == {Role.generation: "minimax_tools@1+think_tags@1", Role.embedding: "none@1"}
