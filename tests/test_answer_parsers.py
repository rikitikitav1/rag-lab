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


def _probe(monkeypatch, parser: str, fixture: str | None = None, raises: Exception | None = None, counted: bool = True):
    from use_cases import model_acceptance

    reply = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=fixture and _content(fixture)))],
                            usage=SimpleNamespace(prompt_tokens=40, completion_tokens=12) if counted else None)

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


def test_acceptance_refuses_a_cloud_row_that_sends_no_token_usage(monkeypatch):
    # it would sit on the role, and the first run on it would stop on its first call
    with pytest.raises(ValueError, match="sends no token usage"):
        _probe(monkeypatch, "think_tags", "minimax_tool_long", counted=False)()


def test_a_model_s_own_sampler_is_laid_over_its_role_s_and_the_door_takes_sampler_keys_only(monkeypatch):
    # the cloud guest cut a third of its calls at the role's 1024, and the budget is the model's property
    import llm
    from api.v1.llm_model import ModelPatchRequest
    from models.registry import EngineKind, Placement
    from pydantic import ValidationError

    cloud = engines.EngineSpec(8, "gonka", EngineKind.openai_compatible, "GONKA", Placement.remote)
    long = engines.Resolved("deepseek", cloud, "none", {"max_tokens": 8192})
    sent = llm.sampler("ragas", long).sent
    assert sent["max_tokens"] == 8192 and sent.get("temperature") == 0
    role = llm.config.settings.llm.roles["ragas"].options["max_tokens"]
    assert llm.sampler("ragas", engines.Resolved("deepseek", cloud)).sent["max_tokens"] == role
    assert ModelPatchRequest(options={"max_tokens": 8192}).options == {"max_tokens": 8192}
    assert ModelPatchRequest(options={}).options == {}
    for bad in ({"bogus": 1}, {"max_tokens": 0}, {"max_tokens": "8k"}):
        with pytest.raises(ValidationError):
            ModelPatchRequest(options=bad)


def test_compare_reads_the_judge_s_sampler_by_key():
    # temperature and seed choose tokens; a budget is a ceiling and matters only where it cut
    from evals import compare

    said = compare._what_to_read_first(True, True, True, True, True, one_sampler=False)
    assert "temperature or seed" in said
    cut = compare._what_to_read_first(True, True, True, True, True, one_sampler=True, one_budget=False, cut=3)
    assert "3 verdicts ended on the limit" in cut
    held = compare._what_to_read_first(True, True, True, True, True, one_sampler=True, one_budget=False, cut=0)
    assert "changed nothing here" in held


def test_a_budget_that_fills_the_model_s_window_is_refused_at_the_door(monkeypatch):
    from api.v1 import llm_model
    from models.registry import EngineKind, Placement

    local = engines.EngineSpec(1, "ollama", EngineKind.ollama, "OLLAMA", Placement.gpu)
    monkeypatch.setattr(llm_model.engines, "spec_of_id", lambda engine_id: local)
    monkeypatch.setattr(llm_model.engines, "driver", lambda kind: type("D", (), {"window": lambda self, spec, name: 8192})())
    with pytest.raises(ValueError, match="fills the 8192-token window"):
        llm_model.refuse_a_budget_over_the_window(1, "qwen2.5:7b", {"max_tokens": 8192})
    llm_model.refuse_a_budget_over_the_window(1, "qwen2.5:7b", {"max_tokens": 4096})
    llm_model.refuse_a_budget_over_the_window(1, "qwen2.5:7b", {"temperature": 0})


def test_a_verdict_the_limit_cut_says_so_on_its_row():
    from types import SimpleNamespace

    from job_handlers import judging

    cut = SimpleNamespace(reason="r", elapsed=1.0, model="m", prompt_tokens=1, completion_tokens=1024, cut_by_length=True)
    whole = SimpleNamespace(reason="r", elapsed=1.0, model="m", prompt_tokens=1, completion_tokens=10)
    assert judging._axis_metric(cut)["judge_cut_by_length"] is True
    assert "judge_cut_by_length" not in judging._axis_metric(whole)


def test_a_reply_the_limit_ended_before_its_score_fails_as_a_cut_and_its_axis_says_so(monkeypatch):
    # the judge wrote a reason, then tabs until its limit, and the row read "no JSON object"
    from types import SimpleNamespace

    from job_handlers import judging
    from use_cases import judge

    said = SimpleNamespace(text='{\n  "reason": "grounded"\n\n \t\t\t\t\n', prompt_tokens=1937,
                           completion_tokens=1024, finish_reason="length")
    monkeypatch.setattr(judge.llm, "ask", lambda **kw: said)
    with pytest.raises(judge.JudgeCut, match="limit at 1024 tokens before its score"):
        judge.judge("system", "user", model="m")
    err = judging._error_text(judge.JudgeCut("the judge hit its output limit at 1024 tokens before its score"))
    assert judging._errored_metric({}, "faithfulness", err)["judge_cut_by_length"] is True
    assert "judge_cut_by_length" not in judging._errored_metric({}, "faithfulness", "ValueError: no JSON object")
    said.finish_reason = "stop"
    with pytest.raises(ValueError, match="no JSON object") as broken:
        judge.judge("system", "user", model="m")
    assert not isinstance(broken.value, judge.JudgeCut), "a reply that ended on its own is a broken format"


def test_arms_judged_under_different_json_rules_read_as_two_rulers():
    from evals import compare

    said = compare._what_to_read_first(True, True, True, True, True, one_parser=True, one_grammar=False)
    assert "measures the grammar" in said
    assert compare._grammar_of({"num_ctx": 8192}) != compare._grammar_of(
        {"json_backend": "xgrammar", "json_disable_any_whitespace": "True"})
