from pathlib import Path

import engines
import httpx
import openai
import pytest
from errors import StandFault
from models.registry import EngineKind, Placement

ROOT = Path(__file__).resolve().parent.parent
CLOUD = engines.EngineSpec(8, "gonka", EngineKind.openai_compatible, "GONKA", Placement.remote)


def test_a_key_is_cut_by_its_value_and_from_a_query(monkeypatch):
    # a broker quoted its own internals in an error body, and a key can come back the same way
    from redaction import redact

    monkeypatch.setenv("GONKA_API_KEY", "fakefakefakefake")
    said = redact("401 for fakefakefakefake at https://b.example/v1?key=abcdef&x=1")
    assert "fakefakefakefake" not in said and "abcdef" not in said
    assert said.endswith("?key=<key>&x=1")
    assert redact("a short word stays") == "a short word stays"


def test_every_log_line_goes_through_the_cut(monkeypatch):
    import logging_setup

    monkeypatch.setenv("GONKA_API_KEY", "fakefakefakefake")
    event = logging_setup._redacted(None, "error", {"event": "x", "error": "saw fakefakefakefake", "n": 3})
    assert event == {"event": "x", "error": "saw <key>", "n": 3}


def test_the_worker_never_writes_a_raw_exception_text_into_a_job():
    source = (ROOT / "app" / "worker.py").read_text()
    assert '"error": str(' not in source, "a job row is read without a key through GET /v1/job"


def test_an_address_with_a_key_in_its_query_is_refused(monkeypatch):
    from engines import core

    monkeypatch.setenv("GONKA_BASE_URL", "https://b.example/v1?key=abcdef123")
    with pytest.raises(engines.Unconfigured, match="address carries a key"):
        core.base_url(CLOUD)


@pytest.mark.parametrize("status", [402, 429])
def test_a_quota_or_a_rate_limit_stops_the_run(monkeypatch, status):
    import llm

    request = httpx.Request("POST", "https://b.example/v1/chat/completions")
    refusal = openai.APIStatusError("no", response=httpx.Response(status, request=request), body=None)

    def create(**kw):
        raise refusal

    client = type("C", (), {"chat": type("Ch", (), {"completions": type("Co", (), {"create": staticmethod(create)})})})
    monkeypatch.setattr(llm.engines, "client_for", lambda spec: client)
    monkeypatch.setattr(llm, "_card_for", lambda spec, name: __import__("contextlib").nullcontext())
    with pytest.raises(llm.BrokerRefused, match=f"http {status}") as caught:
        llm._complete(CLOUD, "MiniMaxAI/MiniMax-M2.7", [], {})
    assert isinstance(caught.value, StandFault), "a loop forgives a RuntimeError as one failed row"


def _throttled(monkeypatch, headers: dict, answers_after: int | None):
    import llm

    request = httpx.Request("POST", "https://b.example/v1/chat/completions")
    asked, slept = [], []

    def create(**kw):
        asked.append(kw)
        if answers_after is not None and len(asked) > answers_after:
            return "the answer"
        raise openai.APIStatusError("no", response=httpx.Response(429, headers=headers, request=request), body=None)

    client = type("C", (), {"chat": type("Ch", (), {"completions": type("Co", (), {"create": staticmethod(create)})})})
    monkeypatch.setattr(llm.engines, "client_for", lambda spec: client)
    monkeypatch.setattr(llm, "_card_for", lambda spec, name: __import__("contextlib").nullcontext())
    monkeypatch.setattr(llm.time, "sleep", slept.append)
    return llm, asked, slept


def test_a_429_that_names_its_pause_is_waited_and_asked_again(monkeypatch):
    # two keys met the same throttle after twenty calls, and the first 429 stopped a guest owing 143 rows
    llm, asked, slept = _throttled(monkeypatch, {"retry-after": "7"}, answers_after=1)
    with llm.accounting() as tally:
        assert llm._complete(CLOUD, "m", [], {}, "ragas") == "the answer"
    assert slept == [7.0] and len(asked) == 2
    assert tally.record() == {"ragas": [{"engine": "gonka", "model": "m", "prompt": 0, "completion": 0,
                                         "calls": 0, "max_prompt": 0, "paced": 1, "paced_seconds": 7.0}]}


@pytest.mark.parametrize("headers", [{}, {"retry-after": "3600"}, {"retry-after": "Wed, 21 Oct 2026 07:28:00 GMT"}])
def test_a_429_without_a_pause_or_with_a_cap_stops_the_run_at_once(monkeypatch, headers):
    llm, asked, slept = _throttled(monkeypatch, headers, answers_after=None)
    with pytest.raises(llm.BrokerRefused, match="http 429"):
        llm._complete(CLOUD, "m", [], {}, "ragas")
    assert slept == [] and len(asked) == 1


def test_a_throttle_that_never_lifts_stops_after_its_waits(monkeypatch):
    llm, asked, slept = _throttled(monkeypatch, {"retry-after-ms": "1500"}, answers_after=None)
    with pytest.raises(llm.BrokerRefused):
        llm._complete(CLOUD, "m", [], {}, "ragas")
    assert slept == [1.5] * llm.PACE_TRIES and len(asked) == llm.PACE_TRIES + 1


@pytest.mark.parametrize("status", [500, 502, 503])
def test_a_broker_that_says_it_is_broken_stops_the_run_and_a_local_server_fails_one_row(monkeypatch, status):
    # a 502 from the broker was forgiven row by row; ollama's 500 on one broken tool call is that row's alone
    import llm

    request = httpx.Request("POST", "https://b.example/v1/chat/completions")
    failure = openai.APIStatusError("no", response=httpx.Response(status, request=request), body=None)

    def create(**kw):
        raise failure

    chat = type("C", (), {"chat": type("Ch", (), {"completions": type("Co", (), {"create": staticmethod(create)})})})
    embed = type("E", (), {"embeddings": type("Em", (), {"create": staticmethod(create)})})
    monkeypatch.setattr(llm, "_card_for", lambda spec, name: __import__("contextlib").nullcontext())
    local = engines.EngineSpec(2, "vllm", EngineKind.vllm, "VLLM", Placement.gpu)
    monkeypatch.setattr(llm.engines, "client_for", lambda spec: chat)
    with pytest.raises(llm.ServerFailed, match=f"http {status}") as caught:
        llm._complete(CLOUD, "m", [], {})
    assert isinstance(caught.value, StandFault)
    with pytest.raises(RuntimeError) as caught:
        llm._complete(local, "m", [], {})
    assert not isinstance(caught.value, StandFault)
    monkeypatch.setattr(llm.engines, "client_for", lambda spec: embed)
    with pytest.raises(llm.ServerFailed):
        llm._embeddings(engines.Resolved("m", CLOUD), ["a"], "embedding")


def test_a_run_that_answered_nothing_stops_for_good_and_is_not_answered_again(monkeypatch):
    import passes
    from evals import runner
    from job_handlers import base, evaluation

    walk = passes.Pass(None, ())
    with pytest.raises(runner.NoAnswers, match="0 of 3"):
        walk.close(owed=3, done=0, nothing=runner.NoAnswers)
    walk.close(owed=3, done=1, nothing=runner.NoAnswers)

    def stopped(**kw):
        raise runner.NoAnswers("r: 0 of 3 questions answered")

    monkeypatch.setattr(evaluation, "require_role_ready", lambda *a, **kw: None)
    monkeypatch.setattr(evaluation, "require_card", lambda *a, **kw: None)
    monkeypatch.setattr(evaluation.runner, "run", stopped)
    monkeypatch.setattr(evaluation, "_claims_on", lambda run_name, job_id: (0, []))
    # the worker retries anything but Final, and a retry answers every question again
    with pytest.raises(base.Final, match="0 of 3"):
        evaluation.eval_run({"run_name": "r"})


@pytest.mark.parametrize("status", [401, 403])
def test_a_refused_key_stops_the_run_instead_of_failing_every_row(monkeypatch, status):
    # a key revoked mid-run failed each row alone, and on the agent path the run read done
    import llm

    request = httpx.Request("POST", "https://b.example/v1/chat/completions")
    kind = {401: openai.AuthenticationError, 403: openai.PermissionDeniedError}[status]
    failure = kind("no", response=httpx.Response(status, request=request), body=None)

    def create(**kw):
        raise failure

    chat = type("C", (), {"chat": type("Ch", (), {"completions": type("Co", (), {"create": staticmethod(create)})})})
    monkeypatch.setattr(llm, "_card_for", lambda spec, name: __import__("contextlib").nullcontext())
    monkeypatch.setattr(llm.engines, "client_for", lambda spec: chat)
    with pytest.raises(llm.KeyRefused, match=f"http {status}") as caught:
        llm._complete(CLOUD, "m", [], {})
    assert isinstance(caught.value, StandFault)


def test_every_door_s_run_meets_the_taken_name_in_the_worker(monkeypatch):
    # the experiment door and /v1/job queued a second run under a taken name, and a question landed twice
    from job_handlers import base, evaluation

    seen = {}
    monkeypatch.setattr(evaluation, "require_role_ready", lambda *a, **kw: None)
    monkeypatch.setattr(evaluation, "require_card", lambda *a, **kw: None)
    monkeypatch.setattr(evaluation.runner, "run", lambda **kw: seen.update(kw) or 1)
    monkeypatch.setattr(evaluation, "_claims_on", lambda run_name, job_id: (3, [2739]))
    with pytest.raises(base.Final, match="is taken"):
        evaluation.eval_run({"run_name": "r", "_job_id": 2743})
    # its own first attempt wrote those rows: the retry answers the rest
    monkeypatch.setattr(evaluation, "_claims_on", lambda run_name, job_id: (3, []))
    evaluation.eval_run({"run_name": "r", "_job_id": 2743, "attempts": 1})
    assert seen["resume"] is True


def test_a_run_reclaimed_after_a_restart_resumes_its_own_rows(monkeypatch):
    import job_queue
    from job_handlers import evaluation

    seen = {}
    monkeypatch.setattr(evaluation, "require_role_ready", lambda *a, **kw: None)
    monkeypatch.setattr(evaluation, "require_card", lambda *a, **kw: None)
    monkeypatch.setattr(evaluation.runner, "run", lambda **kw: seen.update(kw) or 1)
    monkeypatch.setattr(evaluation, "_claims_on", lambda run_name, job_id: (3, []))
    evaluation.eval_run(job_queue.reclaimed({"run_name": "r", "_job_id": 2743}))
    assert seen["resume"] is True
    assert job_queue.reclaimed({"attempts": 1})["attempts"] == 2


def test_a_pause_that_is_not_a_number_of_seconds_is_no_pause():
    # a nan passed the ceiling check and then broke the sleep inside the handler
    import httpx
    import llm
    import openai

    request = httpx.Request("POST", "https://b.example/v1/chat/completions")
    for header in ("nan", "-1", "inf"):
        throttled = openai.RateLimitError(
            "slow down", response=httpx.Response(429, headers={"retry-after": header}, request=request), body=None
        )
        assert llm._retry_after(throttled) is None, header
