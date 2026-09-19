import json
from pathlib import Path

import engines
import pytest
import requests
from engines import balances
from models.registry import EngineKind, Placement

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "cloud"
CLOUD = engines.EngineSpec(8, "gonka", EngineKind.openai_compatible, "GONKA", Placement.remote)


def _answer(status: int, body: dict | None = None) -> requests.Response:
    response = requests.Response()
    response.status_code = status
    response._content = json.dumps(body or {}).encode()
    response.url = "https://b.example/v1/auth/key"
    return response


@pytest.fixture
def keyed(monkeypatch):
    monkeypatch.setenv("GONKA_BASE_URL", "https://b.example/v1")
    monkeypatch.setenv("GONKA_API_KEY", "fakefakefakefake")


def test_the_gonka_reader_reads_the_balance_off_its_key_route(monkeypatch, keyed):
    body = json.loads((FIXTURES / "gonka_auth_key.json").read_text())
    asked = []
    monkeypatch.setattr(balances.requests, "get", lambda url, headers, timeout: asked.append(url) or _answer(200, body))
    got = balances.read(CLOUD, "gonka_key")
    assert asked == ["https://b.example/v1/auth/key"], "the page's /v1 is not doubled"
    assert (got["balance"], got["unit"], got["why"]) == (1.4199308, "usd", None)


def test_a_cloud_that_cannot_say_its_balance_says_why(monkeypatch, keyed):
    assert "no balance reader named" in balances.read(CLOUD, "none")["why"]
    monkeypatch.setattr(balances.requests, "get", lambda url, headers, timeout: _answer(404))
    refused = balances.read(CLOUD, "gonka_key")
    assert refused["balance"] is None and refused["why"] == "gonka answered http 404"
    monkeypatch.delenv("GONKA_API_KEY")
    assert "key is not configured" in balances.read(CLOUD, "gonka_key")["why"]


def test_the_door_takes_only_a_reader_the_stand_has():
    from api.v1.engine import EngineCreateRequest, EnginePatchRequest
    from pydantic import ValidationError

    assert EnginePatchRequest(balance_reader="gonka_key").balance_reader == "gonka_key"
    cloud = EngineCreateRequest(name="c", kind="openai_compatible", env_prefix="C", placement="remote")
    assert cloud.balance_reader == "none"
    with pytest.raises(ValidationError, match="unknown balance reader bogus"):
        EnginePatchRequest(balance_reader="bogus")


CLOUD = engines.EngineSpec(8, "gonka", EngineKind.openai_compatible, "GONKA", Placement.remote)


def test_a_job_reads_only_the_clouds_that_can_say_their_balance(monkeypatch):
    quiet = engines.EngineSpec(9, "other", EngineKind.openai_compatible, "OTHER", Placement.remote)
    monkeypatch.setattr(balances, "_clouds", lambda: [(CLOUD, "gonka_key"), (quiet, "none")])
    assert balances.readable() == [(CLOUD, "gonka_key")]


def test_a_retried_job_keeps_its_first_before_and_its_last_after():
    import job_queue

    first = job_queue.merged_balances(None, {"gonka": {"balance": 1.5, "unit": "usd", "read_at": "t0"}},
                                      {"gonka": {"balance": 1.45, "unit": "usd", "read_at": "t1"}})
    again = job_queue.merged_balances(first, {"gonka": {"balance": 1.45, "unit": "usd", "read_at": "t2"}},
                                      {"gonka": {"balance": 1.4, "unit": "usd", "read_at": "t3"}})
    assert again == {"gonka": {"before": 1.5, "before_at": "t0", "after": 1.4, "after_at": "t3", "unit": "usd",
                               "why": None}}


def test_a_balance_nobody_could_read_says_why_and_is_not_a_zero():
    import job_queue

    failed = {"gonka": {"balance": None, "unit": None, "read_at": "t0", "why": "gonka answered http 503"}}
    read = {"gonka": {"balance": 1.4, "unit": "usd", "read_at": "t1", "why": None}}
    out = job_queue.merged_balances(None, failed, read)["gonka"]
    assert out["before"] is None and out["after"] == 1.4 and out["why"] == "gonka answered http 503"
    assert job_queue.merged_balances(None, {}, read)["gonka"]["why"] == "not read before the attempt"


def test_the_worker_reads_every_cloud_before_a_job_and_after_it_before_the_status(monkeypatch):
    import job_queue
    import worker

    events, left = [], iter([1.5, 1.4])
    monkeypatch.setitem(worker.HANDLERS, "spender", lambda options: events.append("handler"))
    monkeypatch.setitem(worker.job_specs.LOADS, "spender", ("generation",))
    monkeypatch.setattr(
        worker.job_queue, "claim_next", lambda queues: job_queue.ClaimedJob(id=9, type="spender", options={})
    )
    monkeypatch.setattr(worker.job_specs, "check", lambda *a, **kw: None)
    monkeypatch.setattr(balances, "readable", lambda: [(CLOUD, "gonka_key")])
    monkeypatch.setattr(balances, "read", lambda spec, reader: events.append("read") or {
        "balance": next(left), "unit": "usd", "read_at": "t", "why": None})
    monkeypatch.setattr(worker.job_queue, "add_tokens", lambda id, record: events.append("tokens"))
    monkeypatch.setattr(worker.job_queue, "add_balances", lambda id, before, after: events.append(
        ("balances", before["gonka"]["balance"], after["gonka"]["balance"])))
    monkeypatch.setattr(worker.job_queue, "complete", lambda id, elapsed=None: events.append("done"))
    assert worker.run_once(["default"])
    assert events == ["read", "handler", "tokens", "read", ("balances", 1.5, 1.4), "done"]


def test_a_job_that_calls_no_model_asks_no_broker(monkeypatch):
    import job_queue
    import worker

    def unasked():
        raise AssertionError("a job that calls no model asked the broker")

    monkeypatch.setitem(worker.HANDLERS, "quiet", lambda options: None)
    monkeypatch.setitem(worker.job_specs.LOADS, "quiet", ())
    monkeypatch.setattr(
        worker.job_queue, "claim_next", lambda queues: job_queue.ClaimedJob(id=10, type="quiet", options={})
    )
    monkeypatch.setattr(worker.job_specs, "check", lambda *a, **kw: None)
    monkeypatch.setattr(balances, "readable", unasked)
    monkeypatch.setattr(worker.job_queue, "add_tokens", lambda id, record: None)
    monkeypatch.setattr(worker.job_queue, "complete", lambda id, elapsed=None: None)
    assert worker.run_once(["default"])


def test_a_broker_answering_an_odd_shape_is_a_reason_not_an_exception(monkeypatch, keyed):
    # `{"data": null}` raised TypeError out of the worker, and the job hung in running
    monkeypatch.setattr(balances.requests, "get", lambda url, headers, timeout: _answer(200, {"data": None}))
    seen = balances.read(CLOUD, "gonka_key")
    assert seen["balance"] is None and "TypeError" in seen["why"]


def test_the_door_holds_a_balance_a_few_seconds_instead_of_asking_the_broker_each_time(monkeypatch):
    asked = []
    monkeypatch.setattr(balances, "_HELD", {})
    monkeypatch.setattr(balances, "_clouds", lambda: [(CLOUD, "gonka_key")])
    monkeypatch.setattr(balances, "read", lambda spec, reader: asked.append(1) or {"balance": 1.0})
    balances.summary()
    balances.summary()
    assert asked == [1]


def test_a_remote_engine_refuses_an_address_its_key_would_cross_in_the_clear(monkeypatch):
    from engines import core

    with pytest.raises(engines.Unconfigured, match="https"):
        core._refuse_unusable(CLOUD, "http://broker.example")
    assert core._refuse_unusable(CLOUD, "https://broker.example") == "https://broker.example"


def test_a_pull_from_the_hub_carries_its_token_and_no_other_secret(monkeypatch):
    import redaction

    assert redaction.is_secret("GONKA_API_KEY") and redaction.is_secret("SOME_SECRET")
    assert not redaction.is_secret("HF_HOME") and not redaction.is_secret("OLLAMA_BASE_URL")
