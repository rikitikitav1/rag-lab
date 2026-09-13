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
