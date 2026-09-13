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
    assert EngineCreateRequest(name="c", kind="openai_compatible", env_prefix="C", placement="remote").balance_reader == "none"
    with pytest.raises(ValidationError, match="unknown balance reader bogus"):
        EnginePatchRequest(balance_reader="bogus")
