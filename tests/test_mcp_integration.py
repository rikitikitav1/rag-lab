import asyncio
from types import SimpleNamespace

import errors
import httpx
import mcp_client
import pytest
from fastapi.testclient import TestClient
from fastmcp.exceptions import ToolError
from models.mcp_integration import McpStatus, can_switch


@pytest.fixture
def client(monkeypatch):
    import bootstrap

    monkeypatch.setattr(bootstrap, "bootstrap_models", lambda: None)

    import server
    from orm.async_db import get_session

    async def _dummy_session():
        yield None

    server.app.dependency_overrides[get_session] = _dummy_session
    with TestClient(server.app) as c:
        yield c
    server.app.dependency_overrides.clear()


def test_can_switch_valid():
    assert can_switch(McpStatus.disabled, McpStatus.active)
    assert can_switch(McpStatus.active, McpStatus.disabled)
    assert can_switch(McpStatus.active, McpStatus.unreachable)
    assert can_switch(McpStatus.unreachable, McpStatus.active)
    assert can_switch(McpStatus.unreachable, McpStatus.disabled)


def test_can_switch_invalid():
    assert not can_switch(McpStatus.disabled, McpStatus.unreachable)
    assert not can_switch(McpStatus.active, McpStatus.active)


def _integration(**kwargs):
    defaults = {
        "name": "some",
        "status": McpStatus.active,
        "auth": None,
        "last_checked_at": None,
        "last_error": None,
    }
    return SimpleNamespace(**{**defaults, **kwargs})


@pytest.fixture
def secret_allowlist(monkeypatch):
    import config

    monkeypatch.setattr(
        config.settings.mcp_integrations, "secret_env", ["SOME_TOKEN", "SOME_KEY"]
    )


def test_auth_headers_bearer(monkeypatch, secret_allowlist):
    import mcp_client

    monkeypatch.setenv("SOME_TOKEN", "secret")
    integration = _integration(auth={"type": "bearer", "token_env": "SOME_TOKEN"})
    assert mcp_client.auth_headers(integration) == {"Authorization": "Bearer secret"}


def test_auth_headers_custom_header(monkeypatch, secret_allowlist):
    import mcp_client

    monkeypatch.setenv("SOME_KEY", "k123")
    integration = _integration(
        auth={"type": "header", "header": "X-Api-Key", "value_env": "SOME_KEY"}
    )
    assert mcp_client.auth_headers(integration) == {"X-Api-Key": "k123"}


def test_auth_headers_missing_env_and_none(monkeypatch, secret_allowlist):
    import mcp_client

    monkeypatch.delenv("SOME_TOKEN", raising=False)
    with_auth = _integration(auth={"type": "bearer", "token_env": "SOME_TOKEN"})
    assert mcp_client.auth_headers(with_auth) == {}
    assert mcp_client.auth_headers(_integration()) == {}


def test_auth_headers_env_outside_allowlist(monkeypatch, secret_allowlist):
    import mcp_client

    monkeypatch.setenv("SNEAKY_TOKEN", "boom")
    integration = _integration(auth={"type": "bearer", "token_env": "SNEAKY_TOKEN"})
    assert mcp_client.auth_headers(integration) == {}


def test_mark_probe_flips_active_to_unreachable():
    from use_cases.mcp_integration import mark_probe

    integration = _integration(status=McpStatus.active)
    mark_probe(integration, "ConnectError: boom")
    assert integration.status == McpStatus.unreachable
    assert integration.last_error == "ConnectError: boom"
    assert integration.last_checked_at is not None


def test_mark_probe_revives_unreachable():
    from use_cases.mcp_integration import mark_probe

    integration = _integration(status=McpStatus.unreachable)
    mark_probe(integration, None)
    assert integration.status == McpStatus.active
    assert integration.last_error is None


def test_mark_probe_keeps_disabled():
    from use_cases.mcp_integration import mark_probe

    integration = _integration(status=McpStatus.disabled)
    mark_probe(integration, "ConnectError: boom")
    assert integration.status == McpStatus.disabled
    assert integration.last_error == "ConnectError: boom"


def _tool(name, description="fine tool", schema=None):
    return SimpleNamespace(name=name, description=description, inputSchema=schema)


def test_tool_cache_truncates_and_skips_bad_names():
    from use_cases.mcp_integration import tool_cache

    tools = [
        _tool("good", "x" * 10_000, {"type": "object"}),
        _tool("bad name!"),
        _tool("no_description", None),
    ]
    cache = tool_cache(tools)
    assert set(cache) == {"good", "no_description"}
    assert len(cache["good"]["description"]) == 500
    assert cache["good"]["parameters"] == {"type": "object"}
    assert cache["no_description"]["description"] == ""


def test_create_rejects_bad_auth_type(client):
    r = client.post(
        "/v1/mcp_integration",
        json={"name": "x", "url": "https://a.com/mcp", "auth": {"type": "oauth"}},
    )
    assert r.status_code == 422


def test_create_rejects_bad_name_and_url(client):
    bad_name = client.post(
        "/v1/mcp_integration", json={"name": "Bad Name", "url": "https://a.com/mcp"}
    )
    bad_url = client.post(
        "/v1/mcp_integration", json={"name": "ok_name", "url": "ftp://a.com"}
    )
    assert bad_name.status_code == 422
    assert bad_url.status_code == 422


def test_update_rejects_unreachable_status(client):
    r = client.put(
        "/v1/mcp_integration/1",
        json={"url": "https://a.com/mcp", "status": "unreachable"},
    )
    assert r.status_code == 422


def test_create_rejects_double_underscore_name(client):
    r = client.post(
        "/v1/mcp_integration", json={"name": "my__int", "url": "https://a.com/mcp"}
    )
    assert r.status_code == 422
    ok = client.post(
        "/v1/mcp_integration", json={"name": "my_int", "url": "not-a-url"}
    )
    assert ok.status_code == 422


@pytest.mark.parametrize(
    ("exc", "kind"),
    [
        (httpx.ReadTimeout("slow"), "timeout"),
        (httpx.ConnectTimeout("slow connect"), "timeout"),
        (httpx.ConnectError("refused"), "connect"),
        (ConnectionRefusedError(), "connect"),
        (ValueError("nonsense"), "unknown"),
    ],
)
def test_classify_transport_failures(exc, kind):
    assert mcp_client.classify(exc) == kind


@pytest.mark.parametrize(
    ("status", "kind"),
    [(401, "auth"), (403, "auth"), (400, "client"), (404, "client"), (500, "server"), (503, "server")],
)
def test_classify_http_status(status, kind):
    response = httpx.Response(status, request=httpx.Request("POST", "http://x/mcp"))
    exc = httpx.HTTPStatusError("boom", request=response.request, response=response)
    assert mcp_client.classify(exc) == kind


def test_classify_unwraps_exception_groups():
    group = ExceptionGroup("anyio", [ValueError("noise"), httpx.ConnectError("refused")])
    assert mcp_client.classify(group) == "connect"


def test_call_tool_returns_the_kind_instead_of_raising(monkeypatch):
    integration = SimpleNamespace(name="deepwiki", max_result_chars=100)

    def boom(_integration):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(mcp_client, "build_client", boom)
    outcome = asyncio.run(mcp_client.call_tool(integration, "ask", {}))
    assert outcome.error_kind == "connect"
    assert outcome.text.startswith(errors.ERROR_PREFIX)
    assert "(connect)" in outcome.text


def test_classify_unwraps_the_fastmcp_runtime_wrapper():
    wrapper = RuntimeError("Client failed to connect: All connection attempts failed")
    wrapper.__cause__ = ExceptionGroup("TaskGroup", [httpx.ConnectError("refused")])
    assert mcp_client.classify(wrapper) == "connect"


def test_classify_marks_server_side_tool_failures():
    assert mcp_client.classify(ToolError("unknown tool")) == "tool"
