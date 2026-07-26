import pytest
from fastapi.testclient import TestClient


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


def test_agent_max_hops_zero_422(client):
    r = client.post("/v1/agent/question", json={"text": "x", "max_hops": 0})
    assert r.status_code == 422


def test_agent_max_hops_negative_422(client):
    r = client.post("/v1/agent/question", json={"text": "x", "max_hops": -1})
    assert r.status_code == 422


def test_agent_language_invalid_422(client):
    r = client.post("/v1/agent/question", json={"text": "x", "language": "xx"})
    assert r.status_code == 422


def test_eval_run_pipeline_invalid_422(client):
    r = client.post("/v1/eval/run", json={"set_name": "s", "pipeline": "bogus"})
    assert r.status_code == 422


def test_eval_run_rerank_with_agent_400(client):
    r = client.post(
        "/v1/eval/run", json={"set_name": "s", "pipeline": "agent", "rerank": True}
    )
    assert r.status_code == 400


def test_job_limit_over_max_422(client):
    r = client.get("/v1/job", params={"limit": 9999})
    assert r.status_code == 422


def test_question_log_pipeline_invalid_422(client):
    r = client.get("/v1/question-log", params={"pipeline": "bogus"})
    assert r.status_code == 422


def test_import_too_large_413(client):
    big = b"x" * (5 * 1024 * 1024 + 1)
    r = client.post(
        "/v1/questions/import", files={"file": ("big.txt", big)}, data={"set_name": "s"}
    )
    assert r.status_code == 413


def test_body_over_max_413(client):
    big = b"x" * (7 * 1024 * 1024)
    r = client.post(
        "/v1/questions/import", files={"file": ("big.txt", big)}, data={"set_name": "s"}
    )
    assert r.status_code == 413
    assert r.json()["detail"] == "request body too large"


@pytest.mark.parametrize(
    "name",
    ["postgresql/ns/m", "localhost/ns/m", "evil.com/ns/m", "hf.co/../x", "a/b/c/d"],
)
def test_model_create_rejects_bad_registry_422(client, name):
    r = client.post("/v1/model", json={"name": name})
    assert r.status_code == 422


def test_sort_order_invalid_422(client):
    r = client.get("/v1/job", params={"sort_order": "descending"})
    assert r.status_code == 422
