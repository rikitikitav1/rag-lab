import pytest
from fastapi.testclient import TestClient


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


def test_eval_run_rerank_with_agent_ok(client, monkeypatch):
    from types import SimpleNamespace

    import api.v1.eval as eval_mod

    monkeypatch.setattr(
        eval_mod.job_queue, "add_job", lambda s, t, o: SimpleNamespace(id=1, type=t, options=o)
    )

    async def _refresh(session, obj):
        return obj

    monkeypatch.setattr(eval_mod, "commit_and_refresh", _refresh)
    r = client.post(
        "/v1/eval/run", json={"set_name": "s", "pipeline": "agent", "rerank": True}
    )
    assert r.status_code == 200


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


@pytest.fixture
def client_empty_db(monkeypatch):
    import bootstrap

    monkeypatch.setattr(bootstrap, "bootstrap_models", lambda: None)

    import server
    from orm.async_db import get_session

    class _Result:
        def all(self):
            return []

    class _Session:
        async def scalars(self, stmt):
            return _Result()

    async def _session():
        yield _Session()

    server.app.dependency_overrides[get_session] = _session
    with TestClient(server.app) as c:
        yield c
    server.app.dependency_overrides.clear()


def test_question_log_snapshot_filters_are_accepted(client_empty_db):
    for query in (
        "rerank=true",
        "rerank_device=cuda",
        "phased=false",
        "empty_retrieval=true",
        "max_distance=0.3",
        "fallback_policy=corpus_first&fallback_policy=agent_choice",
        "fallback_reason=empty",
    ):
        assert client_empty_db.get(f"/v1/question-log?{query}&limit=1").status_code == 200


def test_compare_needs_at_least_one_run(client):
    assert client.get("/v1/eval/compare").status_code == 422


def test_compare_reports_an_unknown_run_instead_of_empty_pools(client_empty_db):
    r = client_empty_db.get("/v1/eval/compare?runs=nope&runs=also_nope")
    assert r.status_code == 404
    assert "nope" in r.json()["detail"]


def test_question_log_rejects_out_of_range_distance(client):
    assert client.get("/v1/question-log?max_distance=5").status_code == 422


def test_question_log_rejects_unknown_fallback_reason(client):
    assert client.get("/v1/question-log?fallback_reason=bogus").status_code == 422


def test_experiment_rejects_unknown_fallback_policy(client):
    r = client.post(
        "/v1/eval/experiment",
        json={
            "set_name": "s",
            "pipeline": "agent",
            "param": "fallback_policy",
            "values": ["corpus_first", "yolo"],
        },
    )
    assert r.status_code == 400
    assert "yolo" in r.json()["detail"]


def test_fallback_policy_is_rejected_for_single_shot(client):
    r = client.post(
        "/v1/eval/experiment",
        json={"set_name": "s", "param": "fallback_policy", "values": ["corpus_first"]},
    )
    assert r.status_code == 400
    assert "agent" in r.json()["detail"]


def test_bulk_cancel_needs_a_filter(client):
    r = client.post("/v1/job/cancel", json={})
    assert r.status_code == 400
