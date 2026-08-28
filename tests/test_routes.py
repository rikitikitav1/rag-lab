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


def test_every_field_a_run_declares_reaches_the_queue(client, monkeypatch):
    # the options dict is copied field by field, so a field added to the request and not
    # to the copy is accepted, validated and dropped. `variant` was dropped that way and
    # eight arms would have run on the configured corpus instead of the asked one
    from types import SimpleNamespace

    import api.v1.eval as eval_mod

    monkeypatch.setattr(
        eval_mod.job_queue, "add_job", lambda s, t, o: SimpleNamespace(id=1, type=t, options=o)
    )

    async def _refresh(session, obj):
        return obj

    monkeypatch.setattr(eval_mod, "commit_and_refresh", _refresh)
    r = client.post(
        "/v1/eval/run",
        json={"set_name": "s", "variant": "clean_1024", "model": "gemma3:4b"},
    )
    assert r.status_code == 200
    options = r.json()["options"]
    assert set(options) == set(eval_mod.EvalRunRequest.model_fields)
    assert options["variant"] == "clean_1024"
    assert options["model"] == "gemma3:4b"


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


def test_two_cuts_of_one_source_are_read_side_by_side(client, monkeypatch):
    # no job and no re-measuring: the reports are already written per source and variant
    from api.v1 import source as source_api
    from models.corpus import DataSource

    rows = [
        DataSource(
            name="alpha",
            ingest_reports={
                "baseline": [{"verdict": "broken", "score": 74, "breaches": ["a.max"]}],
                "clean_1024": [{"verdict": "ok", "score": 100, "breaches": []}],
            },
        ),
        DataSource(
            name="beta",
            ingest_reports={
                "baseline": [{"verdict": "ok", "score": 90, "breaches": []}],
                "clean_1024": [{"verdict": "ok", "score": 92, "breaches": []}],
            },
        ),
        DataSource(name="gamma", ingest_reports={}),
    ]

    class _Session:
        async def scalars(self, _stmt):
            return rows

    monkeypatch.setattr(source_api, "get_session", lambda: None, raising=False)
    import server
    from orm.async_db import get_session

    async def _session():
        yield _Session()

    server.app.dependency_overrides[get_session] = _session
    try:
        out = client.get("/v1/source/compare?variants=baseline&variants=clean_1024").json()
    finally:
        server.app.dependency_overrides.pop(get_session, None)

    assert out["sources"] == 2, "a source nobody measured is not a disagreement"
    assert out["disagreeing"] == 1
    assert out["rows"][0]["moved"] is True and out["rows"][1]["moved"] is False


def test_the_compare_path_is_not_read_as_a_source_id(client):
    # `/compare` has to be declared before `/{id}`, or FastAPI matches the id route first
    out = client.get("/v1/source/compare?variants=baseline")
    assert out.status_code == 422, "one variant is not a comparison"
