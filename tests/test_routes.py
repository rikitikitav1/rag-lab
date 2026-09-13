import pytest
from fastapi.testclient import TestClient
from stand_specs import queued_job as _queued_job


def test_agent_max_hops_zero_422(client):
    r = client.post("/v1/agent/question", json={"text": "x", "max_hops": 0})
    assert r.status_code == 422


def test_agent_max_hops_negative_422(client):
    r = client.post("/v1/agent/question", json={"text": "x", "max_hops": -1})
    assert r.status_code == 422


def test_agent_language_invalid_422(client):
    r = client.post("/v1/agent/question", json={"text": "x", "language": "xx"})
    assert r.status_code == 422


def test_the_agent_door_refuses_foreign_vectors_rather_than_answering_without_the_corpus(
    client, monkeypatch
):
    import api.v1.agent as agent_door

    import db

    def foreign(*a, **kw):
        raise db.ForeignVectors("variant holds vectors of bge-m3@ollama-cpu")

    monkeypatch.setattr(agent_door, "wait_for_the_card", lambda *roles: None)
    monkeypatch.setattr(agent_door.agent, "run", foreign)
    r = client.post("/v1/agent/question", json={"text": "x"})
    assert r.status_code == 409 and "bge-m3@ollama-cpu" in r.json()["detail"]


def test_eval_run_pipeline_invalid_422(client):
    r = client.post("/v1/eval/run", json={"set_name": "s", "pipeline": "bogus"})
    assert r.status_code == 422


def test_eval_run_rerank_with_agent_ok(client, monkeypatch):
    import api.v1.eval as eval_mod

    monkeypatch.setattr(
        eval_mod.job_queue, "add_job", lambda s, t, o: _queued_job(t, o)
    )

    async def _refresh(session, obj):
        return obj

    monkeypatch.setattr(eval_mod, "commit_and_refresh", _refresh)
    r = client.post(
        "/v1/eval/run", json={"set_name": "s", "pipeline": "agent", "rerank": True}
    )
    assert r.status_code == 200


def test_every_field_a_run_declares_reaches_the_queue(client, monkeypatch):
    # the options dict is copied field by field, so a new field is accepted and never carried
    import api.v1.eval as eval_mod

    monkeypatch.setattr(
        eval_mod.job_queue, "add_job", lambda s, t, o: _queued_job(t, o)
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


class _SessionOfLiveJobs:
    async def scalars(self, _statement):
        return [11, 12]

    async def commit(self):
        return None


def _with_live_jobs():
    import server
    from orm.async_db import get_session

    async def _fake():
        yield _SessionOfLiveJobs()

    server.app.dependency_overrides[get_session] = _fake


def test_cancelling_a_type_with_no_run_name_is_said_out_loud(client, monkeypatch):
    # `type: judge_answers` alone is every live judge job, which is several arms of several runs
    import job_queue

    r = client.post("/v1/job/cancel", json={"type": "judge_answers"})
    assert r.status_code == 400
    assert "every=true" in r.json()["detail"]

    _with_live_jobs()
    monkeypatch.setattr(job_queue, "cancel", lambda ids: list(ids))
    r = client.post("/v1/job/cancel", json={"type": "judge_answers", "every": True})
    assert r.status_code == 200 and r.json()["cancelled"] == [11, 12]


def test_a_cancel_goes_through_the_queue_so_the_experiment_is_not_left_waiting(
    client, monkeypatch
):
    # the route flipped the status itself while `mark_failed_for_run` lives in the queue
    import job_queue

    seen = []
    _with_live_jobs()
    monkeypatch.setattr(job_queue, "cancel", lambda ids: seen.append(ids) or list(ids))

    assert client.post("/v1/job/cancel", json={"run_name": "arm"}).status_code == 200
    assert seen == [[11, 12]], "the route must delegate rather than write the status itself"


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


def _door_that_queues(monkeypatch, *, rows=0, jobs=()):
    import api.v1.eval as eval_mod

    async def _rows(session, run_name):
        return rows

    async def _named(session, run_name):
        return list(jobs)

    async def _refresh(session, obj):
        return obj

    monkeypatch.setattr(eval_mod, "_rows_of", _rows)
    monkeypatch.setattr(eval_mod, "_eval_runs_named", _named)
    monkeypatch.setattr(eval_mod.job_queue, "add_job", lambda s, t, o: _queued_job(t, o))
    monkeypatch.setattr(eval_mod, "commit_and_refresh", _refresh)


def test_a_taken_run_name_is_refused_unless_the_run_is_resumed(client, monkeypatch):
    # a second run under the same name wrote its rows beside the first, one question twice
    from types import SimpleNamespace

    from models import JobStatus

    _door_that_queues(monkeypatch, rows=3)
    r = client.post("/v1/eval/run", json={"run_name": "r", "set_name": "s"})
    assert r.status_code == 409 and "pass resume" in r.json()["detail"]
    # stopped on its first call, as a broker's 502 did: no row, and the name is taken all the same
    _door_that_queues(monkeypatch, jobs=[SimpleNamespace(status=JobStatus.done, options={"run_name": "r"})])
    assert client.post("/v1/eval/run", json={"run_name": "r", "set_name": "s"}).status_code == 409


def test_a_resumed_run_changes_nothing_and_runs_on_the_stopped_jobs_options(client, monkeypatch):
    from types import SimpleNamespace

    from models import JobStatus

    stopped = SimpleNamespace(status=JobStatus.error, options={
        "run_name": "r", "set_name": "s", "model": "MiniMaxAI/MiniMax-M2.7", "pipeline": "single_shot",
        "attempts": 3, "resume": False,
    })
    _door_that_queues(monkeypatch, rows=3, jobs=[stopped])
    edited = client.post("/v1/eval/run", json={"run_name": "r", "resume": True, "model": "other"})
    assert edited.status_code == 422 and "model" in edited.json()["detail"]
    resumed = client.post("/v1/eval/run", json={"run_name": "r", "resume": True})
    assert resumed.status_code == 200
    assert resumed.json()["options"] == {
        "run_name": "r", "set_name": "s", "model": "MiniMaxAI/MiniMax-M2.7", "pipeline": "single_shot", "resume": True,
    }


def test_a_run_is_resumed_only_when_it_exists_and_has_stopped(client, monkeypatch):
    from types import SimpleNamespace

    from models import JobStatus

    _door_that_queues(monkeypatch)
    assert client.post("/v1/eval/run", json={"run_name": "r", "resume": True}).status_code == 404
    running = SimpleNamespace(status=JobStatus.running, options={"run_name": "r", "set_name": "s"})
    _door_that_queues(monkeypatch, jobs=[running])
    assert client.post("/v1/eval/run", json={"run_name": "r", "resume": True}).status_code == 409


def test_the_door_refuses_question_ids_that_repeat_or_are_not_in_the_stand(client, monkeypatch):
    import api.v1.eval as eval_mod

    async def _found(session, ids):
        return {34, 35}

    _door_that_queues(monkeypatch)
    monkeypatch.setattr(eval_mod, "_question_ids_in", _found)
    missing = client.post("/v1/eval/run", json={"question_ids": [34, 35, 99]})
    assert missing.status_code == 422 and "1 of 3 question ids are not in the stand: [99]" in missing.json()["detail"]
    repeated = client.post("/v1/eval/run", json={"question_ids": [34, 34]})
    assert repeated.status_code == 422 and "question ids repeat: [34]" in str(repeated.json()["detail"])
    assert client.post("/v1/eval/run", json={"question_ids": [34, 35]}).status_code == 200


def test_a_taken_run_name_is_refused_at_every_door_that_queues_a_run(client, monkeypatch):
    # only `/v1/eval/run` checked, and the experiment door's retry wrote every question twice
    _door_that_queues(monkeypatch, rows=3)
    job = client.post("/v1/job", json={"type": "eval_run", "options": {"run_name": "r", "set_name": "s"}})
    assert job.status_code == 409 and "pass resume" in job.json()["detail"]
    sweep = client.post("/v1/eval/experiment", json={"run_name": "r", "set_name": "s", "param": "k", "values": [5]})
    assert sweep.status_code == 409


def test_the_chat_refuses_options_it_would_not_read(client):
    # a model, a temperature and tags were accepted and never read
    assert client.post("/v1/chat/question", json={"text": "x", "options": {"model": "m"}}).status_code == 422
    assert client.post("/v1/chat/question", json={"text": "x", "filter": {"tags": ["t"]}}).status_code == 422
