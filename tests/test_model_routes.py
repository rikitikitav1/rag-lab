"""The happy path of `POST /v1/model`, which no test walked when `engine_id` became mandatory."""

import pytest
from models.registry import Engine, EngineKind, Model, Placement, Status


def _engine(engine_id=1, name="ollama", kind=EngineKind.ollama):
    return Engine(id=engine_id, name=name, kind=kind, env_prefix=name.upper(),
                  placement=Placement.gpu)


class FakeAsyncSession:
    def __init__(self, engines=(), taken=False, model=None, listed=()):
        self.engines = list(engines)
        self.taken = taken
        self.model = model
        self.listed = list(listed)
        self.added = []
        self.deleted = []

    async def execute(self, _stmt):
        rows = self.listed

        class _R:
            def all(self):
                return rows

        return _R()

    async def scalars(self, stmt):
        # a rename would quietly turn this into "zero engines", so an unknown shape is loud
        assert "engines" in str(stmt), f"the stub does not know this statement: {stmt}"
        return [e.id for e in self.engines]

    async def scalar(self, stmt):
        text = str(stmt).lower()
        if "exists" in text:
            return self.taken
        if "engines.name" in text:
            return self.engines[0].name if self.engines else None
        if "engines" in text:
            return self.engines[0] if self.engines else None
        return None

    async def get(self, model, ident):
        if model is Engine:
            return next((e for e in self.engines if e.id == ident), None)
        return self.model

    def add(self, obj):
        self.added.append(obj)
        if isinstance(obj, Model) and obj.id is None:
            obj.id = 42
            obj.status = Status.available

    async def delete(self, obj):
        self.deleted.append(obj)

    async def flush(self):
        for obj in self.added:
            if getattr(obj, "id", 0) is None:
                obj.id = 7

    async def commit(self):
        pass

    async def refresh(self, _obj):
        pass


@pytest.fixture
def door(monkeypatch):
    import bootstrap

    monkeypatch.setattr(bootstrap, "bootstrap_models", lambda: None)

    import server
    from fastapi.testclient import TestClient
    from orm.async_db import get_session

    def with_session(session):
        async def _yield():
            yield session

        server.app.dependency_overrides[get_session] = _yield
        return TestClient(server.app)

    yield with_session
    server.app.dependency_overrides.clear()


def test_registering_a_model_names_the_engine_it_lands_on(door):
    session = FakeAsyncSession(engines=[_engine()])
    with door(session) as client:
        r = client.post("/v1/model", json={"name": "qwen2.5:7b"})

    assert r.status_code == 200, r.text
    assert r.json()["engine"] == "ollama"
    added = [o for o in session.added if isinstance(o, Model)]
    # the column is NOT NULL, and a row built without it broke the door rather than the tests
    assert added[0].engine_id == 1


def test_the_pull_job_carries_the_engine_and_not_only_the_name(door):
    from job_queue import Job

    session = FakeAsyncSession(engines=[_engine()])
    with door(session) as client:
        client.post("/v1/model", json={"name": "qwen2.5:7b"})

    job = next(o for o in session.added if isinstance(o, Job))
    assert job.options == {"name": "qwen2.5:7b", "engine_id": 1}


def test_two_engines_and_no_named_one_is_refused_rather_than_guessed(door):
    session = FakeAsyncSession(engines=[_engine(), _engine(2, "vllm", EngineKind.vllm)])
    with door(session) as client:
        r = client.post("/v1/model", json={"name": "qwen2.5:7b"})

    assert r.status_code == 400
    assert "name the engine" in r.json()["detail"]


def test_an_unknown_engine_is_a_404(door):
    session = FakeAsyncSession(engines=[_engine()])
    with door(session) as client:
        r = client.post("/v1/model", json={"name": "qwen2.5:7b", "engine_id": 77})

    assert r.status_code == 404


def test_the_same_name_twice_on_one_engine_is_a_409(door):
    session = FakeAsyncSession(engines=[_engine()], taken=True)
    with door(session) as client:
        r = client.post("/v1/model", json={"name": "qwen2.5:7b"})

    assert r.status_code == 409


def test_a_load_is_queued_as_a_card_handover_and_not_run_by_the_api(door, monkeypatch):
    import job_queue

    asked = []
    from api.v1 import llm_model

    monkeypatch.setattr(job_queue, "enqueue", lambda t, o, **kw: asked.append((t, o)) or 77)
    serves = [False]
    monkeypatch.setattr(llm_model, "_serves", lambda engine, name: serves[0])
    session = FakeAsyncSession(
        engines=[_engine(2, "vllm", EngineKind.vllm)],
        model=Model(id=5, name="Qwen/Q", engine_id=2, status=Status.ready),
    )
    with door(session) as client:
        wrong = client.post("/v1/model/5/load")
        serves[0] = True
        r = client.post("/v1/model/5/load")

    assert wrong.status_code == 422, "a vLLM serving another model must not be woken with a 202"
    assert r.status_code == 202
    assert r.json() == {"job_id": 77, "engine": "vllm", "model": "Qwen/Q"}
    assert asked == [("hand_card", {"engine_id": 2, "model": "Qwen/Q"})]


def test_loading_on_a_cloud_engine_is_not_applicable(door):
    session = FakeAsyncSession(
        engines=[_engine(2, "cloud", EngineKind.openai_compatible)],
        model=Model(id=5, name="gpt", engine_id=2, status=Status.ready),
    )
    with door(session) as client:
        r = client.post("/v1/model/5/load")

    assert r.status_code == 501


def test_the_list_names_the_engine_of_every_row(door):
    session = FakeAsyncSession(listed=[
        (Model(id=1, name="qwen2.5:7b", engine_id=1, status=Status.ready,
               quant="Q4_K_M"), "ollama", "qwen2-7.6b"),
        (Model(id=2, name="qwen2.5:7b", engine_id=2, status=Status.ready), "vllm", None),
    ])
    with door(session) as client:
        r = client.get("/v1/model")

    assert r.status_code == 200, r.text
    # one name, two rows: without the engine the list cannot tell them apart
    assert [row["engine"] for row in r.json()] == ["ollama", "vllm"]
    # what the pull read off the server, so a reader checks the pair without opening the database
    assert [row["quant"] for row in r.json()] == ["Q4_K_M", None]
    assert [row["weights"] for row in r.json()] == ["qwen2-7.6b", None]


def test_deleting_a_model_still_removes_the_weights_after_the_row_is_gone(monkeypatch):
    # the door deletes the row and then queues the job, so a job needing the row never runs
    import engines
    from job_handlers import model_ops
    from models.registry import EngineKind, Placement

    spec = engines.EngineSpec(1, "ollama", EngineKind.ollama, "OLLAMA", Placement.gpu)
    removed = []
    monkeypatch.setattr(model_ops.engines, "spec_of_id", lambda _id: spec)
    monkeypatch.setattr(model_ops.ollama, "delete_model", lambda n, s: removed.append(n))

    class _Session:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def scalars(self, _stmt):
            class _R:
                def first(self):
                    return None

            return _R()

        def execute(self, _stmt):
            class _R:
                def all(self):
                    return []

            return _R()

    monkeypatch.setattr(model_ops, "Session", _Session)
    model_ops.delete_llm_model({"name": "qwen2.5:7b", "engine_id": 1})

    assert removed == ["qwen2.5:7b"], "the row was already gone, the weights still must go"


def test_weights_a_role_still_uses_under_another_name_are_not_destroyed(monkeypatch):
    # `bge-m3` and `bge-m3:latest` are two rows and one artifact, and two engines can share a volume
    import pytest
    from job_handlers import model_ops

    class _Session:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def execute(self, _stmt):
            class _R:
                def all(self):
                    return [("bge-m3", "ollama", "embedding")]

            return _R()

    monkeypatch.setattr(model_ops, "Session", _Session)
    with pytest.raises(ValueError, match="same artifact"):
        model_ops._refuse_if_another_row_needs_these_weights("bge-m3:latest")


def test_a_delete_for_an_engine_that_vanished_refuses_instead_of_picking_another(monkeypatch):
    # falling back to a name search deletes the same name on whatever engine still has it
    import pytest
    from job_handlers import model_ops

    monkeypatch.setattr(model_ops.engines, "spec_of_id", lambda _id: None)
    with pytest.raises(ValueError, match="engine 999 is not registered"):
        model_ops._engine_for_delete("gemma2:9b", 999)


def test_an_engine_moves_off_the_card_but_never_renames_or_repoints(door, monkeypatch):
    from api.v1 import engine as engine_door

    monkeypatch.setenv("VLLM_BASE_URL", "http://vllm:8000")
    running = [True]
    monkeypatch.setattr(engine_door, "_running", lambda spec: running[0])
    row = _engine(2, "vllm", EngineKind.vllm)
    session = FakeAsyncSession(engines=[row])
    with door(session) as client:
        # a live process stays on the card whatever its row says, and the card logic trusts the row
        refused = client.patch("/v1/engine/2", json={"placement": "cpu"})
        same = client.patch("/v1/engine/2", json={"placement": "gpu"})
        running[0] = False
        moved = client.patch("/v1/engine/2", json={"placement": "cpu"})
        renamed = client.patch("/v1/engine/2", json={"name": "other"})
        repointed = client.patch("/v1/engine/2", json={"placement": "gpu", "env_prefix": "OTHER"})

    assert refused.status_code == 409 and "stop it" in refused.json()["detail"]
    assert same.status_code == 200, "restating the placement it has is no move"
    assert moved.status_code == 200 and moved.json()["placement"] == "cpu"
    # a stamp names its engine by name and address: changing either would move the ruler silently
    assert renamed.status_code == 422 and repointed.status_code == 422
    assert row.name == "vllm" and row.env_prefix == "VLLM"


def test_a_vllm_that_does_not_serve_the_name_is_refused_before_the_row(door, monkeypatch):
    from api.v1 import llm_model

    serves = [False]
    monkeypatch.setattr(llm_model, "_serves", lambda engine, name: serves[0])
    monkeypatch.setattr(llm_model.vllm, "artifact_of",
                        lambda name: {"quant": "F16", "size_bytes": 2_000_000_000})
    session = FakeAsyncSession(engines=[_engine(2, "vllm", EngineKind.vllm)])
    with door(session) as client:
        refused = client.post("/v1/model", json={"name": "Qwen/Q", "engine_id": 2})
        serves[0] = True
        taken = client.post("/v1/model", json={"name": "Qwen/Q", "engine_id": 2})

    assert refused.status_code == 400 and "does not serve" in refused.json()["detail"]
    rows = [m for m in session.added if isinstance(m, Model)]
    assert len(rows) == 1, "the refusal comes before the row, not after it"
    assert taken.status_code == 200
    # 11.09: a row the door made carried no quant, while the bootstrap's rows did
    assert (rows[0].quant, rows[0].size_bytes) == ("F16", 2_000_000_000)


def test_a_model_takes_a_quant_by_hand_and_a_weights_key_only_the_hub_knows(door, monkeypatch):
    from api.v1 import llm_model

    known = [False]
    monkeypatch.setattr(llm_model, "_repo_exists", lambda repo: known[0])
    session = FakeAsyncSession(
        engines=[_engine(2, "vllm", EngineKind.vllm)],
        model=Model(id=5, name="Qwen/Q", engine_id=2, status=Status.ready),
    )
    with door(session) as client:
        empty = client.patch("/v1/model/5", json={})
        renamed = client.patch("/v1/model/5", json={"name": "other"})
        unknown = client.patch("/v1/model/5", json={"weights": "Qwen/Nowhere"})
        known[0] = None
        silent = client.patch("/v1/model/5", json={"weights": "Qwen/Nowhere"})
        fixed = client.patch("/v1/model/5", json={"quant": "AWQ"})

    assert empty.status_code == 422 and renamed.status_code == 422
    # a key the hub has never heard of is free text again, and a silent hub is not a no
    assert unknown.status_code == 422 and silent.status_code == 502
    assert fixed.status_code == 200 and fixed.json()["quant"] == "AWQ"

    known[0] = True
    with door(session) as client:
        keyed = client.patch("/v1/model/5", json={"weights": "Qwen/Qwen2.5-7B-Instruct"})
    assert keyed.status_code == 200 and keyed.json()["weights"] == "Qwen/Qwen2.5-7B-Instruct"
    assert session.model.weights_id == 7, "the row points at the weights the hub vouched for"


def test_an_engine_is_running_unless_its_connection_is_refused(monkeypatch):
    import requests
    from api.v1 import engine as engine_door

    spec = engine_door.engines.EngineSpec(2, "vllm", EngineKind.vllm, "VLLM", Placement.gpu)
    monkeypatch.setenv("VLLM_BASE_URL", "http://vllm:8000")
    replies = {"x": None}

    def get(url, timeout):
        if replies["x"]:
            raise replies["x"]

    monkeypatch.setattr(requests, "get", get)
    assert engine_door._running(spec) is True
    replies["x"] = requests.ConnectionError("refused")
    assert engine_door._running(spec) is False
    replies["x"] = requests.Timeout("3 s")
    assert engine_door._running(spec) is True, "a silent server may still hold the card"
    monkeypatch.delenv("VLLM_BASE_URL")
    assert engine_door._running(spec) is False, "an engine with no address runs nowhere"
