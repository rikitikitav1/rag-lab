from types import SimpleNamespace

import pytest
from models.registry import Role
from use_cases import stand_health


def test_the_card_is_read_from_the_driver_and_no_cuda_context_is_left_behind(monkeypatch):
    # a torch read opened a context that held 128 MiB, and bge-m3 then spilled beside llama
    import sys

    import gpu

    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setattr(gpu.subprocess, "run",
                        lambda *a, **kw: SimpleNamespace(returncode=0, stdout="7426, 8188\n"))
    assert gpu.memory_mb() == (7426, 8188)
    monkeypatch.setattr(gpu.subprocess, "run",
                        lambda *a, **kw: SimpleNamespace(returncode=9, stdout=""))
    assert gpu.memory_mb() is None, "no driver, no number"


def test_the_roles_block_shows_both_sides_because_they_drift_in_silence(monkeypatch):
    # the file declares and the database serves, and bootstrap leaves an assigned role alone
    class _Session:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def execute(self, _stmt):
            return SimpleNamespace(
                all=lambda: [(Role.generation, "gemma3:4b", "ollama"),
                             (Role.judging, "qwen2.5:7b", "ollama")]
            )

    monkeypatch.setattr(stand_health, "Session", _Session)
    monkeypatch.setattr(stand_health.engines, "seeded_ollama",
                        lambda: SimpleNamespace(name="ollama"))
    monkeypatch.setattr(
        stand_health.config.settings.llm, "roles",
        {"generation": SimpleNamespace(model="llama3.1:8b", engine=None),
         "judging": SimpleNamespace(model="qwen2.5:7b", engine="ollama")},
    )
    out = stand_health.roles()

    assert out["served"]["generation"] == "gemma3:4b@ollama"
    assert out["declared"]["generation"] == "llama3.1:8b@ollama"
    assert out["drift"] == ["generation"], "the two sides are shown and the difference is named"


def test_a_card_that_cannot_be_read_is_reported_rather_than_raised(monkeypatch):
    # read while a run is going: a probe that raises turns the one window into it into an error
    def no_driver():
        raise RuntimeError("no driver")

    monkeypatch.setattr(stand_health.gpu, "memory_mb", no_driver)
    out = stand_health.card()

    # the name of the failure, not its text: the route answers without a key
    assert out["cuda"] is None and out["error"] == "RuntimeError"


def test_a_role_moved_to_another_engine_under_the_same_name_is_drift(monkeypatch):
    class _Session:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def execute(self, _stmt):
            return SimpleNamespace(all=lambda: [(Role.embedding, "bge-m3", "ollama-cpu")])

    monkeypatch.setattr(stand_health, "Session", _Session)
    monkeypatch.setattr(stand_health.engines, "seeded_ollama",
                        lambda: SimpleNamespace(name="ollama"))
    monkeypatch.setattr(stand_health.config.settings.llm, "roles",
                        {"embedding": SimpleNamespace(model="bge-m3", engine=None)})
    out = stand_health.roles()
    assert out["drift"] == ["embedding"], out
    monkeypatch.setattr(stand_health.config.settings.llm, "roles",
                        {"embedding": SimpleNamespace(model="bge-m3", engine="ollama-cpu")})
    assert stand_health.roles()["drift"] == []


def test_a_role_the_config_declares_and_nothing_serves_is_drift_too(monkeypatch):
    # the drift was read by walking the served side, so a declared role never appeared
    class _Session:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def execute(self, _stmt):
            return SimpleNamespace(all=lambda: [(Role.judging, "qwen2.5:7b", "ollama")])

    monkeypatch.setattr(stand_health, "Session", _Session)
    monkeypatch.setattr(stand_health.engines, "seeded_ollama",
                        lambda: SimpleNamespace(name="ollama"))
    monkeypatch.setattr(
        stand_health.config.settings.llm, "roles",
        {"generation": SimpleNamespace(model="llama3.1:8b", engine=None),
         "judging": SimpleNamespace(model="qwen2.5:7b", engine="ollama")},
    )

    assert stand_health.roles()["drift"] == ["generation"]


def _engine(name, kind, placement):
    import engines
    from models.registry import EngineKind, Placement

    return engines.EngineSpec(1, name, EngineKind(kind), name.upper().replace("-", "_"),
                              Placement(placement))


def test_the_window_is_read_from_the_generator_s_own_engine(monkeypatch):
    # `/api/ps` asked of a vLLM failed the whole stand read
    import engines

    vllm_spec = _engine("vllm", "vllm", "gpu")
    monkeypatch.setattr(stand_health.llm, "resolve",
                        lambda role: engines.Resolved("Qwen/Q", vllm_spec))
    monkeypatch.setattr(stand_health.vllm, "max_model_len", lambda spec, name: 8192)
    monkeypatch.setattr(stand_health.ollama, "window_model",
                        lambda *a, **kw: pytest.fail("ollama asked about a vLLM generator"))
    seen = stand_health.window()
    assert (seen["served"], seen["refuses_past_it"], seen["engine"]) == (8192, True, "vllm")


def test_only_ollama_spills_and_an_asleep_vllm_is_not_off_the_card(monkeypatch):
    import engines

    specs = {"generation": _engine("ollama", "ollama", "gpu"),
             "embedding": _engine("ollama-cpu", "ollama", "cpu"),
             "judging": _engine("vllm", "vllm", "gpu"),
             "paraphrasing": _engine("ollama", "ollama", "gpu"),
             "reranking": _engine("vllm-rerank", "vllm", "gpu")}
    monkeypatch.setattr(stand_health.llm, "resolve",
                        lambda role: engines.Resolved(role, specs[role]))
    on = {"generation": False, "embedding": False, "judging": False, "paraphrasing": None,
          "reranking": False}
    monkeypatch.setattr(stand_health.card_holder, "model_on_card", lambda spec, name: on[name])
    seen = stand_health.roles_on_card()
    assert [r for r, v in seen.items() if v["spilled"]] == ["generation"], seen


def test_a_role_whose_engine_does_not_answer_is_named(monkeypatch):
    # the judge's vLLM died of CUDA OOM, exited 0, and nothing on the stand said so
    import engines

    specs = {"generation": _engine("ollama", "ollama", "gpu"), "embedding": _engine("ollama", "ollama", "gpu"),
             "judging": _engine("vllm", "vllm", "gpu"), "paraphrasing": _engine("ollama", "ollama", "gpu"),
             "reranking": _engine("vllm-rerank", "vllm", "gpu")}
    monkeypatch.setattr(stand_health.llm, "resolve", lambda role: engines.Resolved(role, specs[role]))
    alive = {"ollama": True, "vllm": False, "vllm-rerank": None}
    monkeypatch.setattr(stand_health, "_answers", lambda spec: alive[spec.name])
    monkeypatch.setattr(stand_health.config.settings.rerank, "enabled", False)
    assert stand_health.roles_down() == [
        "judging: vllm does not answer; a host without a card runs `scripts/up.sh --cpu`"
        " (docs/stand_modes.md)"
    ], "an unused reranker is off, and an engine of the card that is down points to the mode"
    monkeypatch.setattr(stand_health.config.settings.rerank, "enabled", True)
    assert stand_health.roles_down()[-1].startswith("reranking: vllm-rerank does not answer;")


def test_a_generator_seated_unasked_is_named_once_its_probe_says_no(monkeypatch):
    # a boot seats the generator without a probe, so a later `False` has to be shown somewhere
    import engines

    vllm = _engine("vllm", "vllm", "gpu")
    monkeypatch.setattr(stand_health.llm, "resolve", lambda role: engines.Resolved("Qwen/Q", vllm))
    monkeypatch.setattr(stand_health, "_answers", lambda spec: True)
    monkeypatch.setattr(stand_health.config.settings.rerank, "enabled", False)
    monkeypatch.setattr(stand_health.vllm, "started_at", lambda spec: "t0")
    probe = [None]
    monkeypatch.setattr(stand_health.vllm, "known_probe", lambda spec, model, started: probe[0])
    assert stand_health.roles_down() == [], "not yet asked is not a verdict"
    probe[0] = False
    assert stand_health.roles_down() == ["generation: Qwen/Q@vllm returns no tool calls"]


def test_readiness_names_the_dead_role_and_stays_up_for_the_chat(monkeypatch):
    import bootstrap

    monkeypatch.setattr(bootstrap, "bootstrap_models", lambda: None)
    import server
    from api import health
    from fastapi.testclient import TestClient
    from orm.async_db import get_session

    class _Session:
        async def execute(self, _stmt):
            return None

    async def _yield():
        yield _Session()

    down = [["judging: vllm does not answer"]]
    monkeypatch.setattr(health.stand_health, "roles_down", lambda: down[0])
    server.app.dependency_overrides[get_session] = _yield
    try:
        with TestClient(server.app) as client:
            got = client.get("/readiness")
            down[0] = []
            fine = client.get("/readiness")
    finally:
        server.app.dependency_overrides.clear()
    assert got.status_code == 200
    assert got.json()["status"] == "degraded" and got.json()["roles_down"] == ["judging: vllm does not answer"]
    # the card's ollama is absent on purpose without a card, so it is not probed on its own
    assert fine.json() == {"postgres": "ok", "status": "ok"}


def test_an_unseated_role_is_named_and_the_others_still_read(monkeypatch):
    # a clean machine without the reranker's weights failed the whole section
    import engines

    def resolve(role):
        if role == "reranking":
            raise engines.Unnamed("no model assigned to role reranking")
        return engines.Resolved(role, _engine("ollama", "ollama", "gpu"))

    monkeypatch.setattr(stand_health.llm, "resolve", resolve)
    monkeypatch.setattr(stand_health.card_holder, "model_on_card", lambda spec, name: True)
    seen = stand_health.roles_on_card()
    assert seen["reranking"]["model"] is None and seen["generation"]["on_card"] is True
    monkeypatch.setattr(stand_health, "_answers", lambda spec: True)
    monkeypatch.setattr(stand_health.config.settings.rerank, "enabled", True)
    assert stand_health.roles_down() == ["reranking: no model is seated"]


def test_an_unreadable_seeded_engine_is_compared_by_name_not_read_as_drift(monkeypatch):
    # the placeholder "the seeded ollama" read as drift on every role without an engine
    class _Session:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def execute(self, _stmt):
            return SimpleNamespace(all=lambda: [(Role.generation, "llama3.1:8b", "ollama")])

    def unreadable():
        raise RuntimeError("the engines table did not answer")

    monkeypatch.setattr(stand_health, "Session", _Session)
    monkeypatch.setattr(stand_health.engines, "seeded_ollama", unreadable)
    monkeypatch.setattr(stand_health.config.settings.llm, "roles",
                        {"generation": SimpleNamespace(model="llama3.1:8b", engine=None)})
    assert stand_health.roles()["drift"] == []
