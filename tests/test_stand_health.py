from types import SimpleNamespace

import pytest
from models.registry import Role
from use_cases import stand_health


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
    import builtins

    real = builtins.__import__

    def no_torch(name, *a, **kw):
        if name == "torch":
            raise RuntimeError("no driver")
        return real(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", no_torch)
    out = stand_health.card()

    assert out["cuda"] is None and "no driver" in out["error"]


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
    # 11.09: `/api/ps` asked of a vLLM failed the whole stand read
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
    # 11.09: the judge's vLLM died of CUDA OOM, exited 0, and nothing on the stand said so
    import engines

    specs = {"generation": _engine("ollama", "ollama", "gpu"), "embedding": _engine("ollama", "ollama", "gpu"),
             "judging": _engine("vllm", "vllm", "gpu"), "paraphrasing": _engine("ollama", "ollama", "gpu"),
             "reranking": _engine("vllm-rerank", "vllm", "gpu")}
    monkeypatch.setattr(stand_health.llm, "resolve", lambda role: engines.Resolved(role, specs[role]))
    alive = {"ollama": True, "vllm": False, "vllm-rerank": None}
    monkeypatch.setattr(stand_health, "_answers", lambda spec: alive[spec.name])
    monkeypatch.setattr(stand_health.config.settings.rerank, "enabled", False)
    assert stand_health.roles_down() == ["judging: vllm does not answer"], "an unused reranker is off"
    monkeypatch.setattr(stand_health.config.settings.rerank, "enabled", True)
    assert stand_health.roles_down()[-1] == "reranking: vllm-rerank does not answer"


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

    monkeypatch.setattr(health.ollama, "list_models", lambda: [])
    monkeypatch.setattr(health.stand_health, "roles_down", lambda: ["judging: vllm does not answer"])
    server.app.dependency_overrides[get_session] = _yield
    try:
        with TestClient(server.app) as client:
            got = client.get("/readiness")
    finally:
        server.app.dependency_overrides.clear()
    assert got.status_code == 200
    assert got.json()["status"] == "degraded" and got.json()["roles_down"] == ["judging: vllm does not answer"]
