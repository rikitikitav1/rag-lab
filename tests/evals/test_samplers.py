import config
import engines
import pytest
import samplers
from engines import drivers
from models.registry import EngineKind, Placement


@pytest.mark.parametrize("options", [
    {"temperature": "hot"}, {"temperature": 3}, {"temperature": True}, {"seed": -1}, {"seed": 1.5},
    {"max_tokens": 0}, {"max_tokens": "4096"}, {"top_k": 5},
])
def test_a_sampler_the_engine_would_choke_on_is_refused_before_it_is_stored(options):
    with pytest.raises(ValueError):
        samplers.check(options)


def test_the_config_roles_are_held_to_the_model_door_s_sampler_rule():
    # the model's options were checked at its door and the role's in the yaml were not
    with pytest.raises(ValueError, match="temperature"):
        config.RoleCfg(model="m", options={"temperature": "hot"})


def test_an_unloaded_ollama_model_is_guarded_by_the_window_it_will_load_with(monkeypatch):
    # the window was read only off a loaded model, so a patch on any other passed unchecked
    spec = engines.EngineSpec(1, "ollama", EngineKind.ollama, "OLLAMA", Placement.gpu)
    monkeypatch.setattr(drivers.ollama, "context_length", lambda model, spec=None: None)
    assert engines.window_or_configured(spec, "qwen2.5:7b") == config.settings.llm.context_length


@pytest.mark.parametrize("penalty", [0.9, 2.5, True, "1.1"])
def test_a_repetition_penalty_outside_one_to_two_is_refused(penalty):
    with pytest.raises(ValueError, match="repetition_penalty"):
        samplers.check({"repetition_penalty": penalty})
    assert samplers.check({"repetition_penalty": 1.05}) == {"repetition_penalty": 1.05}


def test_the_penalty_goes_to_vllm_in_the_body_and_is_refused_by_the_doors_that_drop_it(monkeypatch):
    import llm

    vllm = engines.EngineSpec(3, "vllm", EngineKind.vllm, "VLLM", Placement.gpu)
    local = engines.EngineSpec(1, "ollama", EngineKind.ollama, "OLLAMA", Placement.gpu)
    cloud = engines.EngineSpec(8, "gonka", EngineKind.openai_compatible, "GONKA", Placement.remote)
    wanted = {"temperature": 0, "repetition_penalty": 1.05}
    assert engines.translate(vllm, wanted).sent == wanted
    for spec in (local, cloud):
        assert engines.translate(spec, wanted).dropped == {"repetition_penalty": 1.05}
    monkeypatch.setattr(llm, "sampler", lambda role, picked: engines.Sampler(dict(wanted), {}))
    params = llm._params("judging", None, engines.Resolved("Qwen/Q", vllm))
    assert params == {"temperature": 0, "extra_body": {"repetition_penalty": 1.05}}, (
        "the OpenAI client knows no such field"
    )


def test_the_stand_shows_each_ollama_role_s_penalty_against_the_declared_one(monkeypatch):
    from use_cases import stand_health

    local = engines.EngineSpec(1, "ollama", EngineKind.ollama, "OLLAMA", Placement.gpu)
    monkeypatch.setattr(config.settings.llm, "repetition_penalty", 1.1)
    monkeypatch.setattr(stand_health, "_roles", lambda: [
        ("generation", engines.Resolved("llama3.1:8b", local)), ("embedding", engines.Resolved("bge-m3", local)),
        ("ragas", engines.Resolved("qwen2.5:7b-w16384", local))])
    monkeypatch.setattr(stand_health.ollama, "repetition_penalty_served",
                        lambda model, spec=None: {"llama3.1:8b": 1.1, "qwen2.5:7b-w16384": "unknown"}[model])
    seen = stand_health.repetition_penalty()
    assert seen["served"] == {"llama3.1:8b@ollama": 1.1, "qwen2.5:7b-w16384@ollama": "unknown"}
    assert seen["drift"] == ["qwen2.5:7b-w16384@ollama"], "the embedder takes no penalty and is not read"
