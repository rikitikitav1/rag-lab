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
