from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _layer(tmp_path, text: str) -> str:
    path = tmp_path / "layer.yaml"
    path.write_text(text)
    return str(path)


def test_a_layer_replaces_the_whole_role_table():
    # merged, a judge without `engine` would inherit `vllm` from the main file and quietly break
    import config

    loaded = config._load(str(ROOT / "config.yaml"), str(ROOT / "config.cpu.yaml"))
    roles = loaded.llm.roles
    assert set(roles) == {"embedding", "generation", "judging", "paraphrasing"}
    assert {cfg.engine for cfg in roles.values()} == {"ollama-cpu"}
    assert "reranking" not in roles, "ollama scores no pairs, so the layer seats no reranker"


def test_a_layer_without_a_judge_or_with_more_than_roles_is_refused(tmp_path):
    import config

    base = str(ROOT / "config.yaml")
    no_judge = _layer(tmp_path, "llm:\n  roles:\n    generation: {model: m}\n    embedding: {model: e}\n")
    with pytest.raises(ValueError, match="drops"):
        config._load(base, no_judge)
    more = _layer(tmp_path, "llm:\n  roles: {}\n  context_length: 1\n")
    with pytest.raises(ValueError, match="nothing else"):
        config._load(base, more)


def test_the_seed_registers_the_processor_ollama():
    # on a clean database the no-card layout had no engine to seat its roles on
    import seed
    from models.registry import Placement

    added = []

    class _Session:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def scalar(self, _stmt):
            return False

        def add(self, row):
            added.append(row.name)

        def commit(self):
            pass

    original = seed.Session
    seed.Session = _Session
    try:
        seed.seed_engines()
    finally:
        seed.Session = original
    assert "ollama-cpu" in added
    assert seed.SEEDED_OLLAMA_CPU["placement"] is Placement.cpu


def test_a_role_on_an_ollama_the_pull_list_misses_gets_its_row(monkeypatch):
    # a role with `engine: ollama-cpu` was logged as absent and never seated
    import bootstrap
    import engines
    from models.registry import EngineKind, ModelRole, Placement

    cpu = engines.EngineSpec(5, "ollama-cpu", EngineKind.ollama, "OLLAMA_CPU", Placement.cpu)
    added = []

    class _Session:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def scalars(self, _stmt):
            return SimpleNamespace(all=list)

        def scalar(self, _stmt):
            return None

        def add(self, row):
            added.append(row)

        def flush(self):
            for row in added:
                if getattr(row, "id", 0) is None:
                    row.id = 42

        def commit(self):
            pass

    monkeypatch.setattr(bootstrap, "Session", _Session)
    monkeypatch.setattr(bootstrap.engines, "spec_of_name", lambda name: cpu)
    monkeypatch.setattr(bootstrap.config.settings.llm, "roles",
                        {"judging": SimpleNamespace(model="qwen2.5:7b", engine="ollama-cpu")})
    monkeypatch.setattr(bootstrap.model_acceptance, "refuse_unfit_model", lambda *a: None)

    bootstrap._ensure_roles(None)

    model = next(row for row in added if not isinstance(row, ModelRole))
    assert (model.name, model.engine_id) == ("qwen2.5:7b", 5)
    assert [row.model_id for row in added if isinstance(row, ModelRole)] == [42]


def test_a_card_engine_down_points_to_the_no_card_mode_and_a_processor_one_does_not(monkeypatch):
    import engines
    from models.registry import EngineKind, Placement
    from use_cases import card_wait, stand_health

    vllm = engines.EngineSpec(3, "vllm", EngineKind.vllm, "VLLM", Placement.gpu)
    cpu = engines.EngineSpec(5, "ollama-cpu", EngineKind.ollama, "OLLAMA_CPU", Placement.cpu)
    monkeypatch.setattr(stand_health, "_roles", lambda: [
        ("judging", engines.Resolved("Qwen/Q", vllm)),
        ("generation", engines.Resolved("llama3.1:8b", cpu)),
    ])
    monkeypatch.setattr(stand_health, "_answers", lambda spec: False)
    monkeypatch.setattr(card_wait, "reranker_needed", lambda **kw: False)

    judging, generation = stand_health.roles_down()
    assert "scripts/up.sh --cpu" in judging
    assert "scripts/up.sh" not in generation


def test_an_empty_card_is_said_in_words(monkeypatch):
    from use_cases import stand_health

    monkeypatch.setattr(stand_health.card_holder, "on_card", lambda: [])
    monkeypatch.setattr(stand_health.engines, "card_engines", lambda kind=None: [])
    monkeypatch.setattr(stand_health.engines, "registered", lambda: [])

    assert stand_health.engines_section()["summary"] == "no engine holds the card"
