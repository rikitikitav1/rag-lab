from types import SimpleNamespace

import pytest
from job_handlers import base, card, model_ops


def test_a_name_the_registry_does_not_have_fails_the_pull_once(monkeypatch):
    # three attempts on a manifest that does not exist, and the run waiting on it deferred for an hour
    found = SimpleNamespace(name="tester-nonexistent:1b", engine=SimpleNamespace(kind="ollama", name="ollama"))

    def pull(name, spec):
        raise RuntimeError("Ollama 500 on /api/pull: pull model manifest: file does not exist")

    monkeypatch.setattr(model_ops, "_pair", lambda name, engine_id: found)
    monkeypatch.setattr(model_ops, "_size_seen_before", lambda f: None)
    monkeypatch.setattr(model_ops.engines, "refuse_if_tight", lambda *a: None)
    monkeypatch.setattr(model_ops.engines, "driver", lambda kind: SimpleNamespace(weights_store=lambda: None, pull=pull))
    with pytest.raises(base.Final, match="not in the registry"):
        model_ops.pull_llm_model({"name": found.name})


def test_a_hand_over_to_an_engine_nobody_registered_is_final(monkeypatch):
    # retried three times ahead of the whole queue
    monkeypatch.setattr(card.engines, "spec_of_id", lambda engine_id: None)
    with pytest.raises(base.Final, match="not registered"):
        card.hand_card({"engine_id": 99})
