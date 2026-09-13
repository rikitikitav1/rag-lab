from pathlib import Path
from types import SimpleNamespace

import engines
import pytest
from models.registry import EngineKind, Placement

ROOT = Path(__file__).resolve().parent.parent


def _cloud(id: int, prefix: str) -> engines.EngineSpec:
    return engines.EngineSpec(id, prefix.lower(), EngineKind.openai_compatible, prefix, Placement.remote)


def test_two_clouds_side_by_side_keep_their_own_address_and_key(monkeypatch):
    # gonka is one of many; each cloud is a row with its own prefix, and nothing mixes the two
    from engines import core

    monkeypatch.setattr(core, "_clients", {})
    monkeypatch.setenv("GONKA_BASE_URL", "https://gonka.example/v1")
    monkeypatch.setenv("GONKA_API_KEY", "g-key")
    monkeypatch.setenv("OTHER_BASE_URL", "https://other.example")
    monkeypatch.setenv("OTHER_API_KEY", "o-key")
    gonka, other = _cloud(11, "GONKA"), _cloud(12, "OTHER")
    assert (core.base_url(gonka), core.base_url(other)) == ("https://gonka.example", "https://other.example")
    assert (core.api_key(gonka), core.api_key(other)) == ("g-key", "o-key")
    assert core.address_of(gonka) != core.address_of(other)
    first, second = core.client_for(gonka), core.client_for(other)
    assert str(first.base_url).rstrip("/") == "https://gonka.example/v1", "the page's /v1 is not doubled"
    assert str(second.base_url).rstrip("/") == "https://other.example/v1"
    assert first.api_key == "g-key" and second.api_key == "o-key"


def test_a_cloud_without_its_key_is_refused_by_name(monkeypatch):
    from engines import core

    monkeypatch.setenv("LONE_BASE_URL", "https://lone.example")
    monkeypatch.delenv("LONE_API_KEY", raising=False)
    with pytest.raises(engines.Unconfigured, match="lone: key is not configured"):
        core.api_key(_cloud(13, "LONE"))


def test_a_reply_without_usage_is_named_and_not_read_as_zero():
    # gonka sends usage; a broker that does not would otherwise count a pass as free
    import llm

    spec = _cloud(14, "GONKA")
    from errors import StandFault

    with pytest.raises(llm.NoUsage, match="gonka returned no token usage") as caught:
        llm._usage(SimpleNamespace(usage=None), spec)
    assert isinstance(caught.value, StandFault), "a loop forgives a RuntimeError as one failed row"
    counted = SimpleNamespace(prompt_tokens=7, completion_tokens=3)
    assert llm._usage(SimpleNamespace(usage=counted), spec) is counted


def test_a_new_cloud_reaches_the_containers_without_editing_compose():
    # the anchor listed each pair by name, so a cloud row had no address inside the container
    import yaml

    class Loader(yaml.SafeLoader):
        pass

    for tag in ("!reset", "!override"):
        Loader.add_constructor(tag, lambda loader, node: None)
    compose = yaml.load((ROOT / "docker-compose.yml").read_text(), Loader=Loader)
    for name in ("rag-lab", "bootstrap", "worker"):
        assert compose["services"][name]["env_file"] == [{"path": ".env", "required": False}], name
    assert not [k for k in compose["x-engine-env"] if k.startswith("CLOUD_")]


def test_a_cloud_client_retries_more_than_a_local_one(monkeypatch):
    # gonka held a call 92 s and answered 503, and the same call went through 20 s later
    from engines import core

    monkeypatch.setattr(core, "_clients", {})
    monkeypatch.setenv("GONKA_BASE_URL", "https://gonka.example/v1")
    monkeypatch.setenv("GONKA_API_KEY", "g-key")
    assert core.client_for(_cloud(15, "GONKA")).max_retries == core.CLOUD_RETRIES > 1
    local = engines.EngineSpec(2, "vllm", EngineKind.vllm, "VLLM", Placement.gpu)
    assert core._retries(local) == 1
