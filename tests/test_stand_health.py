from types import SimpleNamespace

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
                all=lambda: [(Role.generation, "gemma3:4b"), (Role.judging, "qwen2.5:7b")]
            )

    monkeypatch.setattr(stand_health, "Session", _Session)
    monkeypatch.setattr(
        stand_health.config.settings.llm, "roles",
        {"generation": SimpleNamespace(model="llama3.1:8b"),
         "judging": SimpleNamespace(model="qwen2.5:7b")},
    )
    out = stand_health.roles()

    assert out["served"]["generation"] == "gemma3:4b"
    assert out["declared"]["generation"] == "llama3.1:8b"
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


def test_a_role_the_config_declares_and_nothing_serves_is_drift_too(monkeypatch):
    # the drift was read by walking the served side, so a declared role never appeared
    class _Session:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def execute(self, _stmt):
            return SimpleNamespace(all=lambda: [(Role.judging, "qwen2.5:7b")])

    monkeypatch.setattr(stand_health, "Session", _Session)
    monkeypatch.setattr(
        stand_health.config.settings.llm, "roles",
        {"generation": SimpleNamespace(model="llama3.1:8b"),
         "judging": SimpleNamespace(model="qwen2.5:7b")},
    )

    assert stand_health.roles()["drift"] == ["generation"]
