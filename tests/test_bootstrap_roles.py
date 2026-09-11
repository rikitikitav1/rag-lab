from conftest import stub_engine


def test_bootstrap_seats_a_role_through_the_same_gate_the_route_uses(monkeypatch):
    # an empty database is the usual way roles are set, so the likeliest path to a unfit model
    from types import SimpleNamespace

    import bootstrap

    seated, refused = [], []

    class _Session:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def scalars(self, _stmt):
            return SimpleNamespace(all=list)

        def scalar(self, _stmt):
            return SimpleNamespace(id=1)

        def add(self, obj):
            seated.append(obj.role)

        def commit(self):
            pass

    monkeypatch.setattr(bootstrap, "Session", _Session)
    monkeypatch.setattr(
        bootstrap.config.settings.llm, "roles",
        {"generation": SimpleNamespace(model="gemma3:4b", engine=None)},
    )
    monkeypatch.setattr(
        bootstrap.model_acceptance, "refuse_unfit_model",
        lambda role, name, engine_id: refused.append(name) or (_ for _ in ()).throw(
            ValueError("no tools")),
    )

    bootstrap._ensure_roles(stub_engine())

    assert refused == ["gemma3:4b"]
    assert seated == [], "a model the gate refuses is not seated by the back door"


def test_a_role_that_names_its_engine_is_looked_up_on_that_engine(monkeypatch):
    # while one engine held every model a bare name was enough; two engines make it a silent pick
    from types import SimpleNamespace

    import bootstrap
    import engines
    from conftest import stub_engine

    seen = []

    class _Session:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def scalars(self, _stmt):
            return SimpleNamespace(all=list)

        def scalar(self, stmt):
            seen.append(stmt.compile().params)
            return None

        def add(self, _obj):
            pass

        def commit(self):
            pass

    named = engines.EngineSpec(7, "vllm", stub_engine().kind, "VLLM", stub_engine().placement)
    monkeypatch.setattr(bootstrap, "Session", _Session)
    monkeypatch.setattr(engines, "spec_of_name", lambda name: named if name == "vllm" else None)
    monkeypatch.setattr(
        bootstrap.config.settings.llm, "roles",
        {"judging": SimpleNamespace(model="qwen", engine="vllm")},
    )

    bootstrap._ensure_roles(stub_engine())

    assert 7 in seen[0].values(), f"the seeded engine was used instead: {seen}"


def test_a_role_naming_an_engine_that_does_not_exist_is_not_seated_on_the_seeded_one(monkeypatch):
    from types import SimpleNamespace

    import bootstrap
    import engines
    from conftest import stub_engine

    monkeypatch.setattr(engines, "spec_of_name", lambda _name: None)
    picked = bootstrap._engine_of_role(
        "judging", SimpleNamespace(model="qwen", engine="typo"), stub_engine()
    )
    assert picked is None


def test_a_model_of_a_role_on_another_engine_is_not_pulled_through_ollama(monkeypatch):
    from types import SimpleNamespace

    import config

    monkeypatch.setattr(
        config.settings.llm, "roles",
        {"generation": SimpleNamespace(model="llama3.1:8b", engine=None),
         "judging": SimpleNamespace(model="Qwen/Qwen2.5-7B-Instruct-AWQ", engine="vllm")},
    )
    assert config.settings.llm.pull_models == ["llama3.1:8b"]
