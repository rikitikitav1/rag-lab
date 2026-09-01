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
        {"generation": SimpleNamespace(model="gemma3:4b")},
    )
    monkeypatch.setattr(
        bootstrap.model_acceptance, "refuse_unfit_model",
        lambda role, name: refused.append(name) or (_ for _ in ()).throw(ValueError("no tools")),
    )

    bootstrap._ensure_roles()

    assert refused == ["gemma3:4b"]
    assert seated == [], "a model the gate refuses is not seated by the back door"
