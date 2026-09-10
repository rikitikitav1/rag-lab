import pytest


@pytest.fixture
def client(monkeypatch):
    import bootstrap

    monkeypatch.setattr(bootstrap, "bootstrap_models", lambda: None)

    import server
    from fastapi.testclient import TestClient
    from orm.async_db import get_session

    async def _dummy_session():
        yield None

    server.app.dependency_overrides[get_session] = _dummy_session
    with TestClient(server.app) as c:
        yield c
    server.app.dependency_overrides.clear()


# a script is not importable: it lives outside `app` and three files loaded one by hand
@pytest.fixture(scope="session")
def script():
    import importlib.util
    from pathlib import Path

    def load(name: str):
        path = Path(__file__).resolve().parent.parent / "scripts" / f"{name}.py"
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    return load


@pytest.fixture(scope="session")
def preflight(script):
    return script("preflight_grid")


# the shared sync-Session stub; the hand-written ones in other files are not folded in yet
class FakeSession:
    def __init__(self, row=None):
        self.row = row
        self.added = []
        self.committed = 0

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def get(self, _model, _ident):
        return self.row

    def add(self, obj):
        self.added.append(obj)

    def commit(self):
        self.committed += 1

    def rollback(self):
        pass



# one engine on the stand, spelled once: six files were about to hold their own copy of it
def stub_engine():
    import engines
    from models.registry import EngineKind, Placement

    return engines.EngineSpec(1, "ollama", EngineKind.ollama, "OLLAMA", Placement.gpu)


def stub_engines(monkeypatch, module):
    import engines

    spec = stub_engine()
    monkeypatch.setattr(
        module.llm, "resolve_for", lambda role, model=None: engines.Resolved(model or "stub", spec)
    )
    monkeypatch.setattr(module.llm, "resolve", lambda role: engines.Resolved("stub", spec))


# requested only by the modules that carry `real_db.pytestmark`, so no postgres means a skip
@pytest.fixture(scope="module")
def db():
    import real_db

    yield from real_db.scratch()
