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
