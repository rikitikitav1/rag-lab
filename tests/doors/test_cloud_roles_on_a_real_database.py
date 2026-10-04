# a cloud role moves to another cloud model through MCP; a local role is refused, it holds the card
import pytest
from errors import Refusal
from real_db import pytestmark  # noqa: F401
from sqlalchemy import text
from sqlalchemy.orm import sessionmaker


@pytest.fixture
def stand(db, monkeypatch):
    from use_cases import cloud_roles

    monkeypatch.setattr(cloud_roles, "Session", sessionmaker(bind=db))
    monkeypatch.setattr(cloud_roles.engines, "served_models", lambda spec: {"deepseek-v4", "other"})
    monkeypatch.setattr(cloud_roles.model_acceptance, "refuse_unfit_model", lambda role, name, engine_id: None)
    with db.connect() as c:
        c.execute(text("TRUNCATE model_roles, models, engines RESTART IDENTITY CASCADE"))
        c.execute(text("INSERT INTO engines (id, name, kind, env_prefix, placement) VALUES"
                       " (1, 'neuraldeep', 'openai_compatible', 'NEURALDEEP', 'remote'),"
                       " (2, 'gonka', 'openai_compatible', 'GONKA', 'remote'),"
                       " (3, 'ollama', 'ollama', 'OLLAMA', 'gpu')"))
        c.execute(text("INSERT INTO models (id, name, engine_id, status) VALUES"
                       " (1, 'gemma', 1, 'ready'), (2, 'llama', 3, 'ready')"))
        c.execute(text("SELECT setval('models_id_seq', 2)"))
        c.execute(text("INSERT INTO model_roles (role, model_id) VALUES ('questioning', 1), ('generation', 2)"))
    return cloud_roles


def test_a_cloud_role_moves_to_a_cloud_model_and_a_local_one_is_refused(stand, db):
    done = stand.seat("questioning", "deepseek-v4", "gonka")
    assert done == {"role": "questioning", "model": "deepseek-v4", "engine": "gonka", "was": "gemma on neuraldeep"}
    with db.connect() as c:
        seated = c.execute(text("SELECT m.name FROM model_roles r JOIN models m ON m.id = r.model_id"
                                " WHERE r.role = 'questioning'")).scalar()
    assert seated == "deepseek-v4"
    with pytest.raises(Refusal, match="local roles are changed by a person"):
        stand.seat("generation", "deepseek-v4", "gonka")
    with pytest.raises(Refusal, match="not a cloud broker"):
        stand.seat("questioning", "llama", "ollama")
    with pytest.raises(Refusal, match="does not serve"):
        stand.seat("questioning", "nope", "gonka")
