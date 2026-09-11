# the three constraints this arc is about, asserted by the database rather than by metadata
import pytest
from real_db import pytestmark  # noqa: F401
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError


def _engine(c, name):
    return c.execute(text(
        "INSERT INTO engines (name, kind, env_prefix, placement)"
        " VALUES (:n, 'ollama', :p, 'gpu') RETURNING id"
    ), {"n": name, "p": name.upper()}).scalar()


def test_a_model_without_an_engine_is_refused_by_the_database(db):
    # 745 tests were green while `POST /v1/model` built exactly this row and broke on it
    with db.connect() as c, pytest.raises(IntegrityError):
        c.execute(text("INSERT INTO models (name) VALUES ('orphan')"))


def test_the_same_name_lives_on_two_engines(db):
    # the whole point of the migration: one name, two engines, two rows
    with db.connect() as c:
        one, two = _engine(c, "one"), _engine(c, "two")
        c.execute(text("INSERT INTO models (name, engine_id) VALUES ('q:7b', :e)"), {"e": one})
        c.execute(text("INSERT INTO models (name, engine_id) VALUES ('q:7b', :e)"), {"e": two})
        assert c.execute(text("SELECT count(*) FROM models WHERE name = 'q:7b'")).scalar() == 2


def test_the_same_name_twice_on_one_engine_is_refused(db):
    with db.connect() as c:
        one = _engine(c, "dup")
        c.execute(text("INSERT INTO models (name, engine_id) VALUES ('m', :e)"), {"e": one})
        with pytest.raises(IntegrityError):
            c.execute(text("INSERT INTO models (name, engine_id) VALUES ('m', :e)"), {"e": one})


def test_an_engine_with_models_cannot_be_deleted(db):
    # a dangling name is allowed in run records, never in the registry
    with db.connect() as c:
        one = _engine(c, "held")
        c.execute(text("INSERT INTO models (name, engine_id) VALUES ('held-model', :e)"), {"e": one})
        with pytest.raises(IntegrityError):
            c.execute(text("DELETE FROM engines WHERE id = :e"), {"e": one})


def test_a_database_built_from_the_schema_carries_no_engine(db):
    # `seed.seed_engines` exists because of this: the row is data, and the dump holds none
    with db.connect() as c:
        assert c.execute(text("SELECT count(*) FROM engines WHERE name = 'ollama'")).scalar() == 0


def test_a_prefix_that_is_not_an_environment_name_is_refused(db):
    # it is interpolated into `os.getenv`, so an empty one would read `_BASE_URL`
    with db.connect() as c:
        for bad in ("", "lower", "WITH SPACE", "X" * 40):
            with pytest.raises(IntegrityError):
                c.execute(text(
                    "INSERT INTO engines (name, kind, env_prefix, placement)"
                    " VALUES ('e', 'ollama', :p, 'gpu')"
                ), {"p": bad})
