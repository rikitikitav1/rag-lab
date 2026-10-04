# a reworded question keeps its accepted original's gold and status, into a new set only
import pytest
from errors import Final
from real_db import pytestmark  # noqa: F401
from sqlalchemy import text
from sqlalchemy.orm import sessionmaker


def test_variants_take_their_originals_gold_and_refuse_an_existing_set_or_an_unaccepted_original(db, monkeypatch):
    from use_cases import question_variants

    monkeypatch.setattr(question_variants, "Session", sessionmaker(bind=db))
    with db.connect() as c:
        c.execute(text("TRUNCATE questions CASCADE"))
        c.execute(text(
            "INSERT INTO questions (id, text_hash, original_text, set_name, language, gold, status) VALUES"
            " (1, 'a', 'How big is a PostgreSQL page?', 'base', 'en', '{\"file\": \"pg/a.md\"}', 'accepted'),"
            " (2, 'b', 'Some open question', 'base', 'en', '{\"file\": \"pg/b.md\"}', 'candidate')"))
        c.execute(text("SELECT setval('questions_id_seq', 2)"))
    done = question_variants.load("jargon", [{"source_question_id": 1, "original_text": "How big is a pg page?"}])
    assert done == {"set_name": "jargon", "questions_written": 1, "texts_already_held": 0}
    with db.connect() as c:
        row = c.execute(text("SELECT gold, status, source_question_id, language FROM questions"
                             " WHERE set_name = 'jargon'")).one()
    assert row == ({"file": "pg/a.md"}, "accepted", 1, "en")
    with pytest.raises(Final, match="exists"):
        question_variants.load("jargon", [{"source_question_id": 1, "original_text": "pg page size?"}])
    with pytest.raises(Final, match="not accepted"):
        question_variants.load("jargon2", [{"source_question_id": 2, "original_text": "x"}])
