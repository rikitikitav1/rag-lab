from real_db import pytestmark  # noqa: F401
from sqlalchemy import text


def _question(c, set_name, drawn_from=None):
    return c.execute(
        text(
            "INSERT INTO questions (original_text, text_hash, language, set_name, source_question_id)"
            " VALUES ('q', md5(random()::text), 'en', :s, :o) RETURNING id"
        ),
        {"s": set_name, "o": drawn_from},
    ).scalar()


# a set's own paraphrase chain goes with it; an answer log or another set drawn from it is counted, not cascaded
def test_a_set_is_counted_and_removed_with_its_own_chain(db, monkeypatch):
    import db as stand_db

    monkeypatch.setattr(stand_db, "engine", db)
    with db.connect() as c:
        first = _question(c, "smoke")
        _question(c, "smoke", drawn_from=first)
        kept = _question(c, "kept")
        _question(c, "kept_paraphrased", drawn_from=kept)
        c.execute(
            text("INSERT INTO question_logs (run_name, question_id, answered) VALUES ('r', :q, true)"), {"q": kept}
        )
        c.commit()

    assert stand_db.question_set_holds("smoke") == {"questions": 2, "answered": 0, "drawn_from": 0}
    assert stand_db.question_set_holds("kept") == {"questions": 1, "answered": 1, "drawn_from": 1}
    assert stand_db.remove_question_set("smoke") == 2
    assert stand_db.question_set_holds("smoke")["questions"] == 0
    assert stand_db.question_set_holds("kept")["questions"] == 1
