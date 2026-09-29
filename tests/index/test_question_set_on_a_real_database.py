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


# a log that took a question after the door's check refuses the removal instead of failing the server
def test_a_question_taken_after_the_check_refuses_the_removal(db, monkeypatch):
    import pytest
    from errors import Final

    import db as stand_db

    monkeypatch.setattr(stand_db, "engine", db)
    with db.connect() as c:
        taken = _question(c, "late")
        c.execute(
            text("INSERT INTO question_logs (run_name, question_id, answered) VALUES ('r', :q, true)"), {"q": taken}
        )
        c.commit()

    with pytest.raises(Final, match="questions of late were taken"):
        stand_db.remove_question_set("late")
    assert stand_db.question_set_holds("late")["questions"] == 1


# a generated pair is written whole: one of its questions already in the base leaves the pair out
def test_a_generated_pair_is_written_whole_or_not_at_all(db, monkeypatch):
    from job_handlers import questions
    from models.eval import text_hash
    from sqlalchemy.orm import sessionmaker

    monkeypatch.setattr(questions, "Session", sessionmaker(bind=db))
    gold = {"file": "a.md", "section": "A > B", "version": None}

    def row(text, pair):
        return {"original_text": text, "text_hash": text_hash(text), "language": "en", "set_name": "gen",
                "kind": "in_corpus", "gold": gold, "reference_answer": "x", "evidence": "y", "pair_id": pair}

    with db.connect() as c:
        taken = _question(c, "old")
        c.execute(
            text("UPDATE questions SET original_text = 'taken', text_hash = :h WHERE id = :id"),
            {"h": text_hash("taken"), "id": taken},
        )
        c.commit()

    written = questions.write_pairs(
        [row("fresh en", "p1"), row("fresh ru", "p1"), row("taken", "p2"), row("its ru", "p2")]
    )

    assert written == (2, 1)
    with db.connect() as c:
        got = c.execute(text("SELECT original_text, pair_id FROM questions WHERE set_name = 'gen' ORDER BY 1")).all()
    assert [tuple(r) for r in got] == [("fresh en", "p1"), ("fresh ru", "p1")]


# two pairs of one batch sharing a question are both dropped whole, never written as halves
def test_two_pairs_sharing_a_question_in_one_batch_are_dropped_whole(db, monkeypatch):
    from job_handlers import questions
    from models.eval import text_hash
    from sqlalchemy.orm import sessionmaker

    monkeypatch.setattr(questions, "Session", sessionmaker(bind=db))
    gold = {"file": "a.md", "section": "A > B", "version": None}

    def row(text, pair):
        return {"original_text": text, "text_hash": text_hash(text), "language": "en", "set_name": "batch",
                "kind": "in_corpus", "gold": gold, "reference_answer": "x", "evidence": "y", "pair_id": pair}

    rows = [row("shared en", "p1"), row("first ru", "p1"), row("shared en", "p2"), row("second ru", "p2")]
    assert questions.write_pairs(rows) == (2, 1)
    with db.connect() as c:
        got = c.execute(text("SELECT original_text FROM questions WHERE set_name = 'batch' ORDER BY 1")).scalars()
        assert list(got) == ["first ru", "shared en"]
