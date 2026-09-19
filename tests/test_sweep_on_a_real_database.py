# which rows the judge takes, asked of the database the sweep actually queries

from real_db import pytestmark  # noqa: F401
from sqlalchemy import text
from sqlalchemy.orm import Session


def _row(c, run_name, judge_wanted):
    question = c.execute(text(
        "INSERT INTO questions (original_text, text_hash, reference_answer, language, set_name)"
        " VALUES ('q', md5(random()::text), 'a', 'en', 's') RETURNING id"
    )).scalar()
    return c.execute(text(
        "INSERT INTO question_logs (run_name, question_id, answered, answer, context, judge_wanted)"
        " VALUES (:r, :q, true, 'said', 'ctx', :w) RETURNING id"
    ), {"r": run_name, "q": question, "w": judge_wanted}).scalar()


def test_the_sweep_leaves_alone_what_a_run_said_no_judge_to(db):
    # the sweep judges every unjudged row there is, and a canary run's rows are not for the judge
    from job_handlers import judging

    with db.connect() as c:
        judged = _row(c, "wants_a_verdict", True)
        quiet = _row(c, "no_judge", False)
        c.commit()

    with Session(db) as session:
        swept = judging._target_log_ids(session, {})
        assert judged in swept
        assert quiet not in swept, "a row nobody asked a verdict about must not cost one"

        by_name = judging._target_log_ids(session, {"run_name": "no_judge"})
        assert by_name == [quiet], "named, a run is judged whatever it said when it ran"
