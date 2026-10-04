import config
import errors
import job_queue
import job_specs
from errors import Final
from orm.sync_db import engine
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError


# the jobs that read or write a set by name: every spec with a `set_name`, read off the specs so a new one counts
def set_jobs() -> tuple[str, ...]:
    return tuple(sorted(t for t, spec in job_specs.SPECS.items() if "set_name" in spec.model_fields))


def removal_refusal(set_name: str, holds: dict, named_in_config: bool, queued: int | None) -> str | None:
    if not holds["questions"]:
        return f"no question set named {set_name}"
    if named_in_config:
        return f"{set_name} is named in config/evals.yaml"
    if queued:
        return f"{set_name} has job {queued} queued or running"
    if holds["answered"]:
        return f"{holds['answered']} answer logs hold questions of {set_name}"
    if holds["drawn_from"]:
        return f"{holds['drawn_from']} questions of other sets are drawn from {set_name}"
    return None


def remove(set_name: str) -> dict:
    verdict = config.settings.verdict
    named = set_name in {*verdict.criterion_sets, *verdict.veto_sets}
    # a job may name its set by default or read it through another key, so any job that reads sets holds the door
    queued = next((j for t in (*set_jobs(), "embed_questions") if (j := job_queue.pending_of_type(t))), None)
    if refusal := removal_refusal(set_name, _holds(set_name), named, queued):
        raise Final(refusal)
    return {"set_name": set_name, "questions": _delete_set(set_name)}



# a set's size, its questions that answer logs hold, and questions of other sets drawn from it
def _holds(set_name: str) -> dict:
    with engine.connect() as conn:
        row = conn.execute(
            text(
                "SELECT count(*), "
                "(SELECT count(*) FROM question_logs l JOIN questions q ON q.id = l.question_id "
                "WHERE q.set_name = :s), "
                "(SELECT count(*) FROM questions c JOIN questions o ON o.id = c.source_question_id "
                "WHERE o.set_name = :s AND c.set_name <> :s) "
                "FROM questions WHERE set_name = :s"
            ),
            {"s": set_name},
        ).one()
    return {"questions": row[0], "answered": row[1], "drawn_from": row[2]}


# the door checks the holds first; a log or a set that took a question since is a refusal, not a server error
def _delete_set(set_name: str) -> int:
    try:
        with engine.begin() as conn:
            conn.execute(
                text(
                    "UPDATE questions SET source_question_id = NULL WHERE set_name = :s AND source_question_id IN "
                    "(SELECT id FROM questions WHERE set_name = :s)"
                ),
                {"s": set_name},
            )
            return conn.execute(text("DELETE FROM questions WHERE set_name = :s"), {"s": set_name}).rowcount
    except IntegrityError as e:
        said = f"questions of {set_name} were taken by a log or another set while it was being removed"
        raise errors.Final(said) from e
