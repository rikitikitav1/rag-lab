"""What a run still owes each axis, and what the rest of its rows are missing to owe it.

`n_scored` says how many rows carry a number. It says nothing about the others, and that silence
cost an hour of card on 06.09: a guest pass ran over two pools holding no reference answer, so two
of three axes could never have produced anything. This counts the debt and names what blocks it.

Every count is taken over one population, the run's answered rows, and `reconciles` says the parts
add up to it. That sum is the live guard on the materiality pair: a predicate answering NULL rather
than false drops rows from both sides at once, and only the sum notices.
"""

import logging_setup
from evals import guest_axes
from evals.replay import REPLAYABLE_ARMS
from models.eval import Question, QuestionLog
from orm.sync_db import Session
from sqlalchemy import and_, func, not_, or_, select

log = logging_setup.get_logger(__name__)


# every count shares one FROM and one WHERE, so they are one statement with filtered aggregates
def _totals(session, run_name: str, columns: dict) -> dict:
    stmt = (
        select(*[value.label(key) for key, value in columns.items()])
        .select_from(QuestionLog)
        .outerjoin(Question, QuestionLog.question_id == Question.id)
        .where(QuestionLog.run_name == run_name, QuestionLog.answered.is_(True))
    )
    return dict(session.execute(stmt).mappings().one())


# the shapes this key takes, in one place, because sql and python read it in different languages
REPLAY_SHAPES = {
    "json null": {"dropped_sources": None},
    "empty list": {"dropped_sources": []},
    "key absent": {},
    "a real drop": {"dropped_sources": [{"source": "a.md"}]},
}


# the sql below must split REPLAY_SHAPES exactly as `replay.rerun` does, and only a base proves it
def _replay_columns() -> dict:
    from sqlalchemy import and_, cast
    from sqlalchemy.dialects.postgresql import JSONB, JSONPATH

    no_turns = or_(
        func.coalesce(func.jsonb_typeof(QuestionLog.transcript), "") != "array",
        QuestionLog.transcript == cast("[]", JSONB),
    )
    # `[*]` in lax mode wraps a json null into a list and answers true, so the type is asked instead
    dropped = QuestionLog.metrics["retrieval"]["dropped_sources"]
    # only a `weak` verdict reads the scores the drop erased, and old rows did not keep them
    weak_unrecorded = and_(
        func.coalesce(QuestionLog.metrics["fallback_reason"].astext, "") == "weak",
        func.coalesce(func.jsonb_typeof(dropped), "") == "array",
        dropped != cast("[]", JSONB),
        not_(
            func.coalesce(
                func.jsonb_path_exists(
                    QuestionLog.metrics["spans"], cast("$[*].dropped", JSONPATH)
                ),
                False,
            )
        ),
    )
    # another arm's row is not replayable by this, and the debt counted it as if it were
    other_arm = func.coalesce(
        QuestionLog.metrics["config"]["orchestrator"]["name"].astext, "langgraph_ported"
    ).notin_(list(REPLAYABLE_ARMS))
    return {
        "replay__other_arm": func.count().filter(other_arm),
        "replay__no_turns": func.count().filter(and_(not_(other_arm), no_turns)),
        "replay__weak_unrecorded": func.count().filter(
            and_(not_(other_arm), not_(no_turns), weak_unrecorded)
        ),
        "replay__replayable": func.count().filter(
            and_(not_(other_arm), not_(no_turns), not_(weak_unrecorded))
        ),
    }


def _answered_it(axis: str):
    return QuestionLog.metrics[(axis, "abstained")].as_boolean()


def _columns() -> dict:
    from job_handlers.judging import (
        GUEST_MATERIAL,
        _not_capped,
        guest_clauses,
        still_to_judge,
    )

    clauses = dict(zip(guest_axes.AXES, guest_clauses(), strict=True))
    columns = {"population": func.count(), "ours": func.count().filter(still_to_judge())}
    # three names serve all three axes, and counting them per axis repeated five of nine
    for name in sorted({n for guest in guest_axes.AXES.values() for n in guest.needs}):
        columns[f"without__{name}"] = func.count().filter(not_(GUEST_MATERIAL[name]))
    for axis, guest in guest_axes.AXES.items():
        columns[f"owed__{axis}"] = func.count().filter(clauses[axis])
        columns[f"answered__{axis}"] = func.count().filter(_answered_it(axis).isnot(None))
        columns[f"abstained__{axis}"] = func.count().filter(_answered_it(axis).is_(True))
        columns[f"unreachable__{axis}"] = func.count().filter(
            or_(*[not_(GUEST_MATERIAL[n]) for n in guest.needs])
        )
        # a fourth bucket, or the guard reads False on the ordinary row that failed three times
        columns[f"capped__{axis}"] = func.count().filter(
            and_(*[GUEST_MATERIAL[n] for n in guest.needs], not_(_not_capped(axis)))
        )
        # priced by this run's own rows: a guess from another pool made the 9.6 an upper bound
        columns[f"seconds__{axis}"] = func.avg(
            QuestionLog.metrics[(axis, "elapsed")].as_float()
        )
    return columns | _replay_columns()


def of(run_name: str) -> dict:
    with Session() as session:
        got = _totals(session, run_name, _columns())
    population = got["population"]
    guests = {}
    for axis, guest in guest_axes.AXES.items():
        owed = got[f"owed__{axis}"]
        seconds = got[f"seconds__{axis}"]
        guests[axis] = {
            "owed": owed,
            "answered": got[f"answered__{axis}"],
            "abstained": got[f"abstained__{axis}"],
            "rows_without": {
                n: got[f"without__{n}"] for n in guest.needs if got[f"without__{n}"]
            },
            "seconds_each": round(float(seconds), 1) if seconds is not None else None,
            "given_up_on": got[f"capped__{axis}"],
            # owed, answered, unreachable and given up on: the four states cover the population
            "reconciles": owed + got[f"answered__{axis}"] + got[f"unreachable__{axis}"]
            + got[f"capped__{axis}"] >= population,
        }
    left = sum(a["owed"] * (a["seconds_each"] or 0) for a in guests.values())
    return {
        "answered_rows": population,
        "ours_still_to_judge": got["ours"],
        "replay": {
            "replayable": got["replay__replayable"],
            "no_turns": got["replay__no_turns"],
            "weak_drop_unrecorded": got["replay__weak_unrecorded"],
            "another_arm": got["replay__other_arm"],
            "reconciles": got["replay__replayable"]
            + got["replay__no_turns"]
            + got["replay__weak_unrecorded"]
            + got["replay__other_arm"]
            == population,
        },
        "guests": guests,
        # what finishing every guest debt would cost at this run's own measured price
        "guest_seconds_left": round(left) or None,
    }


# a diagnostic that raises must not take the measurement standing beside it down
def safely(run_name: str) -> dict:
    try:
        return of(run_name)
    except Exception as e:
        log.error("run_debts.unavailable", run_name=run_name, error=str(e))
        return {"unavailable": f"{type(e).__name__}: {e}"}
