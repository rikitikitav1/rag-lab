import limits
from errors import Refusal
from evals import guest_axes
from models.eval import Question, QuestionLog
from orm.sync_db import Session
from sqlalchemy import and_, cast, func, or_, select
from sqlalchemy.dialects.postgresql import JSONB

# how many times one row may be put to the judge, not how many times the job may retry
MAX_JUDGE_ATTEMPTS = 3


# the runtime image carries no `ragas`, and a pass that cannot score says so before it walks
def guests_available() -> bool:
    from importlib.util import find_spec

    return find_spec("ragas") is not None


def not_capped(axis: str):
    attempts = QuestionLog.metrics[(axis, "attempts")].as_integer()
    return or_(attempts.is_(None), attempts < MAX_JUDGE_ATTEMPTS)


# a row outside a control sample is not owed that axis, and nobody is coming for it
def _not_skipped(axis: str):
    return QuestionLog.metrics[(axis, "skipped")].as_string().is_(None)


# the sql spelling of `guest_axes.HAS`; every entry is total, so `NOT` over it counts the rest
GUEST_MATERIAL = {
    "question_text": func.coalesce(QuestionLog.question_text, "") != "",
    "answer": func.coalesce(QuestionLog.answer, "") != "",
    # no `jsonb_array_length`: it raises on the jsonb scalar 375 rows hold, rather than saying false
    "contexts": and_(
        func.coalesce(func.jsonb_typeof(QuestionLog.contexts), "") == "array",
        QuestionLog.contexts != cast("[]", JSONB),
    ),
    "reference": func.coalesce(Question.reference_answer, "") != "",
}


# an abstention is a verdict, so the key's presence ends the debt, not the score's value
def guest_clauses():
    return [
        and_(
            QuestionLog.metrics[(axis, "abstained")].as_string().is_(None),
            *[GUEST_MATERIAL[name] for name in guest.needs],
            not_capped(axis),
            _not_skipped(axis),
        )
        for axis, guest in guest_axes.AXES.items()
    ]


# the sql half of `_owed`: one rule, and the outcome key is read the same way on both sides
def _not_refused():
    said = QuestionLog.metrics["refusal"].astext
    return func.coalesce(said, "") != "true"


# the one predicate for "a verdict is still coming": the second spelling read faithfulness
def still_to_judge():
    return and_(
        _not_refused(),
        or_(
            and_(
                QuestionLog.relevance.is_(None),
                not_capped("relevance"),
                _not_skipped("relevance"),
            ),
            and_(
                QuestionLog.faithfulness.is_(None),
                QuestionLog.context.isnot(None),
                QuestionLog.context != "",
                not_capped("faithfulness"),
                _not_skipped("faithfulness"),
            ),
            and_(
                QuestionLog.completeness.is_(None),
                Question.reference_answer.isnot(None),
                Question.reference_answer != "",
                not_capped("completeness"),
                _not_skipped("completeness"),
            ),
        ),
    )


# what the door counts before it queues: the same rows the pass would walk
def guest_rows_of(run_name: str) -> int:
    with Session() as session:
        return len(guest_log_ids(session, {"run_name": run_name}))


# what refuses a guest pass before it is queued, at the door and on an experiment's arm alike
def refuse_guest_pass(run_name: str, sample: int | None = None, said_as: str = "") -> None:
    if not guests_available():
        raise Refusal("conflict", f"{said_as}this runtime carries no `ragas`, the guest axes cannot be scored")
    # a typo in the name used to be a job over nothing
    owed = guest_rows_of(run_name)
    if not owed:
        raise Refusal("missing", f"{said_as}run {run_name} owes no guest axis")
    # the pass walks the drawn subsample, so the cap is read over the same rows the handler counts
    will_walk = min(owed, sample) if sample else owed
    if will_walk > limits.MAX_GUEST_ROWS:
        raise Refusal(
            "invalid", f"{said_as}{will_walk} rows would be walked, over the cap of {limits.MAX_GUEST_ROWS}"
        )


def guest_log_ids(session, options) -> list[int]:
    stmt = (
        select(QuestionLog.id)
        .outerjoin(Question, QuestionLog.question_id == Question.id)
        .where(QuestionLog.answered.is_(True), or_(*guest_clauses()))
    )
    if options.get("log_ids"):
        stmt = stmt.where(QuestionLog.id.in_(options["log_ids"]))
    if options.get("run_name"):
        stmt = stmt.where(QuestionLog.run_name == options["run_name"])
    found = list(session.scalars(stmt))
    return drawn(found, options.get("sample"), options.get("seed"))


# the guests are a calibration on a subsample, not an axis of every run: they cost 35x ours a row
def drawn(ids: list[int], sample, seed) -> list[int]:
    if not sample or len(ids) <= int(sample):
        return ids
    import random

    # seeded and over sorted ids, so a sweep redraws the same rows rather than wandering the run
    picked = random.Random(int(seed or 0)).sample(sorted(ids), int(sample))
    return sorted(picked)
