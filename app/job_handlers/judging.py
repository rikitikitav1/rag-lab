import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import job_queue
import llm
import logging_setup
from evals import sampling
from models.eval import Question, QuestionLog
from models.registry import Purpose, Role
from orm import dsn
from orm.sync_db import Session
from sqlalchemy import and_, or_, select
from use_cases import experiment, judge, rejudge

from .base import register, require_model_ready, require_role_ready

log = logging_setup.get_logger(__name__)

# how many times one row may be put to the judge, not how many times the job may retry
_MAX_JUDGE_ATTEMPTS = 3

# how many times the job may come back: this bounds a counter that cannot be recorded
_MAX_SWEEPS = 3


def _not_capped(axis: str):
    attempts = QuestionLog.metrics[(axis, "attempts")].as_integer()
    return or_(attempts.is_(None), attempts < _MAX_JUDGE_ATTEMPTS)


# a row outside a control sample is not owed that axis, and nobody is coming for it
def _not_skipped(axis: str):
    return QuestionLog.metrics[(axis, "skipped")].as_string().is_(None)


# the one predicate for "a verdict is still coming": the second spelling read faithfulness
def still_to_judge():
    return or_(
        and_(
            QuestionLog.relevance.is_(None),
            _not_capped("relevance"),
            _not_skipped("relevance"),
        ),
        and_(
            QuestionLog.faithfulness.is_(None),
            QuestionLog.context.isnot(None),
            QuestionLog.context != "",
            _not_capped("faithfulness"),
            _not_skipped("faithfulness"),
        ),
        and_(
            QuestionLog.completeness.is_(None),
            Question.reference_answer.isnot(None),
            Question.reference_answer != "",
            _not_capped("completeness"),
            _not_skipped("completeness"),
        ),
    )


def _target_log_ids(session, options) -> list[int]:
    stmt = select(QuestionLog.id).where(QuestionLog.answered.is_(True))
    if options.get("log_ids"):
        return list(session.scalars(stmt.where(QuestionLog.id.in_(options["log_ids"]))))

    stmt = stmt.join(Question, QuestionLog.question_id == Question.id).where(still_to_judge())
    if options.get("run_name"):
        stmt = stmt.where(QuestionLog.run_name == options["run_name"])
    return list(session.scalars(stmt))


# drawn over the whole run and over the question, which is what two copies share
def _control_sample(
    session, run_name: str | None, log_ids: list[int], size: int, seed: int
) -> set[int]:
    stmt = select(QuestionLog.id).where(
        QuestionLog.answered.is_(True), QuestionLog.question_id.isnot(None)
    )
    stmt = (
        stmt.where(QuestionLog.run_name == run_name)
        if run_name
        else stmt.where(QuestionLog.id.in_(log_ids))
    )
    ordered = stmt.order_by(sampling.by_id_and_seed(QuestionLog.question_id, seed))
    return set(session.scalars(ordered.limit(size)))


@register("judge_answers")
def judge_answers(options: dict) -> None:
    bench = _bench_from(options)
    if bench.model:
        # the arm's override: a mistyped tag passed `require_role_ready` and failed per log
        require_model_ready(bench.model)
    else:
        require_role_ready(Role.judging)
    # resolved before any log: inside the loop it was swallowed per axis
    for purpose in _PURPOSES:
        bench.template(purpose)

    run_name = options.get("run_name")
    if run_name:
        # before the work: a retry judged its rows and then found `failed` refusing to aggregate
        experiment.revive_for_run(run_name)

    # a control does not need every row: on a sample it costs a third less and still says
    control = tuple(options.get("control_axes") or ())
    sample = options.get("control_sample")

    with Session() as session:
        log_ids = _target_log_ids(session, options)
        judged_for_control = (
            _control_sample(
                session, run_name, log_ids, int(sample), int(options.get("control_seed") or 0)
            )
            if control and sample
            else set(log_ids)
        )

    force = bool(options.get("log_ids"))
    width = judge_width(options.get("judge_width"))
    judged = 0

    def one(log_id):
        skip = () if log_id in judged_for_control else control
        try:
            return _judge_log(log_id, force=force, bench=bench, width=width, skip=skip)
        except Exception as e:
            log.error("judge.log_failed", log_id=log_id, error=str(e))
            _count_the_attempt(log_id, skip, f"{type(e).__name__}: {e}")
            return False

    if width == 1:
        judged = sum(1 for log_id in log_ids if one(log_id))
    else:
        with ThreadPoolExecutor(max_workers=width) as pool:
            judged = sum(1 for done in pool.map(one, log_ids) if done)
    log.info(
        "judge_answers.done",
        run_name=options.get("run_name"),
        judged=judged,
        total=len(log_ids),
        width=width,
    )
    if run_name:
        _sweep_again_if_rows_are_still_owed(options, run_name)
        experiment.try_aggregate_for_run(run_name)


# a row that broke before any axis ran kept its attempts untouched and stayed owed
def _count_the_attempt(log_id: int, skip: tuple, error: str) -> None:
    try:
        with Session() as session:
            ql = session.get(QuestionLog, log_id)
            if ql is None:
                return
            snapshot = _Snapshot(dict(ql.metrics or {}), {}, {})
            _mark_skipped(snapshot, ql, skip)
            for axis in _owed(ql, skip):
                snapshot.metrics[axis] = _errored_metric(snapshot.metrics, axis, error)
            ql.metrics = snapshot.metrics
            session.commit()
    except Exception as e:
        # the write that records the failure can be the failure. `_MAX_SWEEPS` ends it then
        log.error("judge.attempt_not_recorded", log_id=log_id, error=str(e))


# the job that would retry the row has just ended; sweeps end on the row cap or their own
def _sweep_again_if_rows_are_still_owed(options: dict, run_name: str) -> None:
    if options.get("log_ids"):
        return
    with Session() as session:
        owed = _target_log_ids(session, {"run_name": run_name})
    if not owed:
        return
    sweep = (options.get("sweep") or 0) + 1
    if sweep > _MAX_SWEEPS:
        # stopping quietly would leave the experiment in `running`, the trap this closes
        log.error("judge_answers.sweeps_exhausted", run_name=run_name, owed=len(owed))
        experiment.mark_failed_for_run(run_name)
        return
    log.warning("judge_answers.sweeping_again", run_name=run_name, owed=len(owed), sweep=sweep)
    carried = {k: v for k, v in options.items() if not k.startswith("_")}
    job_queue.enqueue("judge_answers", {**carried, "sweep": sweep})


# derived from the axes rather than listed, so a fourth axis does not need a second edit
_PURPOSES = tuple(Purpose[f"judge_{axis}"] for axis in rejudge.AXES)


# capped at the slots this worker believes the server has. A mirror, not the server
def judge_width(asked=None) -> int:
    asked = max(1, int(asked if asked is not None else os.getenv("JUDGE_WIDTH", "1")))
    slots = _parallel_slots()
    # each row in flight holds a session across its judge calls, so the pool bounds it too
    pool = dsn.POOL_SIZE + dsn.MAX_OVERFLOW - 1
    if asked > min(slots, pool):
        log.warning("judge.width_capped", asked=asked, slots=slots, pool=pool)
    return min(asked, slots, pool)


def _parallel_slots() -> int:
    return max(1, int(os.getenv("OLLAMA_NUM_PARALLEL", "1")))


def _bench_from(options: dict) -> judge.Bench:
    # an arm names its judge instead of switching the stand's active one, shared with all
    return rejudge.arm_bench(
        {"judge_model": options.get("judge_model"), **(options.get("judge_prompts") or {})}
    )


# once per row, not per axis: the width belongs to the pass and the sampler to the role
def _stamp(width: int) -> dict:
    sampler = llm.sampler_of("judging")
    return {
        "seed": sampler.get("seed"),
        "width": width,
        # `/api/ps` does not report slots, so the record names the mirror it was capped by
        "slots_believed": _parallel_slots(),
    }


# the python side of `still_to_judge`, with the material each verdict needs
def _owed(ql, skip=()) -> tuple[str, ...]:
    material = {
        "relevance": True,
        "faithfulness": bool(ql.context),
        "completeness": bool(ql.question and ql.question.reference_answer),
    }
    return tuple(
        axis
        for axis in rejudge.AXES
        if getattr(ql, axis) is None and axis not in skip and material[axis]
    )


# one way: `_not_skipped` drops the row for ever, and only an explicit `log_ids` job reaches it
def _mark_skipped(snapshot, ql, skip) -> None:
    for axis in skip:
        if getattr(ql, axis) is None:
            snapshot.metrics[axis] = {"skipped": "outside the control sample"}


def _judge_log(log_id: int, force: bool = False, bench=None, width: int = 1, skip=()) -> bool:
    with Session() as session:
        ql = session.get(QuestionLog, log_id)
        if ql is None or not ql.answered:
            return False

        question = ql.question.original_text
        reference = ql.question.reference_answer
        snapshot = _Snapshot(dict(ql.metrics), dict(ql.prompts), dict(ql.models))
        bench = bench or judge.ACTIVE

        stamp = _stamp(width)
        _mark_skipped(snapshot, ql, skip)
        owed = _owed(ql, skip)
        wrote = _judge_axis(
            ql, snapshot, "relevance", "relevance" in owed, force,
            judge.relevance_verdict, (question, ql.answer, bench), stamp,
        )
        wrote |= _judge_axis(
            ql, snapshot, "faithfulness", "faithfulness" in owed, force,
            judge.faithful_verdict, (question, ql.answer, ql.context, bench), stamp,
        )
        wrote |= _judge_axis(
            ql, snapshot, "completeness", "completeness" in owed, force,
            judge.completeness_verdict, (question, ql.answer, reference, bench), stamp,
        )

        ql.metrics = snapshot.metrics
        ql.prompts = snapshot.prompts
        ql.models = snapshot.models
        session.commit()
        return wrote


@dataclass
class _Snapshot:
    metrics: dict
    prompts: dict
    models: dict


def _judge_axis(ql, snapshot, axis, precondition, force, verdict_fn, args, stamp=None) -> bool:
    metrics = snapshot.metrics
    if not precondition or (not force and _errored(metrics, axis)):
        return False
    v, err = _run_axis(ql.id, axis, verdict_fn, *args)
    if v:
        setattr(ql, axis, str(v.score))
        metrics[axis] = _axis_metric(v, stamp)
        # the row says what scored it: a rejudge that cannot name the difference compares two
        snapshot.models[rejudge.JUDGE_MODEL_KEY] = v.model
        if v.purpose is not None:
            snapshot.prompts[v.purpose.name] = v.prompt_version
        return True
    metrics[axis] = _errored_metric(metrics, axis, err)
    return False


def _errored(metrics: dict, axis: str) -> bool:
    return metrics.get(axis, {}).get("attempts", 0) >= _MAX_JUDGE_ATTEMPTS


# an exception's text can carry a whole statement, and `metrics` is returned by the API
_ERROR_CHARS = 300


def _errored_metric(metrics: dict, axis: str, err: str) -> dict:
    was = metrics.get(axis) or {}
    # a row judged again is no longer skipped, and both marks at once read as neither
    was = {k: v for k, v in was.items() if k != "skipped"}
    return {**was, "error": err[:_ERROR_CHARS], "attempts": was.get("attempts", 0) + 1}


def _run_axis(log_id, axis, verdict_fn, *args):
    try:
        return verdict_fn(*args), None
    except Exception as e:
        log.error("judge.axis_failed", axis=axis, log_id=log_id, error=str(e))
        # the kind and what it said: a row that failed three times left only an exception name
        return None, f"{type(e).__name__}: {e}"


# a fan-out changes verdicts and a seed makes a pass repeatable
def _axis_metric(verdict, stamp: dict | None = None) -> dict:
    return {
        "reason": verdict.reason,
        "elapsed": verdict.elapsed,
        "model": verdict.model,
        **(stamp or {}),
    }
