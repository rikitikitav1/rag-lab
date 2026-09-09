import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import job_queue
import limits
import llm
import logging_setup
from evals import guest_axes, sampling
from models.eval import Question, QuestionLog
from models.registry import Purpose, Role
from orm import dsn
from orm.sync_db import Session
from sqlalchemy import and_, cast, func, or_, select, text
from sqlalchemy.dialects.postgresql import JSONB
from use_cases import experiment, judge, rejudge

from .base import register, require_model_ready, require_role_ready

log = logging_setup.get_logger(__name__)

# how many times one row may be put to the judge, not how many times the job may retry
_MAX_JUDGE_ATTEMPTS = 3

# how many times the job may come back: this bounds a counter that cannot be recorded
_MAX_SWEEPS = 3

# the lock spans three model calls, so a second waiter gives up rather than queueing behind them
_LOCK_WAIT_MS = 5000


def _bounded_wait(session) -> None:
    session.execute(text(f"SET LOCAL lock_timeout = '{_LOCK_WAIT_MS}ms'"))

# a pass whose guests cost several times our own can lose the judge to a neighbour halfway
def judge_on_card() -> bool | None:
    seen = [m for m in llm.residency() if m["model"] == llm.resolve_name("judging")]
    return seen[0]["vram_mb"] >= seen[0]["size_mb"] if seen else None


# the runtime image carries no `ragas`, and a pass that cannot score says so before it walks
def guests_available() -> bool:
    from importlib.util import find_spec

    return find_spec("ragas") is not None


def _not_capped(axis: str):
    attempts = QuestionLog.metrics[(axis, "attempts")].as_integer()
    return or_(attempts.is_(None), attempts < _MAX_JUDGE_ATTEMPTS)


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
            _not_capped(axis),
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
    return and_(_not_refused(), or_(
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
    ))


def _target_log_ids(session, options) -> list[int]:
    stmt = select(QuestionLog.id).where(QuestionLog.answered.is_(True))
    if options.get("log_ids"):
        asked = list(options["log_ids"])
        found = set(session.scalars(stmt.where(QuestionLog.id.in_(asked))))
        # the caller's order is kept: the judge's own history of requests is a measured variable
        return [i for i in asked if i in found]

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


# the console's only action marked the row and the job judged on to the end regardless
def _stop_asked(job_id) -> bool:
    return job_id is not None and job_queue.is_cancelled(job_id)


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
    job_id = options.get("_job_id")
    residency = _residency(job_id)
    judged = 0

    def one(log_id):
        if _stop_asked(job_id):
            return False
        skip = () if log_id in judged_for_control else control
        try:
            return _judge_log(
                log_id, force=force, bench=bench, width=width, skip=skip, residency=residency
            )
        except Exception as e:
            log.error("judge.log_failed", log_id=log_id, error=str(e))
            _count_the_attempt(log_id, skip, f"{type(e).__name__}: {e}")
            return False

    if width == 1:
        for log_id in log_ids:
            if _stop_asked(job_id):
                break
            judged += bool(one(log_id))
    else:
        with ThreadPoolExecutor(max_workers=width) as pool:
            judged = sum(1 for done in pool.map(one, log_ids) if done)
    stopped = _stop_asked(job_id)
    log.info(
        "judge_answers.done",
        run_name=options.get("run_name"),
        judged=judged,
        total=len(log_ids),
        width=width,
        cancelled=stopped or None,
    )
    # a sweep after a cancellation queues the work again, which is the opposite of cancelling
    if run_name and not stopped:
        _sweep_again_if_rows_are_still_owed(options, run_name)
        experiment.try_aggregate_for_run(run_name)


# a row that broke before any axis ran kept its attempts untouched and stayed owed
def _count_the_attempt(log_id: int, skip: tuple, error: str) -> None:
    try:
        with Session() as session:
            _bounded_wait(session)
            ql = session.get(QuestionLog, log_id, with_for_update=True)
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


# a probe run three times is an operation: as a script it needed a waiter beside the queue
@register("judge_language")
def judge_language(options: dict) -> None:
    from evals import judge_language as probe
    from evals import measurements

    require_role_ready(Role.judging)
    run_name = options.get("run_name") or "arc3_agent_baseline"
    rows = int(options.get("rows") or 40)
    job_id = options.get("_job_id")
    out = probe.measure(
        run_name, rows,
        note=lambda line: log.info("judge_language.pair", pair=line),
        stop=lambda: _stop_asked(job_id),
    )
    # the path is derived, never taken from options: a number with no file cannot be cited
    where = measurements.record("judge_language", run_name, out)
    log.info("judge_language.done", run_name=run_name, wrote=where, result=out)


# by request only: the guests cost 9.6x our own three axes, and no run waits on them
@register("judge_guest_axes")
def judge_guest_axes(options: dict) -> None:
    if not guests_available():
        raise ValueError("this runtime carries no `ragas`, the guest axes cannot be scored here")
    require_role_ready(Role.judging)

    with Session() as session:
        log_ids = _guest_log_ids(session, options)
    # counted over the rows this pass will walk, which is what the REST door counts as well
    if len(log_ids) > limits.MAX_GUEST_ROWS:
        raise ValueError(
            f"{len(log_ids)} rows owe a guest axis, over the cap of {limits.MAX_GUEST_ROWS}"
        )
    # drawn once: a sweep that redraws turns a budget of fifty rows into a hundred and fifty
    budget = set(log_ids)
    width = judge_width(options.get("judge_width"))
    job_id = options.get("_job_id")
    seen = _residency(job_id)
    stamp = _stamp(width, seen)
    started_on_card = seen.on_card

    def one(log_id):
        if _stop_asked(job_id):
            return False
        try:
            return _score_guests(log_id, stamp)
        except Exception as e:
            log.error("guest_axes.log_failed", log_id=log_id, error=str(e))
            return False

    scored, walked = 0, 0
    # its own sweeps, in this process: a queued one lands where the handler is and `ragas` is not
    for sweep in range(_MAX_SWEEPS):
        if not log_ids or _stop_asked(job_id):
            break
        walked += len(log_ids)
        if width == 1:
            for log_id in log_ids:
                if _stop_asked(job_id):
                    break
                scored += bool(one(log_id))
        else:
            with ThreadPoolExecutor(max_workers=width) as pool:
                scored += sum(1 for done in pool.map(one, log_ids) if done)
        with Session() as session:
            still = _guest_log_ids(session, {**options, "sample": None})
        log_ids = [i for i in still if i in budget]
        if log_ids and not _stop_asked(job_id):
            log.warning("judge_guest_axes.sweeping_again", owed=len(log_ids), sweep=sweep + 1)
    if log_ids and not _stop_asked(job_id):
        log.error("judge_guest_axes.sweeps_exhausted", owed=len(log_ids))
    ended_on_card = judge_on_card()
    # a swallowed reading leaves `started_on_card` None, and None against a real bool is not a move
    if None not in (started_on_card, ended_on_card) and started_on_card != ended_on_card:
        # the rows carry their own reading; this says the pass is not one residency any more
        log.error(
            "judge_guest_axes.residency_moved",
            started_on_card=started_on_card,
            ended_on_card=ended_on_card,
            rows=len(log_ids),
        )
    log.info(
        "judge_guest_axes.done",
        run_name=options.get("run_name"),
        scored=scored,
        total=walked,
        width=width,
        on_card=ended_on_card,
    )


# what the door counts before it queues: the same rows the pass would walk
def guest_rows_of(run_name: str) -> int:
    with Session() as session:
        return len(_guest_log_ids(session, {"run_name": run_name}))


def _guest_log_ids(session, options) -> list[int]:
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
    return _drawn(found, options.get("sample"), options.get("seed"))


# the guests are a calibration on a subsample, not an axis of every run: they cost 35x ours a row
def _drawn(ids: list[int], sample, seed) -> list[int]:
    if not sample or len(ids) <= int(sample):
        return ids
    import random

    # seeded and over sorted ids, so a sweep redraws the same rows rather than wandering the run
    picked = random.Random(int(seed or 0)).sample(sorted(ids), int(sample))
    return sorted(picked)


# read, close, score, merge: the lock went and the session stayed open through minutes of calls
def _score_guests(log_id: int, stamp: dict) -> bool:
    with Session() as session:
        ql = session.get(QuestionLog, log_id)
        if ql is None or not ql.answered:
            return False
        metrics = dict(ql.metrics or {})
        owed = list(guest_axes.owed(ql, metrics))
        row = guest_axes.carried(ql)

    scored, wrote = {}, False
    for axis in owed:
        if _errored(metrics, axis):
            continue
        try:
            # the same stamp our axes carry, plus the card: a number says how it was taken
            scored[axis] = {**stamp, **guest_axes.score(axis, row), "on_card": judge_on_card()}
            wrote = True
        except Exception as e:
            log.error("guest_axes.failed", axis=axis, log_id=log_id, error=str(e))
            scored[axis] = _errored_metric(metrics, axis, f"{type(e).__name__}: {e}")
    return _merge_guest_scores(log_id, scored) and wrote


# the lock is held for one statement, not for the pass: our verdicts still cannot vanish under it
def _merge_guest_scores(log_id: int, scored: dict) -> bool:
    if not scored:
        return False
    try:
        with Session() as session:
            _bounded_wait(session)
            ql = session.get(QuestionLog, log_id, with_for_update=True)
            if ql is None:
                return False
            ql.metrics = {**(ql.metrics or {}), **scored}
            session.commit()
        return True
    except Exception as e:
        # a guest number nobody could write is a number nobody took: the row stays owed
        log.error("guest_axes.not_written", log_id=log_id, error=str(e))
        return False


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


@dataclass(frozen=True)
class Residency:
    id: int | None
    on_card: bool | None


# `/api/ps` carries no load moment, so a residency is named by the pass that caused the load
def _residency(job_id) -> Residency:
    try:
        on_card = judge_on_card()
        # partly on the card is a different instrument: layers on the cpu answer with other kernels
        if on_card is not True:
            return Residency(job_id, on_card)
        prev = _last_residency()
        # `/api/ps` cannot say who loaded what between two passes, but the queue can
        if prev is None or _loaded_since(prev, job_id):
            return Residency(job_id, on_card)
        return Residency(prev, on_card)
    except Exception as e:
        # a reading about the card must not take down the pass that only wanted to stamp itself
        log.warning("judge.residency_unknown", error=str(e))
        return Residency(None, None)


def _loaded_since(prev: int, job_id) -> bool:
    import job_specs
    from models.jobs import Job, JobStatus

    # `running` and the ones that died after loading evict the judge exactly as `done` ones do
    ours = {JobStatus.new}
    with Session() as session:
        types = session.scalars(
            select(Job.type).where(Job.id > prev, Job.id != job_id, Job.status.notin_(ours))
        )
        return any(job_specs.disturbs_the_judge(t) for t in types)


def _last_residency() -> int | None:
    from sqlalchemy import desc

    with Session() as session:
        for axis in rejudge.AXES:
            got = session.scalar(
                select(QuestionLog.metrics[(axis, "residency_id")].as_integer())
                .where(QuestionLog.metrics[(axis, "residency_id")].as_integer().isnot(None))
                .order_by(desc(QuestionLog.id))
                .limit(1)
            )
            if got is not None:
                return got
    return None


# once per row, not per axis: the width belongs to the pass and the sampler to the role
def _stamp(width: int, residency: Residency | None = None) -> dict:
    from datetime import datetime, timezone

    seen = residency or Residency(None, None)
    sampler = llm.sampler_of("judging")
    return {
        # at temperature zero this says the sampler took no part, not that the pass repeats
        "seed": sampler.get("seed"),
        "width": width,
        # two arms are comparable only inside one residency, and this is how a reader checks
        "residency_id": seen.id,
        # a pass that found the judge half on the cpu is not the pass that found it whole
        "on_card": seen.on_card,
        # numbers from two backends must not add up silently, so the record names its own
        "engine": llm.engine(),
        "judged_at": datetime.now(timezone.utc).isoformat(),
        # `/api/ps` does not report slots, so the record names the mirror it was capped by
        "slots_believed": _parallel_slots(),
    }


# the owner's rule of 06.09: on a refusal an axis does not apply, so both sides stay silent
def _refused(metrics) -> bool:
    return (metrics or {}).get("refusal") is True


# the python side of `still_to_judge`, with the material each verdict needs
def _owed(ql, skip=()) -> tuple[str, ...]:
    if _refused(ql.metrics):
        return ()
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
        if getattr(ql, axis, None) is None:
            snapshot.metrics[axis] = {"skipped": "outside the control sample"}


# scored outside the lock: three model calls under `FOR UPDATE` closed an axis on the loser's timeout
def _judge_log(log_id: int, force: bool = False, bench=None, width: int = 1, skip=(),
               residency: Residency | None = None) -> bool:
    bench = bench or judge.ACTIVE
    with Session() as session:
        ql = session.get(QuestionLog, log_id)
        if ql is None or not ql.answered:
            return False
        metrics = dict(ql.metrics)
        owed = _owed(ql, skip)
        calls = {
            "relevance": (judge.relevance_verdict, (ql.question.original_text, ql.answer, bench)),
            "faithfulness": (
                judge.faithful_verdict,
                (ql.question.original_text, ql.answer, ql.context, bench),
            ),
            "completeness": (
                judge.completeness_verdict,
                (ql.question.original_text, ql.answer, ql.question.reference_answer, bench),
            ),
        }

    stamp = _stamp(width, residency)
    taken = {}
    for axis, (verdict_fn, args) in calls.items():
        if axis not in owed or (not force and _errored(metrics, axis)):
            continue
        taken[axis] = _run_axis(log_id, axis, verdict_fn, *args)
    return _merge_our_scores(log_id, taken, skip, stamp, force)


# the lock is held for one statement: a verdict taken meanwhile wins unless this pass was forced
def _merge_our_scores(log_id: int, taken: dict, skip, stamp: dict, force: bool) -> bool:
    if not taken and not skip:
        return False
    try:
        with Session() as session:
            _bounded_wait(session)
            ql = session.get(QuestionLog, log_id, with_for_update=True)
            if ql is None:
                return False
            snapshot = _Snapshot(dict(ql.metrics), dict(ql.prompts), dict(ql.models))
            _mark_skipped(snapshot, ql, skip)
            still = _owed(ql, skip)
            wrote = False
            for axis, (v, err) in taken.items():
                if axis not in still and not force:
                    continue
                wrote |= _apply_axis(ql, snapshot, axis, v, err, stamp)
            _settle_outcome(ql, snapshot)
            ql.metrics = snapshot.metrics
            ql.prompts = snapshot.prompts
            ql.models = snapshot.models
            session.commit()
            return wrote
    except Exception as e:
        # a verdict nobody could write is a verdict nobody took: the row stays owed
        log.error("judge.not_written", log_id=log_id, error=str(e))
        return False


# groundedness is unknowable when the answer is written, so the judge is what settles the outcome
def _settle_outcome(ql, snapshot) -> None:
    from evals.stats import score_of
    from outcomes import Outcome

    said = snapshot.metrics.get("outcome")
    if said is None or ql.faithfulness is None:
        return
    # only the downgrade it alone can make: settling more would freeze a wider derivation
    if said == Outcome.answered and score_of(ql.faithfulness) == 0:
        snapshot.metrics["settled_outcome"] = str(Outcome.answered_ungrounded)


@dataclass
class _Snapshot:
    metrics: dict
    prompts: dict
    models: dict


def _apply_axis(ql, snapshot, axis, v, err, stamp=None) -> bool:
    metrics = snapshot.metrics
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
