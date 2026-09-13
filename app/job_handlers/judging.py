import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import engines
import job_queue
import limits
import llm
import logging_setup
import token_fields
from engines import card
from errors import StandFault
from evals import guest_axes, measurements, sampling
from models.eval import Question, QuestionLog
from models.registry import Engine, EngineKind, Placement, Purpose, Role
from orm import dsn
from orm.sync_db import Session
from passes import Pass, Seat
from redaction import redact
from sqlalchemy import DateTime, and_, cast, func, or_, select, text
from sqlalchemy.dialects.postgresql import JSONB
from use_cases import experiment, judge, rejudge

from .base import Final, register, require_card, require_model_ready, require_role_ready

log = logging_setup.get_logger(__name__)

# how many times one row may be put to the judge, not how many times the job may retry
_MAX_JUDGE_ATTEMPTS = 3

# how many times the job may come back: this bounds a counter that cannot be recorded
_MAX_SWEEPS = 3

# the lock spans three model calls, so a second waiter gives up rather than queueing behind them
_LOCK_WAIT_MS = 5000


def _bounded_wait(session) -> None:
    session.execute(text(f"SET LOCAL lock_timeout = '{_LOCK_WAIT_MS}ms'"))


residency_instrument = card.placement_instrument


# a pass whose guests cost several times our own can lose the judge to a neighbour halfway
def judge_on_card(model: str | None = None, role: str = "judging") -> bool | None:
    judged_by = llm.resolve_for(role, model)
    return card.model_on_card(judged_by.engine, judged_by.name)


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


# a run finished on another engine than the one that began it is two rulers in one record
def _refuse_a_second_judge(run_name: str, model: str | None) -> None:
    judge = llm.resolve_for("judging", model).engine.name
    stamps = [QuestionLog.metrics[(axis, "engine_name")].as_string() for axis in rejudge.AXES]
    with Session() as session:
        rows = session.execute(select(*stamps).where(QuestionLog.run_name == run_name)).all()
    earlier = sorted({name for row in rows for name in row if name} - {judge})
    if earlier:
        raise Final(
            f"run {run_name} was judged on {', '.join(earlier)} and the judge now sits on {judge};"
            f" seat it back there, or rejudge the whole run into a copy through /v1/eval/rejudge"
        )


# guest numbers from two models on one run are two rulers in one record, the gemma mix by design
def _refuse_a_second_guest(run_name: str, model: str | None) -> None:
    guest = llm.resolve_for(Role.ragas, model)
    now = (guest.name, guest.engine.name)
    stamps = [QuestionLog.metrics[(axis, key)].as_string() for axis in guest_axes.NAMES for key in ("model", "engine")]
    with Session() as session:
        rows = session.execute(select(*stamps).where(QuestionLog.run_name == run_name)).all()
    seen = {(row[i], row[i + 1]) for row in rows for i in range(0, len(row), 2) if row[i]}
    # a stamp from before the engine was written names the model alone
    earlier = sorted(f"{m}@{e}" for m, e in seen if (m, e) != now and not (e is None and m == now[0]))
    if earlier:
        raise Final(
            f"run {run_name} holds guest numbers from {', '.join(earlier)} and this pass would score with"
            f" {now[0]}@{now[1]}; score a copy through a rejudge arm with guest_model"
        )


@register("judge_answers")
def judge_answers(options: dict) -> None:
    bench = _bench_from(options)
    if bench.model:
        # the arm's override: a mistyped tag passed `require_role_ready` and failed per log
        require_model_ready(bench.model, "judging")
    else:
        require_role_ready(Role.judging, take_card=False)
    if options.get("run_name") and not options.get("log_ids"):
        _refuse_a_second_judge(options["run_name"], bench.model)
    # once, with the bench's model when it names one: the role gate asked for the card a second time
    require_card("judging", bench.model)
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
    model = bench.model if bench else None
    walk = Pass(job_id, (Seat(Role.judging, model),), residency=lambda: _residency(job_id, model))
    judged = 0

    def one(log_id):
        if not walk.before_row():
            return None
        skip = () if log_id in judged_for_control else control
        try:
            return _judge_log(
                log_id, force=force, bench=bench, width=width, skip=skip, residency=walk
            )
        except StandFault:
            raise
        except Exception as e:
            log.error("judge.log_failed", log_id=log_id, error=str(e))
            _count_the_attempt(log_id, skip, _error_text(e))
            return False

    judged = _each(log_ids, width, one)
    stopped = walk.cancelled()
    log.info(
        "judge_answers.done",
        run_name=options.get("run_name"),
        judged=judged,
        total=len(log_ids),
        width=width,
        cancelled=stopped or None,
    )
    # a forced pass over named rows can find each of them owing nothing, and that is not a broken judge
    walk.close(owed=0 if force else len(log_ids), done=judged)
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
            snapshot = Snapshot(dict(ql.metrics or {}), {}, {})
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

    require_role_ready(Role.judging)
    run_name = options.get("run_name") or "arc3_agent_baseline"
    rows = int(options.get("rows") or 40)
    job_id = options.get("_job_id")
    # the generator may sit on another card engine and give the card back, which is not a spill
    walk = Pass(job_id, (Seat(Role.judging), Seat(Role.generation, spill_from=None)),
                residency=lambda: _residency(job_id))
    # the control read out of regime twice, and nothing said whether the judge was whole on the card
    out = probe.measure(
        run_name, rows,
        note=lambda line: log.info("judge_language.pair", pair=line),
        stop=lambda: not walk.before_row(),
        log_ids=options.get("log_ids"),
        stamp=stamp_of(judge_width(options.get("judge_width")), walk.get()),
    )
    walk.close(owed=out.get("n_asked", 0), done=sum(1 for row in out.get("rows", ()) if row["score"] is not None))
    # the path is derived, never taken from options: a number with no file cannot be cited
    where = measurements.record("judge_language", run_name, out)
    log.info("judge_language.done", run_name=run_name, wrote=where, result=out)


# by request only: the guests cost 9.6x our own three axes, and no run waits on them
@register("judge_guest_axes")
def judge_guest_axes(options: dict) -> None:
    if not guests_available():
        raise Final("this runtime carries no `ragas`, the guest axes cannot be scored here")
    model = options.get("guest_model")
    if model:
        require_model_ready(model, "ragas")
    else:
        require_role_ready(Role.ragas, take_card=False)
    # every refusal before the card: a refused pass that took it first reseated it three times
    if options.get("run_name") and not options.get("log_ids"):
        _refuse_a_second_guest(options["run_name"], model)
    with Session() as session:
        log_ids = _guest_log_ids(session, options)
    # counted over the rows this pass will walk, which is what the REST door counts as well
    if len(log_ids) > limits.MAX_GUEST_ROWS:
        raise Final(
            f"{len(log_ids)} rows owe a guest axis, over the cap of {limits.MAX_GUEST_ROWS}"
        )
    require_card("ragas", model)
    # drawn once: a sweep that redraws turns a budget of fifty rows into a hundred and fifty
    budget = set(log_ids)
    width = judge_width(options.get("judge_width"))
    messages = options.get("messages") or guest_axes.MESSAGE_FORMS[0]
    job_id = options.get("_job_id")
    walk = Pass(job_id, (Seat(Role.ragas, model), Seat(Role.ragas_embedding, spill_from=None)),
                residency=lambda: _residency(job_id, model, role=Role.ragas))
    # the guest's own stamp names its engine and sampler; ours would hand the row a second spelling of each
    stamp = {k: v for k, v in stamp_of(width, walk.get(), model, role=Role.ragas).items() if k not in ("engine", "sampler")}

    def one(log_id):
        if not walk.before_row():
            return None
        try:
            return _score_guests(log_id, stamp, messages, on_card=walk.on_card(Role.ragas), model=model)
        except StandFault:
            raise
        except Exception as e:
            log.error("guest_axes.log_failed", log_id=log_id, error=str(e))
            return False

    scored, walked = 0, 0
    # its own sweeps, in this process: a queued one lands where the handler is and `ragas` is not
    for sweep in range(_MAX_SWEEPS):
        if not log_ids or walk.cancelled():
            break
        walked += len(log_ids)
        scored += _each(log_ids, width, one)
        with Session() as session:
            still = _guest_log_ids(session, {**options, "sample": None})
        log_ids = [i for i in still if i in budget]
        if log_ids and not walk.cancelled():
            log.warning("judge_guest_axes.sweeping_again", owed=len(log_ids), sweep=sweep + 1)
    if log_ids and not walk.cancelled():
        log.error("judge_guest_axes.sweeps_exhausted", owed=len(log_ids))
    walk.close(owed=len(budget), done=scored)
    log.info(
        "judge_guest_axes.done",
        run_name=options.get("run_name"),
        scored=scored,
        total=walked,
        width=width,
        on_card=walk.ended_on_card,
    )


# what the door counts before it queues: the same rows the pass would walk
def guest_rows_of(run_name: str) -> int:
    with Session() as session:
        return len(_guest_log_ids(session, {"run_name": run_name}))


# what refuses a guest pass before it is queued, at the door and on an experiment's arm alike
def guest_pass_refusal(run_name: str, sample: int | None = None) -> tuple[int, str] | None:
    if not guests_available():
        return 409, "this runtime carries no `ragas`, the guest axes cannot be scored"
    # a typo in the name used to be a job over nothing
    owed = guest_rows_of(run_name)
    if not owed:
        return 404, f"run {run_name} owes no guest axis"
    # the pass walks the drawn subsample, so the cap is read over the same rows the handler counts
    will_walk = min(owed, sample) if sample else owed
    if will_walk > limits.MAX_GUEST_ROWS:
        return 400, f"{will_walk} rows would be walked, over the cap of {limits.MAX_GUEST_ROWS}"
    return None


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


# a row the pass stopped before is None, so the loop ends on the one cancel reading the row itself took
def _each(log_ids: list[int], width: int, one) -> int:
    if width == 1:
        done = 0
        for log_id in log_ids:
            got = one(log_id)
            if got is None:
                break
            done += bool(got)
        return done
    with ThreadPoolExecutor(max_workers=width) as pool:
        return sum(1 for got in pool.map(llm.carried(one), log_ids) if got)


# read, close, score, merge: the lock went and the session stayed open through minutes of calls
def _score_guests(log_id: int, stamp: dict, messages: str = guest_axes.MESSAGE_FORMS[0], *,
                  on_card: bool | None = None, model: str | None = None) -> bool:
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
            # our stamp plus the card read here: the stamp's `on_card` is about the pass
            scored[axis] = {**stamp, **guest_axes.score(axis, row, messages, model=model), "on_card_at_this_row": on_card}
            wrote = True
        except StandFault:
            raise
        except Exception as e:
            log.error("guest_axes.failed", axis=axis, log_id=log_id, error=str(e))
            scored[axis] = _errored_metric(metrics, axis, _error_text(e))
            # the same input overflows on every retry: the axis is dropped here, and the row says why
            if _chain_has(e, llm.InputOverWindow):
                scored[axis] |= {"attempts": _MAX_JUDGE_ATTEMPTS, "input_over_window": True}
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
    import job_specs

    # the next sweep is a new job: the previous one's retry counter is not its business
    carried = {k: v for k, v in options.items()
               if not k.startswith("_") and k not in job_specs.WORKER_KEYS}
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
    # which instrument minted the id: a reader cannot tell a queue guess from a server's own clock
    source: str | None = None


# a residency is read by the instrument its own engine has, and the engines have different ones
def _residency(job_id, model: str | None = None, role: str = "judging") -> Residency:
    try:
        judge = llm.resolve_for(role, model)
        engine = judge.engine
        if engine.placement is Placement.remote:
            return Residency(None, False, card.NO_RESIDENCY)
        if engine.kind is EngineKind.vllm:
            return _by_process_start(job_id, engine)
        on_card = judge_on_card(model, role)
        # partly on the card is a different instrument: layers on the cpu answer with other kernels
        if on_card is not True:
            return Residency(job_id, on_card, residency_instrument(engine))
        # `/api/ps` carries no load moment, so a residency is named by the pass that caused the load
        source = f"{residency_instrument(engine)} and the queue"
        last = _last_residency(engine.name)
        # `/api/ps` cannot say who loaded what between two passes, but the queue and the card can
        fresh = last is None or _loaded_since(last[0], job_id, judge)
        if fresh or _card_changed_hands(last[1], engine.name):
            return Residency(job_id, on_card, source)
        return Residency(last[0], on_card, source)
    except Exception as e:
        # a reading about the card must not take down the pass that only wanted to stamp itself
        log.warning("judge.residency_unknown", error=str(e))
        return Residency(None, None)


def _loaded_since(prev: int, job_id, judge) -> bool:
    from models.jobs import Job, JobStatus

    # `running` and the ones that died after loading evict the judge exactly as `done` ones do
    ours = {JobStatus.new}
    with Session() as session:
        asked = select(Job.type, Job.options).where(Job.id > prev, Job.status.notin_(ours))
        # `Job.id != None` renders as a no-op, and an ad hoc pass then never inherits a residency
        jobs = session.execute(asked if job_id is None else asked.where(Job.id != job_id)).all()
    seen = {}
    return any(evicts_the_judge(t, o or {}, judge, seen) for t, o in jobs)


# any other model on the card evicts: on the judge's engine it shares memory, on another it takes it
def evicts_the_judge(job_type: str, options: dict, judge, seen: dict | None = None) -> bool:
    import job_specs

    if job_type == "hand_card":
        return (options.get("engine_id"), options.get("model") or judge.name) != (
            judge.engine.id, judge.name
        )
    roles = job_specs.LOADS.get(job_type)
    if roles is None:
        return True
    overrides = {role: options.get(key) for role, key in job_specs.MODEL_OVERRIDES.items()}
    seen = {} if seen is None else seen
    for role in roles:
        key = (str(role), overrides.get(role))
        if key not in seen:
            seen[key] = _puts_another_model_on_the_card(*key, judge)
        if seen[key]:
            return True
    return False


def _puts_another_model_on_the_card(role: str, model: str | None, judge) -> bool:
    try:
        picked = llm.resolve_for(role, model)
    except StandFault:
        raise
    except Exception:
        # a role nobody can resolve now may have loaded anything when it ran
        return True
    if picked.engine.placement not in engines.CARD:
        return False
    return (picked.engine.id, picked.name) != (judge.engine.id, judge.name)


# one process holds one model, so passes under the same start share a residency whatever ran beside
def _by_process_start(job_id, engine) -> Residency:
    started = engines.started_at(engine)
    if started is None:
        # the id is the pass's own; the field names who minted it, and why the server did not
        return Residency(job_id, None, "the pass, vllm /metrics unreachable")
    last = _last_residency(engine.name, started)
    return Residency(job_id if last is None else last[0], None, "vllm /metrics process start")


# only this engine's rows: a vLLM pass once lent its number to the ollama pass that followed it
def _last_residency(engine_name: str, started_at: str | None = None) -> tuple[int, str] | None:
    from sqlalchemy import desc

    with Session() as session:
        for axis in rejudge.AXES:
            residency = QuestionLog.metrics[(axis, "residency_id")].as_integer()
            asked = select(residency, QuestionLog.metrics[(axis, "judged_at")].as_string()).where(
                residency.isnot(None),
                QuestionLog.metrics[(axis, "engine_name")].as_string() == engine_name,
            )
            if started_at is not None:
                started = QuestionLog.metrics[(axis, "engine_added", "started_at")].as_string()
                asked = asked.where(started == started_at)
            else:
                # a pass that found the judge on the cpu minted a number the card must not inherit
                asked = asked.where(QuestionLog.metrics[(axis, "on_card")].as_boolean().is_(True))
            # rows are created by the run and judged later, so the order that matters is the judging
            judged = QuestionLog.metrics[(axis, "judged_at")].as_string()
            latest = desc(cast(judged, DateTime(timezone=True))).nulls_last()
            got = session.execute(asked.order_by(latest).limit(1)).first()
            if got is not None:
                return got[0], got[1]
    return None


# one gpu engine holds the card at a time: a verdict from another since then means it changed hands
def _card_changed_hands(since: str | None, engine_name: str) -> bool:
    if since is None:
        # a row too old to say when it was judged cannot vouch that nothing came between
        return True
    holders = select(Engine.name).where(
        Engine.name != engine_name, Engine.placement.in_(engines.CARD)
    )
    when = DateTime(timezone=True)
    with Session() as session:
        for axis in rejudge.AXES:
            judged = cast(QuestionLog.metrics[(axis, "judged_at")].as_string(), when)
            other = QuestionLog.metrics[(axis, "engine_name")].as_string().in_(holders)
            later = select(QuestionLog.id).where(other, judged > cast(since, when)).limit(1)
            if session.scalar(later):
                return True
    return False


# once per row, not per axis: the width belongs to the pass and the sampler to the engine that answered
def stamp_of(width: int, residency: Residency | None = None, model: str | None = None, role: str = "judging") -> dict:
    from datetime import datetime, timezone

    seen = residency or Residency(None, None)
    # the bench can override the judge, and then the row was answered by that model's engine
    judged_by = llm.resolve_for(role, model)
    sampler = llm.sampler(role, judged_by)
    return {
        # at temperature zero this says the sampler took no part, not that the pass repeats
        "seed": sampler.sent.get("seed"),
        # what went out: a budget set on the model changed the judge, and the record did not say
        "sampler": sampler.sent,
        # what the role asked for and the engine would not carry: never read as applied
        "engine_refused": sampler.dropped,
        # and what the engine added on its own, read from the server: our vLLM start is not default
        "engine_added": engines.added_by(judged_by.engine, judged_by.name),
        "width": width,
        # two arms are comparable only inside one residency, and this is how a reader checks
        "residency_id": seen.id,
        "residency_source": seen.source,
        # a pass that found the judge half on the cpu is not the pass that found it whole
        "on_card": seen.on_card,
        # null with no instrument named is "nowhere to ask", not "asked and did not see it"
        "on_card_read_from": residency_instrument(judged_by.engine),
        # the address of the engine that answered, not of the process: they differed and it lied
        "engine": engines.address_of(judged_by.engine),
        # the address alone cannot say which engine: two of them can share a host and a port
        "engine_name": judged_by.engine.name,
        "judged_at": datetime.now(timezone.utc).isoformat(),
        # `/api/ps` does not report slots, so the record names the mirror it was capped by
        "slots_believed": _parallel_slots(),
    }


# on a refusal an axis does not apply, so both sides stay silent
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

    taken = {}
    for axis, (verdict_fn, args) in calls.items():
        if axis not in owed or (not force and _errored(metrics, axis)):
            continue
        taken[axis] = _run_axis(log_id, axis, verdict_fn, *args)
    # after the calls: probing before them read the card the generator had just taken
    stamp = stamp_of(width, _seen(residency), bench.model if bench else None)
    return _merge_our_scores(log_id, taken, skip, stamp, force)


def _seen(residency) -> Residency | None:
    return residency.get() if isinstance(residency, Pass) else residency


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
            snapshot = Snapshot(dict(ql.metrics), dict(ql.prompts), dict(ql.models))
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
    # only the downgrade it alone can make, by the operator `outcomes.classify` uses
    if said == Outcome.answered and not (score_of(ql.faithfulness) > 0):
        snapshot.metrics["settled_outcome"] = str(Outcome.answered_ungrounded)
        return
    # a settlement that only ever writes makes the outcome a fact of the first pass, not of today
    snapshot.metrics.pop("settled_outcome", None)


@dataclass
class Snapshot:
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


# ragas wraps what our client raised, so the cause is looked for down the chain
def _chain_has(e: BaseException | None, kind: type) -> bool:
    seen = set()
    while e is not None and id(e) not in seen:
        if isinstance(e, kind):
            return True
        seen.add(id(e))
        e = e.__cause__ or e.__context__
    return False


# the kind and what it said, with known keys cut: a row that failed three times left only an exception name
def _error_text(e: Exception) -> str:
    return f"{type(e).__name__}: {redact(str(e))}"


def _errored_metric(metrics: dict, axis: str, err: str) -> dict:
    was = metrics.get(axis) or {}
    # a row judged again is no longer skipped, and both marks at once read as neither
    was = {k: v for k, v in was.items() if k != "skipped"}
    return {**was, "error": err[:_ERROR_CHARS], "attempts": was.get("attempts", 0) + 1}


def _run_axis(log_id, axis, verdict_fn, *args):
    try:
        return verdict_fn(*args), None
    except StandFault:
        raise
    except Exception as e:
        log.error("judge.axis_failed", axis=axis, log_id=log_id, error=str(e))
        return None, _error_text(e)


# a fan-out changes verdicts and a seed makes a pass repeatable
def _axis_metric(verdict, stamp: dict | None = None) -> dict:
    return {
        "reason": verdict.reason,
        "elapsed": verdict.elapsed,
        "model": verdict.model,
        # named for the judge, because the row already carries the answering call's count
        token_fields.JUDGE_PROMPT: verdict.prompt_tokens,
        # a thinking judge on a broker pays for its output, and the output is where it thinks
        token_fields.JUDGE_COMPLETION: getattr(verdict, "completion_tokens", None),
        **({"judge_parser": verdict.parser} if getattr(verdict, "parser", None) else {}),
        **({"judge_answer_parse": verdict.answer_parse} if getattr(verdict, "answer_parse", None) else {}),
        **({token_fields.JUDGE_CUT: True} if getattr(verdict, "cut_by_length", False) else {}),
        **(stamp or {}),
    }
