import asyncio
import json
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import corpus_search
import job_specs
import logging_setup
import token_fields
import version
from models import FailKind, Job, JobStatus
from models.experiment import fail_running_on
from models.prereg import preregistered
from orm.sync_db import Session
from search_scope import ScopeRefused
from sqlalchemy import case, func, select, text, update

log = logging_setup.get_logger(__name__)

# the job the worker is running in this context: what its handler queues names it as the parent
running_job: ContextVar[int | None] = ContextVar("running_job", default=None)


@dataclass
class ClaimedJob:
    id: int
    type: str
    options: dict


# the lane is a property of the type: left to the caller, a card job reached the io lane by hand
def enqueue(type: str, options: dict | None = None, queue: str | None = None, parent_id: int | None = None) -> int:
    with Session() as session:
        job = _checked(type, options, queue, parent_id)
        session.add(job)
        session.commit()
        return job.id


# the spec's own check, then the search's step as at the MCP door: sources out of search, a version none holds
def _check(type: str, options: dict | None) -> None:
    checked = job_specs.check(type, options)
    if checked is None or not hasattr(checked, "scope") or not checked.scope().narrowed:
        return
    try:
        corpus_search.refuse_bad_scope(checked.scope(), getattr(checked, "variant", None))
    except ScopeRefused as bad:
        raise job_specs.Refused(f"scope: {bad}") from bad


# the spec checks that a closing run names a promise; only the base can say the promise exists
def _promise(options: dict | None) -> str | None:
    name = (options or {}).get("prereg")
    if name and not _preregistered(name):
        raise job_specs.Refused(f"prereg: no preregistration named {name!r}")
    return name or None


# one lane for the card: a caller naming another lane would let two card jobs run at once
def _lane(type: str, asked: str | None) -> str:
    lane = job_specs.lane(type)
    if asked is not None and asked != lane:
        raise ValueError(f"{type} lives in the {lane} lane, not in {asked}")
    return lane


# options matter as well as the type; the id, so a door answers with it rather than queue a second
def pending_of_type(type: str, **options) -> int | None:
    with Session() as session:
        query = select(Job.id).where(
            Job.type == type,
            Job.status.in_(ACTIVE),
        )
        for key, value in options.items():
            # a json null renders as "None" through astext, so two nulls would look like two jobs
            if value is None:
                query = query.where(Job.options[key].astext.is_(None))
            else:
                query = query.where(Job.options[key].astext == str(value))
        return session.scalar(query.limit(1))


# a waiting or running job whose list option holds the value, as a veto build over several variants
def pending_listing(type: str, key: str, value: str) -> int | None:
    with Session() as session:
        query = select(Job.id).where(
            Job.type == type, Job.status.in_(ACTIVE), Job.options[key].contains([value])
        )
        return session.scalar(query.limit(1))


def running_of_type(type: str) -> bool:
    with Session() as session:
        return bool(session.scalar(
            select(Job.id).where(Job.type == type, Job.status == JobStatus.running).limit(1)
        ))


# one rule for a handover already waiting: the same engine, the same model if named, the same seat
def pending_handover(engine_id: int, model: str | None = None, seat: str | None = None) -> int | None:
    asked = {"engine_id": engine_id, "model": model, "seat": seat}
    return pending_of_type("hand_card", **{k: v for k, v in asked.items() if v is not None})


# the card's lane: a job running there may hold the card, and nothing from outside takes it away
def running_in_lane(lane: str) -> bool:
    with Session() as session:
        return bool(session.scalar(
            select(Job.id).where(Job.queue == lane, Job.status == JobStatus.running).limit(1)
        ))


# the row a door answers with once the job is queued in another session
def get(job_id: int) -> Job:
    with Session() as session:
        return session.get(Job, job_id)


# stage a job in the caller's async transaction (caller commits)
async def add_job(
    session, type: str, options: dict | None = None, queue: str | None = None, parent_id: int | None = None
) -> Job:
    job = await prepared(type, options, queue, parent_id)
    session.add(job)
    return job


def _preregistered(name: str) -> bool:
    with Session() as session:
        return preregistered(session, name)


# a checked job not yet in any session; the checks read the base, so they run off the loop
async def prepared(
    type: str, options: dict | None = None, queue: str | None = None, parent_id: int | None = None
) -> Job:
    return await asyncio.to_thread(_checked, type, options, queue, parent_id)


def _checked(type: str, options: dict | None, queue: str | None, parent_id: int | None = None) -> Job:
    _check(type, options)
    return Job(type=type, options=options or {}, queue=_lane(type, queue), prereg=_promise(options),
               parent_id=parent_id if parent_id is not None else running_job.get())


# what a job would be queued as, with nothing queued: the checks, the options as the handler reads them, the lane
def dry_run(type: str, options: dict | None = None, queue: str | None = None) -> dict:
    checked = job_specs.check(type, options)
    _check(type, options)
    return {"type": type, "lane": _lane(type, queue),
            "options": checked.model_dump(exclude_unset=True) if checked is not None else options or {}}


# a job takes the card in its own turn, so the turn is the job's type, and an old job goes early
def _turn():
    ranked = case(job_specs.PRIORITY, value=Job.type, else_=0)
    starved = Job.created_at < func.now() - text(
        f"interval '{job_specs.STARVED_AFTER_MINUTES} minutes'"
    )
    # raised, never lowered: a starved API handover keeps its own place ahead
    return case((starved, func.least(ranked, -1)), else_=ranked)


def claim_next(queues: list[str]) -> ClaimedJob | None:
    with Session() as session:
        job = session.scalars(
            select(Job)
            .where(
                Job.status == JobStatus.new,
                Job.queue.in_(queues),
                Job.apply_since <= func.now(),
            )
            .order_by(_turn(), Job.apply_since)
            .with_for_update(skip_locked=True)
            .limit(1)
        ).first()
        if job is None:
            return None
        job.status = JobStatus.running
        job.code = version.mine()
        claimed = ClaimedJob(id=job.id, type=job.type, options=dict(job.options))
        session.commit()
        return claimed


# a job the worker died under this many times kills it again on every claim: it fails instead of holding its lane
MAX_RECLAIMS = 5


def requeue_stale(queues: list[str]) -> list[int]:
    with Session() as session:
        jobs = session.scalars(
            select(Job).where(Job.status == JobStatus.running, Job.queue.in_(queues))
        ).all()
        ids, died = [], []
        for job in jobs:
            job.options = reclaimed(job.options)
            if job.options["reclaims"] > MAX_RECLAIMS:
                job.status = JobStatus.error
                job.error = {"error": f"the worker stopped under this job {MAX_RECLAIMS + 1} times in a row",
                             "kind": FailKind.worker_died, "attempts": job.options["attempts"]}
                died.append(job.id)
                continue
            job.status = JobStatus.new
            ids.append(job.id)
        session.commit()
    announce_finished(died)
    return ids


# a restart is an attempt: without the mark a run met its own rows and refused itself as taken
def reclaimed(options: dict) -> dict:
    return {**options, "attempts": options.get("attempts", 0) + 1, "reclaims": options.get("reclaims", 0) + 1}


def complete(id: int, elapsed: float | None = None, result: dict | None = None) -> None:
    fields = {"status": JobStatus.done}
    if elapsed is not None:
        fields["elapsed"] = elapsed
    if result:
        fields["result"] = result
    _update(id, **fields)


def fail(id: int, error: dict, kind: FailKind, elapsed: float | None = None) -> None:
    fields = {"status": JobStatus.error, "error": {**error, "kind": kind}}
    if elapsed is not None:
        fields["elapsed"] = elapsed
    _update(id, **fields)


# added to earlier attempts, on a cancelled job too; one that spent nothing writes {}, null is before the count
def add_tokens(id: int, record: dict | None) -> None:
    with Session() as session:
        job = session.get(Job, id)
        if job is None:
            return
        job.tokens = merged_tokens(job.tokens, record or {})
        session.commit()


def add_balances(id: int, before: dict, after: dict) -> None:
    with Session() as session:
        job = session.get(Job, id)
        if job is None:
            return
        job.balances = merged_balances(job.balances, before, after)
        session.commit()


# the first attempt's before and the last one's after: what the whole job took off the key, retries included
def merged_balances(was: dict | None, before: dict, after: dict) -> dict:
    out = {name: dict(entry) for name, entry in (was or {}).items()}
    for name, seen in after.items():
        held = out.setdefault(name, {})
        why = None
        if "before" not in held:
            start = before.get(name)
            held.update(before=(start or {}).get("balance"), before_at=(start or {}).get("read_at"))
            why = start.get("why") if start else "not read before the attempt"
        held.update(after=seen.get("balance"), after_at=seen.get("read_at"), unit=seen.get("unit") or held.get("unit"))
        held["why"] = seen.get("why") or why or held.get("why")
    return out


def merged_tokens(was: dict | None, more: dict) -> dict:
    out = {role: [dict(entry) for entry in entries] for role, entries in (was or {}).items()}
    for role, entries in more.items():
        held = out.setdefault(role, [])
        for entry in entries:
            same = next((e for e in held if (e["engine"], e["model"]) == (entry["engine"], entry["model"])), None)
            if same is None:
                held.append(dict(entry))
                continue
            for key in token_fields.SUMMED:
                if key in entry or key in same:
                    same[key] = same.get(key, 0) + entry.get(key, 0)
            for key in token_fields.MAXED:
                if key in entry or key in same:
                    same[key] = max(same.get(key, 0), entry.get(key, 0))
    return out


# the reason rides in the options beside the attempts, so a job waiting out a backoff says why on its row
def reschedule(
    id: int, options: dict, delay: timedelta, elapsed: float | None = None, because: str | None = None,
    kind: str = "retry",
) -> None:
    retry_at = datetime.now(timezone.utc) + delay
    if because is not None:
        options = {**options, "waiting_because": {"kind": kind, "error": because, "retry_at": retry_at.isoformat()}}
    fields = {
        "status": JobStatus.new,
        "options": options,
        "apply_since": retry_at,
    }
    if elapsed is not None:
        fields["elapsed"] = elapsed
    _update(id, **fields)


# a job that has not finished: waiting, held or running, the states a cancel and every guard act on
ACTIVE = (JobStatus.new, JobStatus.paused, JobStatus.running)
FINISHED = (JobStatus.done, JobStatus.error, JobStatus.cancelled)
FINISHED_CHANNEL = "job_finished"

# a dead answerer and a dead judge strand the experiment the same way
EXPERIMENT_JOBS = ("eval_run", "judge_answers")


# the live jobs a bulk door acts on; every filter narrows, and none given means every live job
def live_ids(statuses=None, type: str | None = None, run_name: str | None = None, ids=None,
             parent_id: int | None = None) -> list[int]:
    stmt = select(Job.id).where(Job.status.in_([s for s in ACTIVE if not statuses or s in statuses]))
    if type:
        stmt = stmt.where(Job.type == type)
    if run_name:
        stmt = stmt.where(Job.options["run_name"].astext == run_name)
    if ids:
        stmt = stmt.where(Job.id.in_(ids))
    if parent_id is not None:
        stmt = stmt.where(Job.parent_id == parent_id)
    with Session() as session:
        return list(session.scalars(stmt.order_by(Job.id)))


# what a cancel of these ids would take: the ids themselves plus the judges of their runs
def cancel_reach(ids: list[int]) -> list[int]:
    with Session() as session:
        jobs = session.scalars(select(Job).where(Job.id.in_(ids), Job.status.in_(ACTIVE))).all()
        return sorted({j.id for j in jobs} | set(_their_judges(session, jobs)))


# a held job keeps its id, its place by apply_since and its options; only a waiting one can be held
def pause(ids: list[int]) -> list[int]:
    return _move(ids, JobStatus.new, JobStatus.paused)


def resume(ids: list[int]) -> list[int]:
    return _move(ids, JobStatus.paused, JobStatus.new)


def _move(ids: list[int], was: JobStatus, to: JobStatus) -> list[int]:
    if not ids:
        return []
    with Session() as session:
        # one statement: a job the worker claims between a read and a write is not held while it runs
        moved = session.scalars(
            update(Job).where(Job.id.in_(ids), Job.status == was).values(status=to).returning(Job.id)
        ).all()
        session.commit()
    return sorted(moved)


def cancel_with_its_judge(job_id: int) -> list[int]:
    return cancel([job_id])


# a run's judge is cancelled with the job it waits on, or it waits for rows never coming
def _their_judges(session, jobs) -> list[int]:
    runs = [
        (j.options or {}).get("run_name")
        for j in jobs
        if j.type == "eval_run" and (j.options or {}).get("run_name")
    ]
    if not runs:
        return []
    return list(
        session.scalars(
            select(Job.id).where(
                Job.type == "judge_answers",
                Job.status.in_(ACTIVE),
                Job.options["run_name"].astext.in_(runs),
            )
        )
    )


def cancel(ids: list[int]) -> list[int]:
    if not ids:
        return []
    with Session() as session:
        jobs = session.scalars(
            select(Job).where(Job.id.in_(ids), Job.status.in_(ACTIVE))
        ).all()
        judges = _their_judges(session, jobs)
        if judges:
            jobs = session.scalars(
                select(Job).where(
                    Job.id.in_({*ids, *judges}), Job.status.in_(ACTIVE)
                )
            ).all()
        cancelled = [j.id for j in jobs]
        # a running one is announced by the worker when its handler returns
        unclaimed = [j.id for j in jobs if j.status != JobStatus.running]
        stranded = [
            (j.options or {}).get("run_name")
            for j in jobs
            if j.type in EXPERIMENT_JOBS and (j.options or {}).get("run_name")
        ]
        for job in jobs:
            if job.status != JobStatus.running and job.tokens is None:
                job.tokens = {}
            job.status = JobStatus.cancelled
        # under a savepoint: a failed update leaves the cancel standing, and an experiment waiting on it would wait
        failed = []
        for run_name in stranded:
            try:
                with session.begin_nested():
                    if fail_running_on(session, run_name):
                        failed.append(run_name)
            except Exception as e:
                log.warning("job.experiment_not_failed", run_name=run_name, error=str(e))
        session.commit()
    for run_name in failed:
        log.warning("experiment.failed", run_name=run_name)
    announce_finished(unclaimed)
    return cancelled


# a listener on the channel learns of a finished job at once; with nobody listening the notice is dropped
def announce_finished(ids: list[int]) -> None:
    if not ids:
        return
    with Session() as session:
        for job in session.scalars(select(Job).where(Job.id.in_(ids), Job.status.in_(FINISHED))):
            payload = {"id": job.id, "type": job.type, "status": job.status.value,
                       "run_name": (job.options or {}).get("run_name")}
            session.execute(text("SELECT pg_notify(:channel, :payload)"),
                            {"channel": FINISHED_CHANNEL, "payload": json.dumps(payload)})
        session.commit()


def is_cancelled(id: int) -> bool:
    with Session() as session:
        job = session.get(Job, id)
        return job is not None and job.status == JobStatus.cancelled


def _update(id: int, **fields) -> None:
    with Session() as session:
        job = session.get(Job, id)
        if job is None:
            return
        # a cancel is final, but how long the job ran before it is still the job's
        if job.status == JobStatus.cancelled:
            fields = {k: v for k, v in fields.items() if k == "elapsed"}
        for key, value in fields.items():
            setattr(job, key, value)
        session.commit()


# the judge wakes once per batch of live answers, and the batch waits this long for company
LIVE_BATCH_SECONDS = 300


# appended only while nobody has taken the job: a running one read its rows when it was claimed
def judge_live(log_id: int) -> int:
    with Session() as session:
        waiting = session.execute(text("""
            UPDATE jobs
            SET options = jsonb_set(options, '{log_ids}', (options->'log_ids') || to_jsonb(:log_id))
            WHERE id = (
                SELECT id FROM jobs
                WHERE type = 'judge_answers' AND status = 'new' AND options->>'live' = 'true'
                ORDER BY id LIMIT 1 FOR UPDATE SKIP LOCKED
            )
            RETURNING id
        """), {"log_id": log_id}).first()
        if waiting is not None:
            session.commit()
            return waiting[0]
        options = {"log_ids": [log_id], "live": True}
        job_specs.check("judge_answers", options)
        job = Job(
            type="judge_answers", options=options, queue=_lane("judge_answers", None),
            apply_since=datetime.now(timezone.utc) + timedelta(seconds=LIVE_BATCH_SECONDS),
        )
        session.add(job)
        session.commit()
        return job.id
