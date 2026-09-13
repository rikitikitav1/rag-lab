from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import job_specs
import logging_setup
import token_fields
from models import Job, JobStatus
from orm.sync_db import Session
from sqlalchemy import case, func, select, text

log = logging_setup.get_logger(__name__)


@dataclass
class ClaimedJob:
    id: int
    type: str
    options: dict


# the lane is a property of the type: left to the caller, a card job reached the io lane by hand
def enqueue(type: str, options: dict | None = None, queue: str | None = None) -> int:
    job_specs.check(type, options)
    with Session() as session:
        job = Job(type=type, options=options or {}, queue=_lane(type, queue))
        session.add(job)
        session.commit()
        return job.id


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
            Job.status.in_([JobStatus.new, JobStatus.running]),
        )
        for key, value in options.items():
            # a json null renders as "None" through astext, so two nulls would look like two jobs
            if value is None:
                query = query.where(Job.options[key].astext.is_(None))
            else:
                query = query.where(Job.options[key].astext == str(value))
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


def add_job(
    session, type: str, options: dict | None = None, queue: str | None = None
) -> Job:
    # stage a job in the caller's transaction (caller commits); async-safe: .add() is sync
    job_specs.check(type, options)
    job = Job(type=type, options=options or {}, queue=_lane(type, queue))
    session.add(job)
    return job


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
        claimed = ClaimedJob(id=job.id, type=job.type, options=dict(job.options))
        session.commit()
        return claimed


def requeue_stale(queues: list[str]) -> list[int]:
    with Session() as session:
        jobs = session.scalars(
            select(Job).where(Job.status == JobStatus.running, Job.queue.in_(queues))
        ).all()
        ids = [job.id for job in jobs]
        for job in jobs:
            job.status = JobStatus.new
        session.commit()
        return ids


def complete(id: int, elapsed: float | None = None) -> None:
    fields = {"status": JobStatus.done}
    if elapsed is not None:
        fields["elapsed"] = elapsed
    _update(id, **fields)


def fail(id: int, error: dict, elapsed: float | None = None) -> None:
    fields = {"status": JobStatus.error, "error": error}
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


def reschedule(
    id: int, options: dict, delay: timedelta, elapsed: float | None = None
) -> None:
    fields = {
        "status": JobStatus.new,
        "options": options,
        "apply_since": datetime.now(timezone.utc) + delay,
    }
    if elapsed is not None:
        fields["elapsed"] = elapsed
    _update(id, **fields)


# a job that has not run yet or is running: the only two states a cancellation can act on
ACTIVE = (JobStatus.new, JobStatus.running)

# a dead answerer and a dead judge strand the experiment the same way
EXPERIMENT_JOBS = ("eval_run", "judge_answers")


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
        stranded = [
            (j.options or {}).get("run_name")
            for j in jobs
            if j.type in EXPERIMENT_JOBS and (j.options or {}).get("run_name")
        ]
        for job in jobs:
            if job.status == JobStatus.new and job.tokens is None:
                job.tokens = {}
            job.status = JobStatus.cancelled
        session.commit()
    # after the commit: an experiment waiting on a cancelled arm waits for ever
    for run_name in stranded:
        try:
            from use_cases import experiment

            experiment.mark_failed_for_run(run_name)
        except Exception as e:
            log.warning("job.experiment_not_failed", run_name=run_name, error=str(e))
    return cancelled


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
