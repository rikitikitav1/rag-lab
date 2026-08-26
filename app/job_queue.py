from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from models import Job, JobStatus
from orm.sync_db import Session
from sqlalchemy import func, select


@dataclass
class ClaimedJob:
    id: int
    type: str
    options: dict


def enqueue(type: str, options: dict | None = None, queue: str = "default") -> int:
    with Session() as session:
        job = Job(type=type, options=options or {}, queue=queue)
        session.add(job)
        session.commit()
        return job.id


# a second identical job is not idempotence, it is a queue nobody reads
def pending_of_type(type: str) -> bool:
    with Session() as session:
        return bool(
            session.scalar(
                select(Job.id).where(
                    Job.type == type,
                    Job.status.in_([JobStatus.new, JobStatus.running]),
                ).limit(1)
            )
        )


def add_job(
    session, type: str, options: dict | None = None, queue: str = "default"
) -> Job:
    # stage a job in the caller's transaction (caller commits); async-safe: .add() is sync
    job = Job(type=type, options=options or {}, queue=queue)
    session.add(job)
    return job


def claim_next(queues: list[str]) -> ClaimedJob | None:
    with Session() as session:
        job = session.scalars(
            select(Job)
            .where(
                Job.status == JobStatus.new,
                Job.queue.in_(queues),
                Job.apply_since <= func.now(),
            )
            .order_by(Job.apply_since)
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


def cancel(ids: list[int]) -> list[int]:
    if not ids:
        return []
    with Session() as session:
        jobs = session.scalars(
            select(Job).where(
                Job.id.in_(ids),
                Job.status.in_([JobStatus.new, JobStatus.running]),
            )
        ).all()
        cancelled = [j.id for j in jobs]
        for job in jobs:
            job.status = JobStatus.cancelled
        session.commit()
        return cancelled


def is_cancelled(id: int) -> bool:
    with Session() as session:
        job = session.get(Job, id)
        return job is not None and job.status == JobStatus.cancelled


def _update(id: int, **fields) -> None:
    with Session() as session:
        job = session.get(Job, id)
        if job is None or job.status == JobStatus.cancelled:
            return
        for key, value in fields.items():
            setattr(job, key, value)
        session.commit()
