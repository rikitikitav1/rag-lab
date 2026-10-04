from collections import defaultdict
from datetime import UTC, datetime, timedelta

import job_queue
from models.jobs import Job, JobStatus
from orm.sync_db import Session
from sqlalchemy import func, select

# the mean a type's waiting jobs are priced at is read over this much of its history
MEAN_OVER = timedelta(days=1)
_SOURCED = ("source", "run_name", "set_name")


def _named(options: dict | None) -> str | None:
    options = options or {}
    return next((str(options[k]) for k in _SOURCED if options.get(k)), None)


# the queue in one screen: what waits by type, what runs and for how long, what finished lately, and when it all ends
def stats(window_minutes: int = 60) -> dict:
    now = datetime.now(UTC)
    since = now - timedelta(minutes=window_minutes)
    with Session() as session:
        waiting: dict[str, int] = dict(session.execute(
            select(Job.type, func.count()).where(Job.status == JobStatus.new).group_by(Job.type)
        ).all())
        # inside the waiting: jobs backing off a retry or a deferral, which a backed-up queue is not
        deferred: dict[str, int] = dict(session.execute(
            select(Job.type, func.count()).where(Job.status == JobStatus.new, Job.apply_since > now)
            .group_by(Job.type)
        ).all())
        paused: dict[str, int] = dict(session.execute(
            select(Job.type, func.count()).where(Job.status == JobStatus.paused).group_by(Job.type)
        ).all())
        running = [
            {"id": j.id, "type": j.type, "of": _named(j.options),
             "running_s": round((now - j.updated_at.replace(tzinfo=UTC)).total_seconds())}
            for j in session.scalars(select(Job).where(Job.status == JobStatus.running).order_by(Job.id))
        ]
        finished: dict[str, dict[str, int]] = defaultdict(dict)
        for kind, status, n in session.execute(
            select(Job.type, Job.status, func.count())
            .where(Job.status.not_in(job_queue.ACTIVE), Job.updated_at >= since.replace(tzinfo=None))
            .group_by(Job.type, Job.status)
        ):
            finished[kind][str(status)] = n
        means = {
            kind: (round(mean), n)
            for kind, mean, n in session.execute(
                select(Job.type, func.avg(Job.elapsed), func.count())
                .where(Job.status == JobStatus.done, Job.elapsed.is_not(None),
                       Job.updated_at >= (now - MEAN_OVER).replace(tzinfo=None))
                .group_by(Job.type)
            )
        }
    priced = {kind: n * means[kind][0] for kind, n in waiting.items() if kind in means}
    return {
        "at": now.isoformat(timespec="minutes"),
        "waiting": dict(sorted(waiting.items(), key=lambda kv: -kv[1])),
        "deferred": deferred,
        "paused": paused,
        "running": running,
        f"finished_last_{window_minutes}_min": dict(finished),
        "mean_seconds": {kind: {"mean": m, "n": n} for kind, (m, n) in sorted(means.items())},
        # the waiting jobs priced at their type's mean of the last day; a type with no history is named, not guessed
        "eta_seconds": sum(priced.values()),
        "eta_by_type": dict(sorted(priced.items(), key=lambda kv: -kv[1])),
        "unpriced": sorted(kind for kind in waiting if kind not in means),
    }
