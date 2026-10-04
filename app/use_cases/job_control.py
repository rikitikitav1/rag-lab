from dataclasses import dataclass, field

import job_queue
import job_specs
from errors import Refusal, RefusalKind
from models import JobStatus


@dataclass
class Scope:
    run_name: str | None = None
    type: str | None = None
    ids: list[int] = field(default_factory=list)
    parent_id: int | None = None
    statuses: list[JobStatus] = field(default_factory=list)
    # a type with no run name, ids or parent is every live job of that kind, which is a thing to say out loud
    every: bool = False


def _picked(scope: Scope) -> list[int]:
    if not (scope.run_name or scope.type or scope.ids or scope.parent_id is not None):
        raise Refusal(RefusalKind.invalid, "name a run_name, a type, ids or a parent_id")
    if scope.every and (scope.run_name or scope.ids or scope.parent_id is not None):
        raise Refusal(RefusalKind.invalid, "every=true widens a scope to a whole job type, so it takes no narrower one")
    narrowed = scope.run_name or scope.ids or scope.parent_id is not None
    if scope.type and not narrowed and not scope.every:
        raise Refusal(RefusalKind.invalid, f"type '{scope.type}' alone reaches every live job of that kind:"
                                           " narrow it, or pass every=true and mean it")
    return job_queue.live_ids(scope.statuses, scope.type, scope.run_name, scope.ids, scope.parent_id)


# through the queue: a cancel fails the stranded experiment and takes each run's judge along
def cancel(scope: Scope, dry_run: bool = False) -> dict:
    ids = _picked(scope)
    if dry_run:
        return {"would_cancel": job_queue.cancel_reach(ids)}
    return {"cancelled": job_queue.cancel(ids)}


# only waiting jobs are held, so a scope over running ones holds none of them
def pause(scope: Scope, dry_run: bool = False) -> dict:
    ids = job_queue.live_ids([JobStatus.new], *_narrow(scope))
    return {"would_pause": ids} if dry_run else {"paused": job_queue.pause(ids)}


def resume(scope: Scope, dry_run: bool = False) -> dict:
    ids = job_queue.live_ids([JobStatus.paused], *_narrow(scope))
    return {"would_resume": ids} if dry_run else {"resumed": job_queue.resume(ids)}


def _narrow(scope: Scope) -> tuple:
    _picked(scope)
    return scope.type, scope.run_name, scope.ids, scope.parent_id


# the door an agent queues by: refused the same way as REST, and a dry run queues nothing
def enqueue(type: str, options: dict | None = None, dry_run: bool = False) -> dict:
    if type not in job_specs.SPECS and type not in job_specs.FREE:
        raise Refusal(RefusalKind.invalid, f"no such job type: {type}")
    if type == "eval_run":
        raise Refusal(RefusalKind.invalid, "an eval_run is queued by enqueue_eval_run, which runs its base checks")
    try:
        if dry_run:
            return {"dry_run": job_queue.dry_run(type, options)}
        return {"job_id": job_queue.enqueue(type, options)}
    except job_specs.Refused as bad:
        raise Refusal(RefusalKind.malformed, str(bad)) from bad


# the run's name and question checks read the base the async way, as at POST /v1/job
async def enqueue_eval_run(options: dict, dry_run: bool = False) -> dict:
    from orm.async_db import session_factory
    from use_cases.eval_runs import queued_eval_run

    try:
        async with session_factory() as session:
            job = await queued_eval_run(session, options)
            if dry_run:
                return {"dry_run": job_queue.dry_run("eval_run", job.options)}
            session.add(job)
            await session.commit()
            return {"job_id": job.id}
    except job_specs.Refused as bad:
        raise Refusal(RefusalKind.malformed, str(bad)) from bad
