from datetime import datetime

import job_queue
import job_specs
from crud import get_or_404
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.concurrency import run_in_threadpool
from models.jobs import Job, JobStatus
from orm.async_db import get_session
from pydantic import BaseModel, computed_field
from query_utils import (
    Page,
    apply_created_between,
    apply_in_filters,
    apply_sort_limit_offset,
)
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

router = APIRouter(prefix="/job", tags=["jobs"])


class JobResponse(BaseModel):
    id: int
    type: str
    queue: str
    status: JobStatus
    options: dict
    error: dict | None
    elapsed: float | None
    tokens: dict | None = None
    balances: dict | None = None
    code: dict | None = None
    result: dict | None = None
    parent_id: int | None = None
    apply_since: datetime
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}

    # the field every door that queues a job answers with, so a client reads the id one way
    @computed_field
    @property
    def job_id(self) -> int:
        return self.id


SORT_MAP = {
    "id": Job.id,
    "created_at": Job.created_at,
    "updated_at": Job.updated_at,
    "elapsed": Job.elapsed,
}


@router.get("", response_model=list[JobResponse])
async def list_jobs(
    type: list[str] | None = Query(default=None),
    status: list[JobStatus] | None = Query(default=None),
    created_from: datetime | None = Query(default=None),
    created_to: datetime | None = Query(default=None),
    run_name: str | None = Query(default=None),
    parent_id: int | None = Query(default=None),
    page: Page = Depends(),
    session: AsyncSession = Depends(get_session),
):
    stmt = apply_in_filters(select(Job), {Job.type: type, Job.status: status})
    stmt = apply_created_between(stmt, Job.created_at, created_from, created_to)
    if run_name:
        stmt = stmt.where(Job.options["run_name"].astext == run_name)
    if parent_id is not None:
        stmt = stmt.where(Job.parent_id == parent_id)

    stmt = apply_sort_limit_offset(
        stmt=stmt,
        sort_map=SORT_MAP,
        sort_by=page.sort_by,
        sort_order=page.sort_order,
        limit=page.limit,
        offset=page.offset,
    )

    result = await session.scalars(stmt)
    return result.all()


class EnqueueRequest(BaseModel):
    type: str
    options: dict = {}


# one door for every type: a door of its own is for work done before the enqueue, not for checking
@router.post("", response_model=JobResponse, status_code=201)
async def enqueue_job(
    request: EnqueueRequest,
    session: AsyncSession = Depends(get_session),
):
    if request.type not in job_specs.SPECS and request.type not in job_specs.FREE:
        raise HTTPException(status_code=400, detail=f"no such job type: {request.type}")
    options = request.options
    if request.type == "eval_run" and options.get("resume"):
        from use_cases.eval_runs import resumed_options

        extra = sorted(set(options) - {"run_name", "resume"})
        options = await resumed_options(session, options.get("run_name"), extra)
    # checked once, off the loop, before the name lookup; a refusal answers 400 through the app's handler
    job = await job_queue.prepared(request.type, options)
    if request.type == "eval_run" and not options.get("resume"):
        from use_cases.eval_runs import refuse_a_taken_run, refuse_missing_questions

        if options.get("run_name"):
            await refuse_a_taken_run(session, options["run_name"])
        # the run's own door refuses ids the stand lacks, and this one queues the same job
        if options.get("question_ids"):
            await refuse_missing_questions(session, options["question_ids"])
    session.add(job)
    await session.commit()
    await session.refresh(job)
    return job


# the queue in one screen: waiting by type, running, finished in the window, and the waiting priced at each type's mean
@router.get("/stats")
async def queue_stats(window_minutes: int = Query(default=60, ge=1, le=1440)) -> dict:
    from use_cases import queue_stats

    return await run_in_threadpool(queue_stats.stats, window_minutes)


@router.get("/{id}", response_model=JobResponse)
async def show_job(id: int, session: AsyncSession = Depends(get_session)):
    return await get_or_404(Job, id, session)


class CancelResponse(BaseModel):
    cancelled: list[int]


class ScopeRequest(BaseModel):
    run_name: str | None = None
    type: str | None = None
    ids: list[int] = []
    parent_id: int | None = None
    # narrows a cancel to waiting or running jobs; a pause reads only waiting ones and a resume only held ones
    status: list[JobStatus] = []
    # a type with no run name, ids or parent is every live job of that kind, which is a thing to say out loud
    every: bool = False
    dry_run: bool = False

    def scope(self):
        from use_cases.job_control import Scope

        return Scope(run_name=self.run_name, type=self.type, ids=self.ids, parent_id=self.parent_id,
                     statuses=self.status, every=self.every)


@router.post("/cancel")
async def cancel_jobs(request: ScopeRequest) -> dict:
    from use_cases import job_control

    return await run_in_threadpool(job_control.cancel, request.scope(), request.dry_run)


# a held job keeps its id and its place, and the worker passes it by until it is resumed
@router.post("/pause")
async def pause_jobs(request: ScopeRequest) -> dict:
    from use_cases import job_control

    return await run_in_threadpool(job_control.pause, request.scope(), request.dry_run)


@router.post("/resume")
async def resume_jobs(request: ScopeRequest) -> dict:
    from use_cases import job_control

    return await run_in_threadpool(job_control.resume, request.scope(), request.dry_run)


@router.post("/{id}/cancel", response_model=CancelResponse)
async def cancel_job(id: int, session: AsyncSession = Depends(get_session)):
    await get_or_404(Job, id, session)
    cancelled = await run_in_threadpool(job_queue.cancel_with_its_judge, id)
    return CancelResponse(cancelled=cancelled)
