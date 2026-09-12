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
    page: Page = Depends(),
    session: AsyncSession = Depends(get_session),
):
    stmt = apply_in_filters(select(Job), {Job.type: type, Job.status: status})
    stmt = apply_created_between(stmt, Job.created_at, created_from, created_to)
    if run_name:
        stmt = stmt.where(Job.options["run_name"].astext == run_name)

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
    try:
        job_specs.check(request.type, request.options)
    except job_specs.Refused as bad:
        raise HTTPException(status_code=400, detail=str(bad)) from bad

    job = job_queue.add_job(session, request.type, request.options)
    await session.commit()
    await session.refresh(job)
    return job


@router.get("/{id}", response_model=JobResponse)
async def show_job(id: int, session: AsyncSession = Depends(get_session)):
    return await get_or_404(Job, id, session)


class CancelResponse(BaseModel):
    cancelled: list[int]


class BulkCancelRequest(BaseModel):
    run_name: str | None = None
    type: str | None = None
    # a type with no run name is every live job of that kind, which is a thing to say out loud
    every: bool = False


@router.post("/cancel", response_model=CancelResponse)
async def cancel_jobs(
    request: BulkCancelRequest,
    session: AsyncSession = Depends(get_session),
):
    if not request.run_name and not request.type:
        raise HTTPException(status_code=400, detail="run_name or type is required")
    if request.every and request.run_name:
        raise HTTPException(
            status_code=400,
            detail="every=true widens a cancel to a whole job type, so it takes no run_name",
        )
    if request.type and not request.run_name and not request.every:
        raise HTTPException(
            status_code=400,
            detail=f"type '{request.type}' with no run_name cancels every live job of that"
            " kind: name the run, or pass every=true and mean it",
        )
    stmt = select(Job.id).where(Job.status.in_(job_queue.ACTIVE))
    if request.run_name:
        stmt = stmt.where(Job.options["run_name"].astext == request.run_name)
    if request.type:
        stmt = stmt.where(Job.type == request.type)
    ids = list(await session.scalars(stmt))
    # through the queue: it fails the stranded experiment and takes each run's judge along
    return CancelResponse(cancelled=await run_in_threadpool(job_queue.cancel, ids))


@router.post("/{id}/cancel", response_model=CancelResponse)
async def cancel_job(id: int, session: AsyncSession = Depends(get_session)):
    await get_or_404(Job, id, session)
    cancelled = await run_in_threadpool(job_queue.cancel_with_its_judge, id)
    return CancelResponse(cancelled=cancelled)
