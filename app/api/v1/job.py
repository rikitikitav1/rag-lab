from datetime import datetime

from crud import get_or_404
from fastapi import APIRouter, Depends, Query
from models.jobs import Job, JobStatus
from orm.async_db import get_session
from pydantic import BaseModel
from query_utils import Page, apply_sort_limit_offset
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

router = APIRouter(prefix="/job", tags=["jobs"])


class JobResponse(BaseModel):
    id: int
    type: str
    status: JobStatus
    options: dict
    error: dict | None
    elapsed: float | None
    apply_since: datetime
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


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
    page: Page = Depends(),
    session: AsyncSession = Depends(get_session),
):
    stmt = select(Job)
    if type is not None:
        stmt = stmt.where(Job.type.in_(type))
    if status is not None:
        stmt = stmt.where(Job.status.in_(status))
    if created_from is not None:
        stmt = stmt.where(Job.created_at >= created_from)
    if created_to is not None:
        stmt = stmt.where(Job.created_at <= created_to)

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


@router.get("/{id}", response_model=JobResponse)
async def show_job(id: int, session: AsyncSession = Depends(get_session)):
    return await get_or_404(Job, id, session)


class CancelResponse(BaseModel):
    cancelled: list[int]


_ACTIVE = (JobStatus.new, JobStatus.running)


@router.post("/{id}/cancel", response_model=CancelResponse)
async def cancel_job(id: int, session: AsyncSession = Depends(get_session)):
    job = await get_or_404(Job, id, session)
    targets = [job]

    run_name = (job.options or {}).get("run_name")
    if job.type == "eval_run" and run_name:
        deps = await session.scalars(
            select(Job).where(
                Job.type == "judge_answers",
                Job.status.in_(_ACTIVE),
                Job.options["run_name"].astext == run_name,
            )
        )
        targets.extend(deps)

    cancelled = []
    for t in targets:
        if t.status in _ACTIVE:
            t.status = JobStatus.cancelled
            cancelled.append(t.id)
    await session.commit()
    return CancelResponse(cancelled=cancelled)
