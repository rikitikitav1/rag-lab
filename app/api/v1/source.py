from datetime import datetime
from typing import Literal

import job_queue
from crud import get_or_404
from fastapi import APIRouter, Depends
from models.corpus import DataChunk, DataSource
from orm.async_db import commit_and_refresh, get_session
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from api.v1.eval import JobEnqueuedResponse

router = APIRouter(prefix="/source", tags=["sources"])


class SourceResponse(BaseModel):
    id: int
    name: str
    kind: str
    active: bool
    chunks: int
    ingest_quality: str | None = None
    ingest_variant: str | None = None
    ingest_checked_at: datetime | None = None

    model_config = {"from_attributes": True}


class SourceActiveRequest(BaseModel):
    active: bool


class SourceAnalyzeRequest(BaseModel):
    variant: str | None = None
    mode: Literal["indexed", "dry"] = "indexed"


class SourceReportResponse(BaseModel):
    name: str
    ingest_quality: str | None
    ingest_variant: str | None
    ingest_checked_at: datetime | None
    reports: dict


def _response(source, chunks: int) -> SourceResponse:
    return SourceResponse(
        id=source.id,
        name=source.name,
        kind=source.kind,
        active=source.active,
        chunks=chunks,
        ingest_quality=source.ingest_quality,
        ingest_variant=source.ingest_variant,
        ingest_checked_at=source.ingest_checked_at,
    )


@router.get("", response_model=list[SourceResponse])
async def list_sources(session: AsyncSession = Depends(get_session)):
    counts = dict(
        (
            await session.execute(
                select(DataChunk.source_id, func.count()).group_by(DataChunk.source_id)
            )
        ).all()
    )
    sources = (await session.scalars(select(DataSource).order_by(DataSource.name))).all()
    return [_response(s, counts.get(s.id, 0)) for s in sources]


@router.put("/{id}", response_model=SourceResponse)
async def set_source_active(
    id: int,
    request: SourceActiveRequest,
    session: AsyncSession = Depends(get_session),
):
    source = await get_or_404(DataSource, id, session)
    source.active = request.active
    await session.commit()
    count = await session.scalar(
        select(func.count()).select_from(DataChunk).where(DataChunk.source_id == id)
    )
    return _response(source, count or 0)


@router.get("/{id}/report", response_model=SourceReportResponse)
async def get_source_report(id: int, session: AsyncSession = Depends(get_session)):
    source = await get_or_404(DataSource, id, session)
    return SourceReportResponse(
        name=source.name,
        ingest_quality=source.ingest_quality,
        ingest_variant=source.ingest_variant,
        ingest_checked_at=source.ingest_checked_at,
        reports=source.ingest_reports or {},
    )


@router.post("/{id}/analyze", response_model=JobEnqueuedResponse)
async def analyze_source(
    id: int,
    request: SourceAnalyzeRequest,
    session: AsyncSession = Depends(get_session),
) -> JobEnqueuedResponse:
    from use_cases.index import check_variant

    source = await get_or_404(DataSource, id, session)
    if request.variant is not None:
        check_variant(request.variant)
    options = {"source": source.name, "variant": request.variant, "mode": request.mode}
    job = job_queue.add_job(session, "analyze_source", options)
    await commit_and_refresh(session, job)
    return JobEnqueuedResponse(job_id=job.id, type=job.type, options=job.options)
