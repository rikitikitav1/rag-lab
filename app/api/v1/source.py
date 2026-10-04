from datetime import datetime
from typing import Literal

import config
import job_queue
from corpus_keys import VARIANT_RE
from crud import get_or_404
from errors import Final
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi import Path as PathParam
from models.corpus import DataChunk, DataSource, Stage
from orm.async_db import commit_and_refresh, get_session
from paths import RAW, ROOT
from pydantic import BaseModel, Field
from sources.declaration import Declaration, IntakeOverride
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.concurrency import run_in_threadpool
from use_cases import source_intake
from use_cases.source_intake import declared_row

from api.v1.eval import JobEnqueuedResponse

router = APIRouter(prefix="/source", tags=["sources"])


class SourceResponse(BaseModel):
    id: int
    name: str
    kind: str
    active: bool
    # every variant's rows; the verdict beside it is about one cut, so its count travels along
    chunks: int
    chunks_in_variant: int = 0
    ingest_quality: str | None = None
    ingest_variant: str | None = None
    ingest_checked_at: datetime | None = None
    stage: str = Stage.accepted
    language: str | None = None
    raw_verdict: str | None = None

    model_config = {"from_attributes": True}


# one source whole: where it comes from and what its raw conversion said, beside the list's fields
class SourceDetail(SourceResponse):
    licence: str | None = None
    declaration: dict | None = None
    raw: dict = {}
    drift: dict | None = None


class SourceActiveRequest(BaseModel):
    active: bool


class SourceAnalyzeRequest(BaseModel):
    variant: str | None = None
    mode: Literal["indexed", "dry"] = "indexed"


class SourceVerdicts(BaseModel):
    name: str
    verdicts: dict[str, str | None]
    scores: dict[str, int | None]
    breaches: dict[str, list[str]]
    moved: bool


class SourceCompareResponse(BaseModel):
    variants: list[str]
    sources: int
    disagreeing: int
    rows: list[SourceVerdicts]


class SourceReportResponse(BaseModel):
    name: str
    ingest_quality: str | None
    ingest_variant: str | None
    ingest_checked_at: datetime | None
    reports: dict


def _response(source, chunks: int, in_variant: int = 0) -> SourceResponse:
    return SourceResponse(
        id=source.id,
        name=source.name,
        kind=source.kind,
        active=source.active,
        chunks=chunks,
        chunks_in_variant=in_variant,
        ingest_quality=source.ingest_quality,
        ingest_variant=source.ingest_variant,
        ingest_checked_at=source.ingest_checked_at,
        stage=source.stage,
        language=source.language,
        raw_verdict=source_intake.run_under_review(source).get("verdict"),
    )


def _detail(source, chunks: int, in_variant: int = 0) -> SourceDetail:
    return SourceDetail(**source_intake.view(source, chunks, in_variant))


# a source's chunks, all and those in the variant it was last indexed as
async def _counts(session: AsyncSession, source) -> tuple[int, int]:
    count = await session.scalar(select(func.count()).select_from(DataChunk).where(DataChunk.source_id == source.id))
    in_variant = (
        await session.scalar(
            select(func.count())
            .select_from(DataChunk)
            .where(DataChunk.source_id == source.id, DataChunk.variant == source.ingest_variant)
        )
        if source.ingest_variant
        else 0
    )
    return count or 0, in_variant or 0


@router.get("", response_model=list[SourceResponse])
async def list_sources(
    stage: Stage | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
    session: AsyncSession = Depends(get_session),
):
    stmt = select(DataSource).order_by(DataSource.name).limit(limit).offset(offset)
    if stage is not None:
        stmt = stmt.where(DataSource.stage == stage)
    sources = (await session.scalars(stmt)).all()
    ids = [s.id for s in sources]
    counts = dict(
        (
            await session.execute(
                select(DataChunk.source_id, func.count())
                .where(DataChunk.source_id.in_(ids))
                .group_by(DataChunk.source_id)
            )
        ).all()
    )
    # `ingest_variant`, not the configured one: two numbers side by side must describe one thing
    per_variant = {
        (source_id, variant): n
        for source_id, variant, n in (
            await session.execute(
                select(DataChunk.source_id, DataChunk.variant, func.count())
                .where(DataChunk.source_id.in_(ids))
                .group_by(DataChunk.source_id, DataChunk.variant)
            )
        ).all()
    }
    return [_response(s, counts.get(s.id, 0), per_variant.get((s.id, s.ingest_variant), 0)) for s in sources]


def _transition(step, source, *args):
    try:
        return step(source, *args)
    except Final as e:
        raise HTTPException(status_code=409, detail=str(e)) from e


@router.put("/{id}", response_model=SourceResponse)
async def set_source_active(
    id: int,
    request: SourceActiveRequest,
    session: AsyncSession = Depends(get_session),
):
    source = await get_or_404(DataSource, id, session)
    _transition(source_intake.set_active, source, request.active)
    await session.commit()
    return _response(source, *await _counts(session, source))


# a source's own intake knobs over the stand's, read by its next onboarding; an empty body clears them
@router.put("/{id}/intake", response_model=SourceDetail)
async def set_source_intake(
    id: int, request: IntakeOverride, session: AsyncSession = Depends(get_session)
) -> SourceDetail:
    source = await get_or_404(DataSource, id, session)
    _transition(source_intake.set_intake, source, request.model_dump(exclude_none=True, exclude_defaults=True))
    await commit_and_refresh(session, source)
    return _detail(source, *await _counts(session, source))


class SkipPaths(BaseModel):
    skip_paths: list[str] = Field(max_length=200)


# the folders and files a source leaves out at its next onboarding; an empty list clears them
@router.put("/{id}/skip_paths", response_model=SourceDetail)
async def set_source_skip_paths(
    id: int, request: SkipPaths, session: AsyncSession = Depends(get_session)
) -> SourceDetail:
    source = await get_or_404(DataSource, id, session)
    _transition(source_intake.set_skip_paths, source, request.skip_paths)
    await commit_and_refresh(session, source)
    return _detail(source, *await _counts(session, source))


class Markup(BaseModel):
    markup: Literal["hugo", "mdn"] | None = None


# the markup family a source's pages are written in; null clears it
@router.put("/{id}/markup", response_model=SourceDetail)
async def set_source_markup(id: int, request: Markup, session: AsyncSession = Depends(get_session)) -> SourceDetail:
    source = await get_or_404(DataSource, id, session)
    _transition(source_intake.set_markup, source, request.markup)
    await commit_and_refresh(session, source)
    return _detail(source, *await _counts(session, source))


# a variant's chunks in every source, as a smoke leaves them; the variant the stand searches is refused
@router.delete("/variant/{variant}")
async def remove_variant(variant: str = PathParam(pattern=VARIANT_RE.pattern)) -> dict:
    try:
        return await run_in_threadpool(source_intake.remove_variant, variant, config.settings.corpus.variant)
    except Final as e:
        raise HTTPException(status_code=409, detail=str(e)) from e


# a source with its chunks and the stand's own files of it; inactive is `PUT /{id}`, this is gone
@router.delete("/{id}")
async def remove_source(id: int, session: AsyncSession = Depends(get_session)) -> dict:
    source = await get_or_404(DataSource, id, session)
    try:
        return await run_in_threadpool(source_intake.remove_source, source, RAW)
    except Final as e:
        raise HTTPException(status_code=409, detail=str(e)) from e


# declared before `/{id}` so a literal path is not read as an id
@router.get("/compare", response_model=SourceCompareResponse)
async def compare_sources(
    variants: list[str] = Query(min_length=2),
    session: AsyncSession = Depends(get_session),
):
    from corpus_keys import check_variant

    for variant in variants:
        check_variant(variant)
    sources = list(await session.scalars(select(DataSource).order_by(DataSource.name)))
    rows, disagreeing = [], 0
    for source in sources:
        reports = source.ingest_reports or {}
        latest = {v: (reports.get(v) or [{}])[-1] for v in variants}
        if not any(latest[v] for v in variants):
            continue
        verdicts = {v: latest[v].get("verdict") for v in variants}
        moved = len({verdicts[v] for v in variants}) > 1
        disagreeing += moved
        rows.append(
            SourceVerdicts(
                name=source.name,
                verdicts=verdicts,
                scores={v: latest[v].get("score") for v in variants},
                breaches={v: latest[v].get("breaches") or [] for v in variants},
                moved=moved,
            )
        )
    return SourceCompareResponse(variants=variants, sources=len(rows), disagreeing=disagreeing, rows=rows)


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
    from corpus_keys import check_variant

    source = await get_or_404(DataSource, id, session)
    if request.variant is not None:
        check_variant(request.variant)
    options = {"source": source.name, "variant": request.variant, "mode": request.mode}
    job = await job_queue.add_job(session, "analyze_source", options)
    await commit_and_refresh(session, job)
    return JobEnqueuedResponse.model_validate(job)


# a source added by hand starts declared: where its files come from, never which engine reads them
@router.post("", response_model=SourceDetail, status_code=201)
async def declare_source(request: Declaration, session: AsyncSession = Depends(get_session)) -> SourceDetail:
    if refusal := source_intake.declaration_refusal(request, ROOT):
        raise HTTPException(status_code=422, detail=refusal)
    if await session.scalar(select(DataSource.id).where(DataSource.name == request.name)):
        raise HTTPException(status_code=409, detail=source_intake.name_taken(request.name))
    source = declared_row(request)
    session.add(source)
    # two declarations of one name at once: the second meets the unique name at commit
    try:
        await commit_and_refresh(session, source)
    except IntegrityError as e:
        await session.rollback()
        raise HTTPException(status_code=409, detail=source_intake.name_taken(request.name)) from e
    return _detail(source, 0)


class SourceOnboardRequest(BaseModel):
    # per engine, a settings file of that tool; the defaults live in `intake.settings`
    settings: dict[str, str] | None = None
    # every piece read by its tool again, past every cache: a run that measures the tool itself
    fresh: bool = False


class SourceAcceptRequest(BaseModel):
    # said when the raw verdict is bad; it stays on the row beside the verdict
    reason: str | None = Field(
        default=None, min_length=source_intake.REASON_CHARS[0], max_length=source_intake.REASON_CHARS[1]
    )


# raw to accepted: the conversion is fit to index, and the index reads it once a source file names it
@router.post("/{id}/accept", response_model=SourceDetail)
async def accept_source(
    id: int, request: SourceAcceptRequest, session: AsyncSession = Depends(get_session)
) -> SourceDetail:
    source = await get_or_404(DataSource, id, session)
    replaced = _transition(source_intake.accept, source, request.reason)
    await commit_and_refresh(session, source)
    source_intake.drop_folder(replaced, source.name)
    return _detail(source, *await _counts(session, source))


# a declared source to a raw folder: the job routes each file to its engine and reports without a gold
@router.post("/{id}/onboard", response_model=JobEnqueuedResponse)
async def onboard_source(
    id: int, request: SourceOnboardRequest, session: AsyncSession = Depends(get_session)
) -> JobEnqueuedResponse:
    source = await get_or_404(DataSource, id, session)
    _transition(source_intake.check_onboard, source)
    options = source_intake.onboard_options(source.name, request.settings, request.fresh)
    job = await job_queue.add_job(session, "onboard_source", options)
    await commit_and_refresh(session, job)
    return JobEnqueuedResponse.model_validate(job)


# declared after `/compare`, so that path is not read as an id
@router.get("/{id}", response_model=SourceDetail)
async def get_source(id: int, session: AsyncSession = Depends(get_session)) -> SourceDetail:
    source = await get_or_404(DataSource, id, session)
    return _detail(source, *await _counts(session, source))
