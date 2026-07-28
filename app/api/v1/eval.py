import time
from typing import Literal

import job_queue
from fastapi import APIRouter, Depends, HTTPException, Query
from models.eval import QuestionLog
from models.registry import Pipeline
from orm.async_db import commit_and_refresh, get_session
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

router = APIRouter(prefix="/eval", tags=["eval"])


class JobEnqueuedResponse(BaseModel):
    job_id: int
    type: str
    options: dict


class ParaphraseRequest(BaseModel):
    limit: int = 100
    source: str | None = None
    set_name: str = "paraphrased"


class EvalRunRequest(BaseModel):
    run_name: str | None = None
    set_name: str | None = None
    question_ids: list[int] | None = None
    rerank: bool | None = None
    pipeline: Pipeline = Pipeline.single_shot
    language: Literal["ru", "en"] | None = None
    k: int | None = None
    max_hops: int | None = None


class ExperimentRequest(BaseModel):
    run_name: str | None = None
    set_name: str | None = None
    question_ids: list[int] | None = None
    rerank: bool | None = None
    pipeline: Pipeline = Pipeline.single_shot
    language: Literal["ru", "en"] | None = None
    param: Literal["k", "max_hops"] = "k"
    values: list[int] = Field(min_length=1)


async def _enqueue(session, type: str, options: dict) -> JobEnqueuedResponse:
    job = job_queue.add_job(session, type, options)
    await commit_and_refresh(session, job)
    return JobEnqueuedResponse(job_id=job.id, type=job.type, options=job.options)


@router.post("/paraphrase", response_model=JobEnqueuedResponse)
async def enqueue_paraphrase(
    request: ParaphraseRequest,
    session: AsyncSession = Depends(get_session),
):
    return await _enqueue(
        session,
        "paraphrase_questions",
        {
            "limit": request.limit,
            "source": request.source,
            "set_name": request.set_name,
        },
    )


class MissItem(BaseModel):
    question_id: int
    question: str
    expected: list[str]
    retrieved: list[str]
    faithfulness: str | None
    relevance: str | None
    completeness: str | None


class MissesResponse(BaseModel):
    run_name: str
    in_corpus: int
    misses: int
    items: list[MissItem]


@router.get("/misses", response_model=MissesResponse)
async def eval_misses(
    run_name: str,
    limit: int = Query(default=50, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
    session: AsyncSession = Depends(get_session),
):
    stmt = (
        select(QuestionLog)
        .options(selectinload(QuestionLog.question))
        .where(QuestionLog.run_name == run_name)
    )
    logs = (await session.scalars(stmt)).all()

    in_corpus = 0
    items: list[MissItem] = []
    for ql in logs:
        q = ql.question
        if not (q and q.marked_sources):
            continue
        in_corpus += 1
        got = [s["source"] for s in (ql.sources or [])]
        hit = any(any(exp in g for exp in q.marked_sources) for g in got)
        if not hit:
            items.append(
                MissItem(
                    question_id=q.id,
                    question=q.original_text,
                    expected=q.marked_sources,
                    retrieved=got,
                    faithfulness=ql.faithfulness,
                    relevance=ql.relevance,
                    completeness=ql.completeness,
                )
            )

    return MissesResponse(
        run_name=run_name,
        in_corpus=in_corpus,
        misses=len(items),
        items=items[offset : offset + limit],
    )


@router.post("/run", response_model=JobEnqueuedResponse)
async def enqueue_eval_run(
    request: EvalRunRequest,
    session: AsyncSession = Depends(get_session),
):
    run_name = request.run_name or f"{request.set_name or 'all'}_{int(time.time())}"
    return await _enqueue(
        session,
        "eval_run",
        {
            "run_name": run_name,
            "set_name": request.set_name,
            "question_ids": request.question_ids,
            "rerank": request.rerank,
            "k": request.k,
            "max_hops": request.max_hops,
            "pipeline": request.pipeline.value,
            "language": request.language,
        },
    )


@router.post("/experiment", response_model=list[JobEnqueuedResponse])
async def enqueue_experiment(
    request: ExperimentRequest,
    session: AsyncSession = Depends(get_session),
):
    if any(v < 1 for v in request.values):
        raise HTTPException(status_code=400, detail="values must be positive")
    base = request.run_name or f"{request.set_name or 'all'}_{request.pipeline.value}_{int(time.time())}"
    jobs = []
    for value in request.values:
        job = await _enqueue(
            session,
            "eval_run",
            {
                "run_name": f"{base}_{request.param}{value:02d}",
                "set_name": request.set_name,
                "question_ids": request.question_ids,
                "rerank": request.rerank,
                "pipeline": request.pipeline.value,
                "language": request.language,
                request.param: value,
            },
        )
        jobs.append(job)
    return jobs
