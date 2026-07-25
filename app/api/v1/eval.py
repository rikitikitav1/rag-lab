import time

import job_queue
from fastapi import APIRouter, Depends
from orm.async_db import commit_and_refresh, get_session
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

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
        },
    )
