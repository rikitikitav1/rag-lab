from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query
from models.eval import Question, QuestionLog
from models.registry import Pipeline
from orm.async_db import get_session
from pydantic import BaseModel
from query_utils import Page, apply_sort_limit_offset
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
from use_cases.agent_policy import FallbackPolicy, FallbackReason, Orchestrator

router = APIRouter(prefix="/question-log", tags=["question-logs"])


class QuestionLogItem(BaseModel):
    id: int
    run_name: str | None
    pipeline: str
    question_id: int | None
    question_text: str | None
    set_name: str | None
    answered: bool
    answer: str | None
    sources: list | None
    models: dict
    prompts: dict
    prompt_tokens: int | None
    completion_tokens: int | None
    elapsed: float | None
    faithfulness: str | None
    relevance: str | None
    completeness: str | None
    metrics: dict
    created_at: datetime


class QuestionLogDetail(QuestionLogItem):
    context: str | None


SORT_MAP = {
    "id": QuestionLog.id,
    "created_at": QuestionLog.created_at,
    "elapsed": QuestionLog.elapsed,
}


def _item(ql: QuestionLog) -> dict:
    return {
        "id": ql.id,
        "run_name": ql.run_name,
        "pipeline": ql.pipeline,
        "question_id": ql.question_id,
        "question_text": ql.question.original_text if ql.question else None,
        "set_name": ql.question.set_name if ql.question else None,
        "answered": ql.answered,
        "answer": ql.answer,
        "sources": ql.sources,
        "models": ql.models,
        "prompts": ql.prompts,
        "prompt_tokens": ql.prompt_tokens,
        "completion_tokens": ql.completion_tokens,
        "elapsed": ql.elapsed,
        "faithfulness": ql.faithfulness,
        "relevance": ql.relevance,
        "completeness": ql.completeness,
        "metrics": ql.metrics,
        "created_at": ql.created_at,
    }


@router.get("", response_model=list[QuestionLogItem])
async def list_question_logs(
    question_id: int | None = Query(default=None),
    text: str | None = Query(default=None, description="substring in question text"),
    set_name: list[str] | None = Query(default=None),
    run_name: list[str] | None = Query(default=None),
    pipeline: list[Pipeline] | None = Query(default=None),
    answered: bool | None = Query(default=None),
    faithfulness: list[str] | None = Query(default=None),
    relevance: list[str] | None = Query(default=None),
    completeness: list[str] | None = Query(default=None),
    created_from: datetime | None = Query(default=None),
    created_to: datetime | None = Query(default=None),
    rerank: bool | None = Query(default=None),
    rerank_device: str | None = Query(default=None, max_length=16),
    phased: bool | None = Query(default=None),
    fallback_policy: list[FallbackPolicy] | None = Query(default=None),
    fallback_reason: list[FallbackReason] | None = Query(default=None),
    orchestrator: list[Orchestrator] | None = Query(default=None),
    empty_retrieval: bool | None = Query(
        default=None, description="true = the corpus returned nothing"
    ),
    max_distance: float | None = Query(
        default=None, ge=0, le=2, description="keep logs whose closest chunk was at least this near"
    ),
    page: Page = Depends(),
    session: AsyncSession = Depends(get_session),
):
    stmt = select(QuestionLog).options(selectinload(QuestionLog.question))

    if text is not None or set_name is not None:
        stmt = stmt.join(Question, QuestionLog.question_id == Question.id)
        if text is not None:
            stmt = stmt.where(Question.original_text.ilike(f"%{text}%"))
        if set_name is not None:
            stmt = stmt.where(Question.set_name.in_(set_name))

    if question_id is not None:
        stmt = stmt.where(QuestionLog.question_id == question_id)
    if run_name is not None:
        stmt = stmt.where(QuestionLog.run_name.in_(run_name))
    if pipeline is not None:
        stmt = stmt.where(QuestionLog.pipeline.in_(pipeline))
    if answered is not None:
        stmt = stmt.where(QuestionLog.answered.is_(answered))
    if faithfulness is not None:
        stmt = stmt.where(QuestionLog.faithfulness.in_(faithfulness))
    if relevance is not None:
        stmt = stmt.where(QuestionLog.relevance.in_(relevance))
    if completeness is not None:
        stmt = stmt.where(QuestionLog.completeness.in_(completeness))
    if created_from is not None:
        stmt = stmt.where(QuestionLog.created_at >= created_from)
    if created_to is not None:
        stmt = stmt.where(QuestionLog.created_at <= created_to)

    config = QuestionLog.metrics["config"]
    retrieval = QuestionLog.metrics["retrieval"]
    if rerank is not None:
        stmt = stmt.where(config["rerank"].as_boolean().is_(rerank))
    if rerank_device is not None:
        stmt = stmt.where(config["rerank_device"].astext == rerank_device)
    if phased is not None:
        stmt = stmt.where(config["phased"].as_boolean().is_(phased))
    if fallback_policy is not None:
        stmt = stmt.where(config["fallback_policy"].astext.in_(fallback_policy))
    if fallback_reason is not None:
        stmt = stmt.where(QuestionLog.metrics["fallback_reason"].astext.in_(fallback_reason))
    if orchestrator is not None:
        stmt = stmt.where(config["orchestrator"]["name"].astext.in_(orchestrator))
    if empty_retrieval is not None:
        count = retrieval["results_count"].as_integer()
        stmt = stmt.where(count == 0 if empty_retrieval else count > 0)
    if max_distance is not None:
        stmt = stmt.where(retrieval["min_distance"].as_float() <= max_distance)

    stmt = apply_sort_limit_offset(
        stmt=stmt,
        sort_map=SORT_MAP,
        sort_by=page.sort_by,
        sort_order=page.sort_order,
        limit=page.limit,
        offset=page.offset,
        default_sort="created_at",
    )

    result = await session.scalars(stmt)
    return [QuestionLogItem(**_item(ql)) for ql in result.all()]


@router.get("/{id}", response_model=QuestionLogDetail)
async def show_question_log(id: int, session: AsyncSession = Depends(get_session)):
    stmt = (
        select(QuestionLog)
        .options(selectinload(QuestionLog.question))
        .where(QuestionLog.id == id)
    )
    ql = await session.scalar(stmt)
    if ql is None:
        raise HTTPException(status_code=404, detail=f"QuestionLog with id={id} not found")
    return QuestionLogDetail(**_item(ql), context=ql.context)
