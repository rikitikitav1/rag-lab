from typing import Literal

import config
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from use_cases import card_wait, chat

import db
from api.v1.schemas import AnswerSource

router = APIRouter(prefix="/chat", tags=["chat"])


class QuestionOptions(BaseModel):
    model: str | None = None
    max_distance: float | None = None
    temperature: float | None = None
    max_tokens: int | None = None


class QuestionFilter(BaseModel):
    # the MCP door validated this and the REST doors did not, onto the same lquery
    category: str | None = Field(default=None, pattern=db.CATEGORY_RE.pattern)
    tags: list[str] = []


class QuestionRequest(BaseModel):
    text: str
    filter: QuestionFilter | None = None
    options: QuestionOptions | None = None
    rerank: bool | None = None
    language: Literal["ru", "en"] | None = None
    # 1..1000 is what the server accepts: a value it refuses dies after the embedding is paid
    ef_search: int | None = Field(default=None, ge=1, le=1000)


class AnswerMetrics(BaseModel):
    success: bool
    model: str
    elapsed_time_seconds: float
    distance_threshold: float
    prompt_tokens: int
    completion_tokens: int


class QuestionResponse(BaseModel):
    text: str
    metrics: AnswerMetrics
    sources: list[AnswerSource] = []


class RetrievalResponse(BaseModel):
    sources: list[AnswerSource]
    elapsed_time_seconds: float


# the answering doors wait for the card through the queue; a layout they cannot serve is a 409
def _wait_for_the_card(*roles) -> None:
    try:
        card_wait.wait_for_the_card(*roles)
    except card_wait.CardBusy as e:
        headers = {"Retry-After": str(e.retry_after)} if e.retry_after else None
        raise HTTPException(status_code=e.status, detail=e.detail, headers=headers) from e


# the cross-encoder is a role on its own engine now, and a door that reranks waits for it too
def _reranking(asked: bool | None) -> tuple[str, ...]:
    return ("reranking",) if chat.resolve_rerank(asked) else ()


@router.post("/question", response_model=QuestionResponse)
def ask(question: QuestionRequest) -> QuestionResponse:
    _wait_for_the_card("embedding", "generation", *_reranking(question.rerank))
    category = question.filter.category if question.filter else None
    res = chat.answer(
        question.text,
        category,
        use_rerank=question.rerank,
        language=question.language,
        ef_search=question.ef_search,
    )
    return QuestionResponse(
        text=res.text,
        metrics=AnswerMetrics(
            success=res.success,
            model=res.metrics.model,
            elapsed_time_seconds=res.elapsed,
            distance_threshold=res.metrics.distance_threshold,
            prompt_tokens=res.metrics.prompt_tokens,
            completion_tokens=res.metrics.completion_tokens,
        ),
        sources=[AnswerSource.of(s) for s in res.sources],
    )


@router.post("/fast_question", response_model=RetrievalResponse)
def quick_ask(question: QuestionRequest) -> RetrievalResponse:
    _wait_for_the_card("embedding", *_reranking(question.rerank))
    category = question.filter.category if question.filter else None
    res = chat.retrieve(
        question.text, category, variant=config.settings.corpus.variant,
        ef_search=question.ef_search,
    )
    return RetrievalResponse(
        sources=[AnswerSource.of(s) for s in res.sources],
        elapsed_time_seconds=res.elapsed,
    )
