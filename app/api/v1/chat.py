from typing import Literal

import config
from fastapi import APIRouter
from pydantic import BaseModel, ConfigDict, Field
from use_cases import card_wait, chat

import db
from api.v1.card_door import wait_for_the_card
from api.v1.schemas import AnswerSource

router = APIRouter(prefix="/chat", tags=["chat"])


# options and tags were accepted and never read: a client that chose a model got the default unsaid
class QuestionFilter(BaseModel):
    model_config = ConfigDict(extra="forbid")
    # the MCP door validated this and the REST doors did not, onto the same lquery
    category: str | None = Field(default=None, pattern=db.CATEGORY_RE.pattern)


class QuestionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str
    filter: QuestionFilter | None = None
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


@router.post("/question", response_model=QuestionResponse)
def ask(question: QuestionRequest) -> QuestionResponse:
    wait_for_the_card(*card_wait.answering_roles(rerank_asked=question.rerank))
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
    wait_for_the_card(*card_wait.retrieving_roles(rerank_asked=question.rerank))
    category = question.filter.category if question.filter else None
    res = chat.retrieve(
        question.text, category, variant=config.settings.corpus.variant,
        ef_search=question.ef_search, use_rerank=question.rerank,
    )
    return RetrievalResponse(
        sources=[AnswerSource.of(s) for s in res.sources],
        elapsed_time_seconds=res.elapsed,
    )
