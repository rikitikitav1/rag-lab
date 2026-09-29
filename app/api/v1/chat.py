from typing import Literal

import config
import search_scope
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field, model_validator
from use_cases import card_wait, chat

import db
from api.v1.card_door import wait_for_the_card
from api.v1.schemas import AnswerSource

router = APIRouter(prefix="/chat", tags=["chat"])


# options and tags were accepted and never read: a client that chose a model got the default unsaid
class QuestionFilter(BaseModel):
    model_config = ConfigDict(extra="forbid")
    # the MCP door validated this and the REST doors did not; a label, never a pattern
    category: str | None = Field(default=None, pattern=search_scope.CATEGORY_RE.pattern)
    sources: list[str] = Field(default=[], max_length=search_scope.MAX_SOURCES)
    version: str | None = Field(default=None, pattern=search_scope.VERSION_RE.pattern)

    # the scope's own rules; whether its sources are in search is the search's step, said as a 422 by `_scoped`
    @model_validator(mode="after")
    def _a_scope_the_search_can_read(self):
        search_scope.refuse_bad_scope(self.scope())
        return self

    def scope(self) -> search_scope.Scope:
        return search_scope.Scope.of(self.category, self.sources, self.version)


# a source named in the filter that the base does not hold is the asker's mistake, said before the answer is paid
def _scoped(call):
    try:
        return call()
    except db.ScopeRefused as e:
        raise HTTPException(status_code=422, detail=str(e)) from e


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
    scope = question.filter.scope() if question.filter else None
    res = _scoped(
        lambda: chat.answer(
            question.text,
            scope,
            use_rerank=question.rerank,
            language=question.language,
            ef_search=question.ef_search,
        )
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
    scope = question.filter.scope() if question.filter else None
    res = _scoped(
        lambda: chat.retrieve(
            question.text,
            scope,
            variant=config.settings.corpus.variant,
            ef_search=question.ef_search,
            use_rerank=question.rerank,
        )
    )
    return RetrievalResponse(
        sources=[AnswerSource.of(s) for s in res.sources],
        elapsed_time_seconds=res.elapsed,
    )
