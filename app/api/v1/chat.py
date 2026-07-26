from typing import Literal

from fastapi import APIRouter
from pydantic import BaseModel
from use_cases import chat

router = APIRouter(prefix="/chat", tags=["chat"])


class QuestionOptions(BaseModel):
    model: str | None = None
    max_distance: float | None = None
    temperature: float | None = None
    max_tokens: int | None = None


class QuestionFilter(BaseModel):
    category: str | None = None
    tags: list[str] = []


class QuestionRequest(BaseModel):
    text: str
    filter: QuestionFilter | None = None
    options: QuestionOptions | None = None
    rerank: bool | None = None
    language: Literal["ru", "en"] | None = None


class AnswerMetrics(BaseModel):
    success: bool
    model: str
    elapsed_time_seconds: float
    distance_threshold: float
    prompt_tokens: int
    completion_tokens: int


class AnswerSource(BaseModel):
    link: str
    vector_distance: float | None = None
    vector_rank: float | None = None
    keyword_rank: float | None = None
    score: float


class QuestionResponse(BaseModel):
    text: str
    metrics: AnswerMetrics
    sources: list[AnswerSource] = []


class RetrievalResponse(BaseModel):
    sources: list[AnswerSource]
    elapsed_time_seconds: float


@router.post("/question", response_model=QuestionResponse)
def ask(question: QuestionRequest) -> QuestionResponse:
    category = question.filter.category if question.filter else None
    res = chat.answer(
        question.text,
        category,
        use_rerank=question.rerank,
        language=question.language,
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
        sources=[
            AnswerSource(
                link=s.source,
                vector_distance=s.vector_distance,
                vector_rank=s.vector_rank,
                keyword_rank=s.keyword_rank,
                score=s.score,
            )
            for s in res.sources
        ],
    )


@router.post("/fast_question", response_model=RetrievalResponse)
def quick_ask(question: QuestionRequest) -> RetrievalResponse:
    category = question.filter.category if question.filter else None
    res = chat.retrieve(question.text, category)
    return RetrievalResponse(
        sources=[
            AnswerSource(
                link=s.source,
                vector_distance=s.vector_distance,
                vector_rank=s.vector_rank,
                keyword_rank=s.keyword_rank,
                score=s.score,
            )
            for s in res.sources
        ],
        elapsed_time_seconds=res.elapsed,
    )
