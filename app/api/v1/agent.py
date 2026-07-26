from typing import Literal

from fastapi import APIRouter
from pydantic import BaseModel, Field
from use_cases import agent

router = APIRouter(prefix="/agent", tags=["agent"])


class AgentRequest(BaseModel):
    text: str
    max_hops: int | None = Field(default=None, ge=1, le=10)
    language: Literal["ru", "en"] | None = None
    debug: bool = False


class AgentSource(BaseModel):
    link: str
    vector_distance: float | None = None
    vector_rank: float | None = None
    keyword_rank: float | None = None
    score: float


class AgentResponse(BaseModel):
    text: str
    success: bool
    hops: int
    prompt_tokens: int
    completion_tokens: int
    elapsed_time_seconds: float
    sources: list[AgentSource] = []
    trace: list[dict] | None = None


def _serialize_trace(messages) -> list[dict]:
    return [m if isinstance(m, dict) else m.model_dump() for m in messages]


@router.post("/question", response_model=AgentResponse)
def ask(request: AgentRequest) -> AgentResponse:
    if request.max_hops is not None:
        res = agent.run(request.text, max_hops=request.max_hops, language=request.language)
    else:
        res = agent.run(request.text, language=request.language)
    return AgentResponse(
        text=res.text,
        success=res.success,
        hops=res.hops,
        prompt_tokens=res.prompt_tokens,
        completion_tokens=res.completion_tokens,
        elapsed_time_seconds=res.elapsed,
        sources=[
            AgentSource(
                link=s.source,
                vector_distance=s.vector_distance,
                vector_rank=s.vector_rank,
                keyword_rank=s.keyword_rank,
                score=s.score,
            )
            for s in res.sources
        ],
        trace=_serialize_trace(res.messages) if request.debug else None,
    )
