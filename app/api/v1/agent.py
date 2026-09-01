from typing import Literal

from fastapi import APIRouter
from pydantic import BaseModel, Field
from use_cases import agent, agent_policy

from api.v1.schemas import AnswerSource

router = APIRouter(prefix="/agent", tags=["agent"])


class AgentRequest(BaseModel):
    text: str
    max_hops: int | None = Field(default=None, ge=1, le=agent_policy.MAX_HOPS)
    language: Literal["ru", "en"] | None = None
    fallback_policy: agent.FallbackPolicy | None = None
    debug: bool = False


class AgentResponse(BaseModel):
    text: str
    success: bool
    hops: int
    prompt_tokens: int
    completion_tokens: int
    elapsed_time_seconds: float
    sources: list[AnswerSource] = []
    trace: list[dict] | None = None


def _serialize_trace(messages) -> list[dict]:
    return [m if isinstance(m, dict) else m.model_dump() for m in messages]


@router.post("/question", response_model=AgentResponse)
def ask(request: AgentRequest) -> AgentResponse:
    res = agent.run(
        request.text,
        max_hops=request.max_hops,
        language=request.language,
        fallback_policy=request.fallback_policy,
    )
    return AgentResponse(
        text=res.text,
        success=res.success,
        hops=res.hops,
        prompt_tokens=res.prompt_tokens,
        completion_tokens=res.completion_tokens,
        elapsed_time_seconds=res.elapsed,
        sources=[AnswerSource.of(s) for s in res.sources],
        trace=_serialize_trace(res.messages) if request.debug else None,
    )
