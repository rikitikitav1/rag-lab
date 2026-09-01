import re
import time
from enum import StrEnum
from typing import Literal

import config
import job_queue
import limits
from evals import compare as compare_uc
from evals import retrieval_metrics
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.concurrency import run_in_threadpool
from models.eval import QuestionLog
from models.registry import MAX_MODEL_NAME, MODEL_NAME_RE, Pipeline, refuse_unknown_registry
from orm.async_db import commit_and_refresh, get_session
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
from use_cases import agent_policy, rejudge, retrieval_compare
from use_cases.agent_policy import GONE, FallbackPolicy, GateSignal, Orchestrator
from use_cases.chat import resolve_rerank
from use_cases.index import VARIANT_RE

# what a run may ask for is not what a log may hold: both retired arms stay queryable
RunnableOrchestrator = StrEnum(
    "RunnableOrchestrator",
    {o.name: o.value for o in Orchestrator if o not in GONE},
)

router = APIRouter(prefix="/eval", tags=["eval"])


class JobEnqueuedResponse(BaseModel):
    job_id: int
    type: str
    options: dict


class RejudgeRequest(BaseModel):
    source: str = Field(min_length=1, max_length=limits.MAX_RUN_NAME)
    run_name: str = Field(min_length=1, max_length=limits.MAX_RUN_NAME)


class RejudgeResponse(BaseModel):
    job_id: int
    run_name: str
    copied: int


class ParaphraseRequest(BaseModel):
    limit: int | None = 100
    source: str | None = None
    set_name: str = "paraphrased"
    seed: str = ""
    per_source: int | None = None
    grow: bool = False


class EvalRunRequest(BaseModel):
    run_name: str | None = Field(default=None, max_length=limits.MAX_RUN_NAME)
    set_name: str | None = None
    question_ids: list[int] | None = Field(default=None, max_length=limits.MAX_QUESTION_IDS)
    rerank: bool | None = None
    pipeline: Pipeline = Pipeline.single_shot
    language: Literal["ru", "en"] | None = None
    k: int | None = Field(default=None, ge=1, le=limits.MAX_K)
    max_hops: int | None = Field(default=None, ge=1, le=agent_policy.MAX_HOPS)
    model: str | None = Field(
        default=None, max_length=MAX_MODEL_NAME, pattern=MODEL_NAME_RE.pattern
    )
    fallback_policy: FallbackPolicy | None = None
    gate_signal: GateSignal | None = None
    weak_distance: float | None = Field(default=None, ge=0, le=2)
    topic_threshold: float | None = Field(default=None, ge=0, le=2)
    orchestrator: RunnableOrchestrator | None = None
    allow_cpu: bool = False
    variant: str | None = Field(default=None, pattern=VARIANT_RE.pattern)


class ExperimentRequest(BaseModel):
    run_name: str | None = Field(default=None, max_length=limits.MAX_RUN_NAME)
    set_name: str | None = None
    question_ids: list[int] | None = Field(default=None, max_length=limits.MAX_QUESTION_IDS)
    rerank: bool | None = None
    pipeline: Pipeline = Pipeline.single_shot
    language: Literal["ru", "en"] | None = None
    param: Literal[
        "k", "max_hops", "model", "fallback_policy", "gate_signal", "weak_distance",
        "topic_threshold", "orchestrator", "variant",
    ] = "k"
    values: list[int | float | str] = Field(
        min_length=1, max_length=retrieval_compare.GRID_CAP
    )
    # the corpus every arm reads unless `variant` is the swept parameter
    variant: str | None = Field(default=None, pattern=VARIANT_RE.pattern)


async def _enqueue(session, type: str, options: dict) -> JobEnqueuedResponse:
    job = job_queue.add_job(session, type, options)
    await commit_and_refresh(session, job)
    return JobEnqueuedResponse(job_id=job.id, type=job.type, options=job.options)


@router.post("/paraphrase", response_model=JobEnqueuedResponse)
async def enqueue_paraphrase(
    request: ParaphraseRequest,
    session: AsyncSession = Depends(get_session),
):
    from evals import build_paraphrased

    options = {
        "limit": request.limit,
        "source": request.source,
        "set_name": request.set_name,
        "seed": request.seed,
        "per_source": request.per_source,
        "grow": request.grow,
    }
    # off the loop and resolved at enqueue: a stale job back in the queue takes a second helping
    options["originals"] = await run_in_threadpool(build_paraphrased.plan, **options)
    return await _enqueue(session, "paraphrase_questions", options)


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
        hit = any(retrieval_metrics.is_gold(g, q.marked_sources) for g in got)
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


class CompareResponse(BaseModel):
    runs: list[str]
    pools: dict
    blended_do_not_rank: dict


@router.get("/compare", response_model=CompareResponse)
async def eval_compare(
    runs: list[str] = Query(min_length=1),
    session: AsyncSession = Depends(get_session),
):
    try:
        named = compare_uc.named_runs(runs)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    loaded = {}
    for run_name in named:
        stmt = (
            select(QuestionLog)
            .options(selectinload(QuestionLog.question))
            .where(QuestionLog.run_name == run_name)
        )
        loaded[run_name] = list((await session.scalars(stmt)).all())

    empty = [name for name, logs in loaded.items() if not logs]
    if empty:
        raise HTTPException(status_code=404, detail=f"no logs for runs: {empty}")
    return compare_uc.compare(loaded)


# rules live beside `measure`, so AXES and the rules cannot name different sets
def validate_axis_values(axis: str, values: list) -> None:
    rule = retrieval_compare.AXIS_RULES.get(axis)
    if rule is None:
        raise HTTPException(status_code=400, detail=f"no rule for axis {axis!r}")
    bad = [v for v in values if not rule(v)]
    if bad:
        detail = f"{axis} takes {retrieval_compare.AXIS_LIMITS[axis]}, got: {bad}"
        if axis == "variant":
            detail += f", declared: {sorted(config.settings.corpus.variants)}"
        raise HTTPException(status_code=400, detail=detail)


def validate_param_values(param: str, values: list, pipeline: Pipeline | None = None) -> None:
    agent_only = (
        "fallback_policy", "max_hops", "gate_signal", "weak_distance", "topic_threshold",
        "orchestrator",
    )
    if param in agent_only and pipeline != Pipeline.agent:
        raise HTTPException(
            status_code=400, detail=f"{param} only applies to the agent pipeline"
        )
    if param == "model":
        bad = [v for v in values if not isinstance(v, str) or not _known_model(v)]
        if bad:
            raise HTTPException(status_code=400, detail=f"invalid model names: {bad}")
    elif param in ("topic_threshold", "weak_distance"):
        bad = [v for v in values if not isinstance(v, int | float) or not 0 <= v <= 2]
        if bad:
            raise HTTPException(status_code=400, detail=f"{param} must be 0..2: {bad}")
    elif param == "variant":
        # inside the chain: outside it every corpus sweep was refused as not an integer
        validate_axis_values("variant", values)
    elif param in ("fallback_policy", "gate_signal", "orchestrator"):
        enums = {
            "fallback_policy": FallbackPolicy, "gate_signal": GateSignal,
            "orchestrator": Orchestrator,
        }
        allowed = {p.value for p in enums[param]}
        if param == "orchestrator":
            allowed -= {o.value for o in GONE}
        bad = [v for v in values if v not in allowed]
        if bad:
            raise HTTPException(
                status_code=400, detail=f"{param} must be one of {sorted(allowed)}: {bad}"
            )
    elif param == "k":
        bad = [v for v in values if not isinstance(v, int) or not 1 <= v <= limits.MAX_K]
        if bad:
            raise HTTPException(status_code=400, detail=f"k must be 1..{limits.MAX_K}: {bad}")
    elif param == "max_hops":
        bad = [v for v in values if not isinstance(v, int) or not 1 <= v <= agent_policy.MAX_HOPS]
        if bad:
            raise HTTPException(
                status_code=400, detail=f"max_hops must be 1..{agent_policy.MAX_HOPS}: {bad}"
            )
    else:
        bad = [v for v in values if not isinstance(v, int) or v < 1]
        if bad:
            raise HTTPException(
                status_code=400, detail=f"{param} values must be positive integers"
            )


def _known_model(name: str) -> bool:
    try:
        refuse_unknown_registry(name)
    except ValueError:
        return False
    return True


def value_suffix(value) -> str:
    if isinstance(value, int):
        return f"{value:02d}"
    return re.sub(r"[^a-zA-Z0-9._-]", "_", str(value))


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
            "rerank": resolve_rerank(request.rerank),
            "k": request.k,
            "max_hops": request.max_hops,
            "model": request.model,
            "pipeline": request.pipeline.value,
            "language": request.language,
            "fallback_policy": request.fallback_policy and request.fallback_policy.value,
            "gate_signal": request.gate_signal and request.gate_signal.value,
            "weak_distance": request.weak_distance,
            "orchestrator": request.orchestrator and request.orchestrator.value,
            "topic_threshold": request.topic_threshold,
            "allow_cpu": request.allow_cpu,
            "variant": request.variant,
        },
    )


# copying here rather than in the judge job makes a taken name a 400, not a dead job
@router.post("/rejudge", response_model=RejudgeResponse)
def enqueue_rejudge(request: RejudgeRequest):
    try:
        copied = rejudge.copy_run(request.source, request.run_name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    try:
        job_id = job_queue.enqueue("judge_answers", {"run_name": request.run_name})
    except BaseException:
        # the copy is committed and the job is not, under a name no retry can reuse
        rejudge.delete_runs([request.run_name])
        raise
    return RejudgeResponse(job_id=job_id, run_name=request.run_name, copied=copied)


@router.post("/experiment", response_model=list[JobEnqueuedResponse])
async def enqueue_experiment(
    request: ExperimentRequest,
    session: AsyncSession = Depends(get_session),
):
    validate_param_values(request.param, request.values, request.pipeline)
    if request.variant:
        validate_axis_values("variant", [request.variant])
    base = (
        request.run_name
        or f"{request.set_name or 'all'}_{request.pipeline.value}_{int(time.time())}"
    )
    # what the row claims it filtered by: ids win over the set, as `_target_texts` reads them
    set_name = request.set_name if not request.question_ids else None
    rerank = resolve_rerank(request.rerank)
    jobs = []
    for value in request.values:
        job = await _enqueue(
            session,
            "eval_run",
            {
                "run_name": f"{base}_{request.param}_{value_suffix(value)}",
                "set_name": set_name,
                "question_ids": request.question_ids,
                "rerank": rerank,
                "pipeline": request.pipeline.value,
                "language": request.language,
                # the swept value wins: it comes after the pinned one
                "variant": request.variant,
                request.param: value,
            },
        )
        jobs.append(job)
    return jobs
