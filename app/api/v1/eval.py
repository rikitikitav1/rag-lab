import re
import time
from enum import StrEnum
from typing import Literal

import config
import job_queue
import job_specs
import limits
import logging_setup
from evals import compare as compare_uc
from evals import retrieval_metrics
from evals.guest_axes import MESSAGE_FORMS
from evals.pools import Ambiguous
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.concurrency import run_in_threadpool
from models import Job
from models.eval import Question, QuestionLog
from models.registry import MAX_MODEL_NAME, MODEL_NAME_RE, Pipeline, refuse_unknown_registry
from orm.async_db import commit_and_refresh, get_session
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
from use_cases import agent_policy, rejudge, retrieval_compare
from use_cases.agent_policy import GONE, FallbackPolicy, GateSignal, Orchestrator
from use_cases.chat import resolve_rerank
from use_cases.index import VARIANT_RE

# a door that queues a job answers with the whole row, the same one `POST /v1/job` answers with
from api.v1.job import JobResponse as JobEnqueuedResponse

# what a run may ask for is not what a log may hold: both retired arms stay queryable
RunnableOrchestrator = StrEnum(
    "RunnableOrchestrator",
    {o.name: o.value for o in Orchestrator if o not in GONE},
)

log = logging_setup.get_logger(__name__)

router = APIRouter(prefix="/eval", tags=["eval"])


class RejudgeRequest(BaseModel):
    source: str = Field(min_length=1, max_length=limits.MAX_RUN_NAME)
    run_name: str = Field(min_length=1, max_length=limits.MAX_RUN_NAME)


class RejudgeResponse(JobEnqueuedResponse):
    run_name: str
    copied: int


class GuestAxesRequest(BaseModel):
    run_name: str = Field(min_length=1, max_length=limits.MAX_RUN_NAME)
    judge_width: int | None = Field(default=None, ge=1, le=limits.MAX_RUNS)
    # the guests calibrate on a subsample, they are not an axis
    sample: int | None = Field(default=None, ge=1, le=limits.MAX_GUEST_ROWS)
    seed: int | None = None
    messages: Literal[*MESSAGE_FORMS] = MESSAGE_FORMS[0]
    guest_model: str | None = Field(default=None, max_length=MAX_MODEL_NAME, pattern=MODEL_NAME_RE.pattern)


class JudgeRequest(BaseModel):
    run_name: str = Field(min_length=1, max_length=limits.MAX_RUN_NAME)
    judge_width: int | None = Field(default=None, ge=1, le=limits.MAX_RUNS)


class LanguageProbeRequest(BaseModel):
    run_name: str = Field(default="arc3_agent_baseline", max_length=limits.MAX_RUN_NAME)
    rows: int = Field(default=40, ge=1, le=limits.MAX_GUEST_ROWS)
    # a declared cut is named, and `rows` would take the first of the pool instead
    log_ids: list[int] | None = Field(default=None, max_length=limits.MAX_GUEST_ROWS)


class ParaphraseRequest(BaseModel):
    limit: int | None = 100
    source: str | None = None
    set_name: str = "paraphrased"
    seed: str = ""
    per_source: int | None = None
    grow: bool = False


# the door is the queue's own model with the name made optional: it invents one when none came
class EvalRunRequest(job_specs.EvalRunFields):
    run_name: str | None = Field(default=None, max_length=limits.MAX_RUN_NAME)
    orchestrator: RunnableOrchestrator | None = None


class ExperimentRequest(BaseModel):
    run_name: str | None = Field(default=None, max_length=limits.MAX_RUN_NAME)
    set_name: str | None = None
    question_ids: limits.QuestionIds = Field(default=None, max_length=limits.MAX_QUESTION_IDS)
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


# `safely` returns no counts when the diagnostic itself failed, and the door read them anyway
def _debts_or_none(run_name: str):
    from evals.run_debts import safely

    debts = safely(run_name)
    if "unavailable" in debts:
        # the handler decides on the rows it finds: a broken diagnostic must not refuse real work
        log.warning("eval.debts_unavailable", run_name=run_name, why=debts["unavailable"])
        return None
    return debts


async def _enqueue(session, type: str, options: dict) -> JobEnqueuedResponse:
    job = job_queue.add_job(session, type, options)
    await commit_and_refresh(session, job)
    return JobEnqueuedResponse.model_validate(job)


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
    # the door used to drop these: a caller read two arms the code itself calls incomparable
    schema_version: int = Field(alias="schema")
    residency: dict
    answering_engines_by_run: dict
    correlation_population: dict
    verdicts: dict | None

    model_config = {"populate_by_name": True}


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
    try:
        return compare_uc.compare(loaded)
    except Ambiguous as e:
        # the run cannot be paired at all, and the reason names the question: a 500 hid it
        raise HTTPException(status_code=409, detail=str(e)) from e


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


async def _rows_of(session, run_name: str) -> int:
    return await session.scalar(
        select(func.count()).select_from(QuestionLog).where(QuestionLog.run_name == run_name)
    ) or 0


async def _question_ids_in(session, ids: list[int]) -> set[int]:
    return set((await session.scalars(select(Question.id).where(Question.id.in_(ids)))).all())


# refused at the door, not an hour in: a run over fewer questions than named reads as the named set
async def _refuse_missing_questions(session, ids: list[int]) -> None:
    missing = sorted(set(ids) - await _question_ids_in(session, ids))
    if missing:
        raise HTTPException(
            status_code=422,
            detail=f"{len(missing)} of {len(ids)} question ids are not in the stand: {missing[:20]}",
        )


async def _eval_runs_named(session, run_name: str) -> list:
    return list((await session.scalars(
        select(Job).where(Job.type == "eval_run", Job.options["run_name"].astext == run_name).order_by(Job.id.desc())
    )).all())


# a taken name, by its rows or by a job that stopped before its first row, is resumed, not run twice
async def refuse_a_taken_run(session, run_name: str) -> None:
    rows = await _rows_of(session, run_name)
    jobs = await _eval_runs_named(session, run_name)
    if rows or jobs:
        raise HTTPException(
            status_code=409,
            detail=f"run {run_name} has {rows} rows and {len(jobs)} eval_run jobs: pass resume"
            " to answer the rest on its own options, or name a new run",
        )


# a resumed run changes nothing: it runs on the stopped job's own options and asks only the unanswered
async def _resume(session, request: EvalRunRequest) -> JobEnqueuedResponse:
    extra = sorted(request.model_fields_set - {"run_name", "resume"})
    if not request.run_name or extra:
        raise HTTPException(
            status_code=422,
            detail="resume takes run_name alone: a resumed run changes nothing"
            + (f", and {', '.join(extra)} would" if extra else ""),
        )
    jobs = await _eval_runs_named(session, request.run_name)
    if not jobs:
        raise HTTPException(status_code=404, detail=f"no eval_run named {request.run_name} to resume")
    if any(job.status in job_queue.ACTIVE for job in jobs):
        raise HTTPException(status_code=409, detail=f"run {request.run_name} is still queued or running")
    # the worker's own bookkeeping belongs to the attempt that stopped, not to the resumed one
    options = {k: v for k, v in jobs[0].options.items() if k not in job_specs.WORKER_KEYS}
    return await _enqueue(session, "eval_run", {**options, "resume": True})


@router.post("/run", response_model=JobEnqueuedResponse)
async def enqueue_eval_run(
    request: EvalRunRequest,
    session: AsyncSession = Depends(get_session),
):
    if request.resume:
        return await _resume(session, request)
    if request.run_name:
        await refuse_a_taken_run(session, request.run_name)
    if request.question_ids:
        await _refuse_missing_questions(session, request.question_ids)
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
            "restate_tools": request.restate_tools,
            "variant": request.variant,
            "resume": False,
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
    row = JobEnqueuedResponse.model_validate(job_queue.get(job_id)).model_dump()
    return RejudgeResponse.model_validate({**row, "run_name": request.run_name, "copied": copied})


# `/rejudge` judges a copy; a run judged in place had no door, and the queue was filled by hand
@router.post("/judge", response_model=JobEnqueuedResponse)
async def enqueue_judge(
    request: JudgeRequest,
    session: AsyncSession = Depends(get_session),
):
    debts = await run_in_threadpool(_debts_or_none, request.run_name)
    if debts is not None and not debts["answered_rows"]:
        raise HTTPException(
            status_code=404, detail=f"run {request.run_name} holds no answered row"
        )
    if debts is not None and not debts["ours_still_to_judge"]:
        raise HTTPException(
            status_code=404, detail=f"run {request.run_name} owes our judge nothing"
        )
    return await _enqueue(
        session,
        "judge_answers",
        {"run_name": request.run_name, "judge_width": request.judge_width},
    )


# two calls a row and no card guard beside the judge: it queues rather than running host side
@router.post("/language-probe", response_model=JobEnqueuedResponse)
async def enqueue_language_probe(
    request: LanguageProbeRequest,
    session: AsyncSession = Depends(get_session),
):
    debts = await run_in_threadpool(_debts_or_none, request.run_name)
    if debts is not None and not debts["answered_rows"]:
        raise HTTPException(
            status_code=404, detail=f"run {request.run_name} holds no answered row"
        )
    return await _enqueue(
        session,
        "judge_language",
        {"run_name": request.run_name, "rows": request.rows, "log_ids": request.log_ids},
    )


# by request only: the guests cost 6.62x our three axes, so no run and no arm asks for them
@router.post("/guest-axes", response_model=JobEnqueuedResponse)
async def enqueue_guest_axes(
    request: GuestAxesRequest,
    session: AsyncSession = Depends(get_session),
):
    from job_handlers.judging import guest_pass_refusal

    refused = await run_in_threadpool(guest_pass_refusal, request.run_name, request.sample)
    if refused:
        raise HTTPException(status_code=refused[0], detail=refused[1])
    return await _enqueue(
        session,
        "judge_guest_axes",
        {"run_name": request.run_name, "judge_width": request.judge_width,
         "sample": request.sample, "seed": request.seed, "messages": request.messages,
         "guest_model": request.guest_model},
    )


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
    names = [f"{base}_{request.param}_{value_suffix(value)}" for value in request.values]
    # all of them before any is queued: a client's retry after a timeout wrote every question twice
    for name in names:
        await refuse_a_taken_run(session, name)
    jobs = []
    for value, name in zip(request.values, names, strict=True):
        job = await _enqueue(
            session,
            "eval_run",
            {
                "run_name": name,
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
