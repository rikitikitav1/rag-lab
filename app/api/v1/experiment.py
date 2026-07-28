import time
from datetime import datetime, timezone
from typing import Literal

import job_queue
from crud import get_or_404
from fastapi import APIRouter, Depends, HTTPException, Query
from models.eval import Question
from models.experiment import Experiment, ExperimentStatus, can_advance
from models.registry import Pipeline
from orm.async_db import commit_and_refresh, get_session
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

router = APIRouter(prefix="/experiment", tags=["experiments"])


class ExperimentCreate(BaseModel):
    name: str | None = None
    dataset: str
    sample_size: int | None = None
    sample_seed: int = 0
    question_ids: list[int] | None = None
    data_prep: dict = Field(default_factory=dict)
    rerank: bool | None = None
    pipeline: Pipeline = Pipeline.single_shot
    language: Literal["ru", "en"] | None = None
    param: Literal["k", "max_hops"] = "k"
    param_values: list[int] = Field(min_length=1)


class ExperimentResponse(BaseModel):
    id: int
    name: str | None
    status: ExperimentStatus
    dataset: str
    sample_size: int | None
    sample_seed: int | None
    question_ids: list[int] | None
    data_prep: dict
    procedure: dict
    param: str
    param_values: list
    run_names: list
    results: dict | None
    conclusion: str | None
    started_at: datetime | None
    finished_at: datetime | None
    elapsed: float | None

    model_config = {"from_attributes": True}


class ConclusionUpdate(BaseModel):
    conclusion: str


async def _resolve_sample(
    session: AsyncSession,
    dataset: str,
    sample_size: int | None,
    sample_seed: int,
    question_ids: list[int] | None,
) -> list[int] | None:
    if question_ids:
        return question_ids
    if sample_size is None:
        return None
    stmt = (
        select(Question.id)
        .where(Question.set_name == dataset)
        .order_by(func.md5(func.concat(Question.id, sample_seed)))
        .limit(sample_size)
    )
    return list(await session.scalars(stmt))


@router.post("", response_model=ExperimentResponse)
async def create_experiment(
    request: ExperimentCreate, session: AsyncSession = Depends(get_session)
):
    ids = await _resolve_sample(
        session,
        request.dataset,
        request.sample_size,
        request.sample_seed,
        request.question_ids,
    )
    exp = Experiment(
        name=request.name,
        status=ExperimentStatus.draft,
        dataset=request.dataset,
        sample_size=request.sample_size,
        sample_seed=request.sample_seed,
        question_ids=ids,
        data_prep=request.data_prep,
        procedure={
            "pipeline": request.pipeline.value,
            "rerank": request.rerank,
            "language": request.language,
        },
        param=request.param,
        param_values=request.param_values,
    )
    session.add(exp)
    await session.flush()

    base = request.name or f"{request.dataset}_{request.pipeline.value}_{int(time.time())}"
    run_names = []
    for value in request.param_values:
        run_name = f"{base}_{request.param}{value:02d}"
        job_queue.add_job(
            session,
            "eval_run",
            {
                "run_name": run_name,
                "set_name": request.dataset if ids is None else None,
                "question_ids": ids,
                "rerank": request.rerank,
                "pipeline": request.pipeline.value,
                "language": request.language,
                request.param: value,
                "experiment_id": exp.id,
            },
        )
        run_names.append(run_name)

    exp.run_names = run_names
    exp.status = ExperimentStatus.running
    exp.started_at = datetime.now(timezone.utc)
    return await commit_and_refresh(session, exp)


@router.get("", response_model=list[ExperimentResponse])
async def list_experiments(
    status: list[ExperimentStatus] | None = Query(default=None),
    dataset: str | None = Query(default=None),
    param: str | None = Query(default=None),
    limit: int = Query(default=25, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    session: AsyncSession = Depends(get_session),
):
    stmt = select(Experiment)
    if status:
        stmt = stmt.where(Experiment.status.in_(status))
    if dataset:
        stmt = stmt.where(Experiment.dataset == dataset)
    if param:
        stmt = stmt.where(Experiment.param == param)
    stmt = stmt.order_by(Experiment.id.desc()).limit(limit).offset(offset)
    return (await session.scalars(stmt)).all()


@router.get("/{id}", response_model=ExperimentResponse)
async def show_experiment(id: int, session: AsyncSession = Depends(get_session)):
    return await get_or_404(Experiment, id, session)


@router.put("/{id}/conclusion", response_model=ExperimentResponse)
async def conclude_experiment(
    id: int,
    request: ConclusionUpdate,
    session: AsyncSession = Depends(get_session),
):
    exp = await get_or_404(Experiment, id, session)
    if not can_advance(exp.status, ExperimentStatus.concluded):
        raise HTTPException(
            status_code=409,
            detail=f"cannot conclude an experiment in status '{exp.status}' (needs 'aggregated')",
        )
    exp.conclusion = request.conclusion
    exp.status = ExperimentStatus.concluded
    return await commit_and_refresh(session, exp)
