from datetime import datetime

from crud import get_or_404
from fastapi import APIRouter, Depends, HTTPException, Query
from models.experiment import Experiment, ExperimentKind, ExperimentStatus, can_advance
from orm.async_db import commit_and_refresh, get_session
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import defer
from use_cases import experiment_report, experiment_setup

router = APIRouter(prefix="/experiment", tags=["experiments"])

class ExperimentResponse(BaseModel):
    id: int
    name: str | None
    kind: ExperimentKind
    status: ExperimentStatus
    dataset: str | None
    sample_size: int | None
    sample_seed: int | None
    question_ids: list[int] | None
    data_prep: dict
    procedure: dict
    param: str
    param_values: list
    axes: dict
    run_names: list
    results: dict | None
    conclusion: str | None
    started_at: datetime | None
    finished_at: datetime | None
    elapsed: float | None

    model_config = {"from_attributes": True}


# the request is the use case's own spec, checked whole wherever it comes from
ExperimentCreate = experiment_setup.ExperimentSpec
ArmsAdd = experiment_setup.ArmsSpec
MAX_ARMS = experiment_setup.MAX_ARMS


class ConclusionUpdate(BaseModel):
    conclusion: str


@router.post("", response_model=ExperimentResponse)
async def create_experiment(request: ExperimentCreate, session: AsyncSession = Depends(get_session)):
    return await experiment_setup.create(session, request)


@router.post("/{id}/arms", response_model=ExperimentResponse)
async def add_arms(id: int, request: ArmsAdd, session: AsyncSession = Depends(get_session)):
    return await experiment_setup.add_arms(session, id, request)


# the shape of each experiment, never its contents: a page of them would be megabytes
class ExperimentListed(BaseModel):
    id: int
    name: str | None
    kind: ExperimentKind
    status: ExperimentStatus
    dataset: str
    sample_size: int | None
    param: str
    param_values: list
    axes: dict
    run_names: list
    conclusion: str | None
    started_at: datetime | None
    finished_at: datetime | None
    elapsed: float | None

    model_config = {"from_attributes": True}


@router.get("", response_model=list[ExperimentListed])
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
    # the megabytes must not reach the ORM: the response model would load and drop them
    stmt = (
        stmt.options(defer(Experiment.results))
        .order_by(Experiment.id.desc())
        .limit(limit)
        .offset(offset)
    )
    return (await session.scalars(stmt)).all()


@router.get("/{id}", response_model=ExperimentResponse)
async def show_experiment(id: int, session: AsyncSession = Depends(get_session)):
    return await get_or_404(Experiment, id, session)


# the report the MCP door reads too: the arms, their judges and the paired deltas, without halves and seeds
@router.get("/{id}/report")
async def experiment_report_of(id: int, pair: str | None = None, session: AsyncSession = Depends(get_session)) -> dict:
    return await session.run_sync(experiment_report.report_of, id, pair)


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
