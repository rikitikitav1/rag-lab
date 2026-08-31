import math
import time
from datetime import datetime, timezone
from typing import Literal

import config
import job_queue
from crud import get_or_404
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.concurrency import run_in_threadpool
from models.eval import Question
from models.experiment import Experiment, ExperimentKind, ExperimentStatus, can_advance
from models.registry import Pipeline
from orm.async_db import commit_and_refresh, get_session
from pydantic import BaseModel, Field, model_validator
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import defer
from use_cases import rejudge, retrieval_compare
from use_cases.chat import resolve_rerank

from api.v1.eval import validate_axis_values, validate_param_values, value_suffix

router = APIRouter(prefix="/experiment", tags=["experiments"])

# `variant` belongs here for the same reason it is a retrieval axis: a generation
# experiment that cannot sweep the corpus is the one sweep this branch exists to run
GENERATION_PARAMS = frozenset({"k", "max_hops", "model", "variant"})

# one arm is one pass over the dataset, so the grid is a bill of hours the door can read
# before anything is enqueued. Grids run so far hold two to six arms
MAX_ARMS = 64


def refuse_oversized_grid(axes: dict[str, list]) -> None:
    arms = math.prod(len(values) for values in axes.values())
    if arms > MAX_ARMS:
        raise ValueError(
            f"{sorted((n, len(v)) for n, v in axes.items())} is {arms} arms,"
            f" over the cap of {MAX_ARMS}: sweep fewer axes or fewer values"
        )


class ExperimentCreate(BaseModel):
    name: str | None = Field(default=None, max_length=120)
    kind: ExperimentKind = ExperimentKind.generation
    # a rejudge names `source_run` instead: its answers already exist
    dataset: str | None = None
    # an upper bound, not a count: the arms drop questions with no embedding and no
    # marked source, and how many were actually measured is in results.procedure.questions
    sample_size: int | None = Field(default=None, ge=1, le=10000)
    sample_seed: int = 0
    question_ids: list[int] | None = Field(default=None, max_length=10000)
    # the corpus every arm reads unless `variant` is the swept parameter. Without it the
    # fan-out ran the configured default while the record said nothing about it
    variant: str | None = None
    data_prep: dict = Field(default_factory=dict)
    rerank: bool | None = None
    pipeline: Pipeline = Pipeline.single_shot
    language: Literal["ru", "en"] | None = None
    # the generation kind keeps its closed set; a comparison names one of its own axes,
    # and the validator below decides which rule applies
    param: str = "k"
    param_values: list[int | str] = Field(default_factory=list, max_length=MAX_ARMS)
    # a retrieval comparison moves several variables at once; param names the one it is
    # reported along, and param_values is filled from it so older readers keep working
    axes: dict[str, list] = Field(default_factory=dict)
    # a rejudge holds the answers still: this names the run whose rows every arm copies
    source_run: str | None = None

    @model_validator(mode="after")
    def _check(self):
        if self.kind == ExperimentKind.rejudge:
            if not self.source_run:
                raise ValueError("a rejudge needs source_run: the answers it re-reads")
            # a rejudge copies the whole run it names, so a sample is refused rather than
            # accepted and dropped: a discarded field reads as a plan nobody carried out
            if self.sample_size is not None or self.question_ids:
                raise ValueError("a rejudge copies a whole run; it takes no sample")
            rejudge.validate_axes(self.axes)
            refuse_oversized_grid(self.axes)
            if self.param not in self.axes:
                raise ValueError(
                    f"param must name one of the axes, got {self.param!r}"
                    f" against {sorted(self.axes)}"
                )
            names = [
                retrieval_compare.arm_name(a) for a in retrieval_compare.arms(self.axes)
            ]
            if len(set(names)) != len(names):
                raise ValueError(f"arms do not have distinct names: {sorted(names)}")
            self.param_values = list(self.axes[self.param])
            return self
        if not self.dataset:
            raise ValueError("dataset is required for this kind of experiment")
        if self.kind == ExperimentKind.generation:
            if self.param not in GENERATION_PARAMS:
                raise ValueError(
                    f"param must be one of {sorted(GENERATION_PARAMS)}, got {self.param!r}"
                )
            if not self.param_values:
                raise ValueError("param_values is required for a generation experiment")
            return self
        if not self.axes:
            raise ValueError("a retrieval comparison needs axes")
        unknown = sorted(set(self.axes) - retrieval_compare.AXES)
        if unknown:
            raise ValueError(f"unknown axes: {unknown}, known: {sorted(retrieval_compare.AXES)}")
        empty = sorted(name for name, values in self.axes.items() if not values)
        if empty:
            raise ValueError(f"axes with no values: {empty}")
        # names were checked and values were not, so an undeclared variant reached
        # aggregated carrying a confident delta over a corpus that does not exist
        for name, values in self.axes.items():
            validate_axis_values(name, values)
        if self.param not in self.axes:
            raise ValueError(
                f"param must name one of the axes, got {self.param!r} against {sorted(self.axes)}"
            )
        # the job refuses these too, and it refuses them after the row reached `running`
        # and the worker retried three times. They are properties of the request, so the
        # door is where they belong
        if self.param == "source":
            raise ValueError("source stratifies a comparison, it cannot be the axis of record")
        refuse_oversized_grid(self.axes)
        grid = retrieval_compare.arms(self.axes)
        names = [retrieval_compare.arm_name(arm) for arm in grid]
        if len(set(names)) != len(names):
            raise ValueError(f"arms do not have distinct names: {sorted(names)}")
        self.param_values = list(self.axes[self.param])
        return self


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


class ConclusionUpdate(BaseModel):
    conclusion: str


def _variants_without_rows(variants: list) -> list:
    import db

    # the default is what an arm without a variant reads, so it is checked like a named one
    named = list(variants) or [config.settings.corpus.variant]
    return [v for v in named if db.is_empty(variant=v)]


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
    if request.kind == ExperimentKind.generation:
        # the pipeline travels with the check: `max_hops` sits in GENERATION_PARAMS and
        # is agent-only, so without it the allow-list admitted a value the validator
        # refused for every request, agent ones included
        validate_param_values(request.param, request.param_values, request.pipeline)
        # the pinned corpus is checked like a swept one: an undeclared or empty variant
        # dies per question inside the runner, and it is knowable here
        if request.variant:
            validate_axis_values("variant", [request.variant])
    ids = await _resolve_sample(
        session,
        request.dataset,
        request.sample_size,
        request.sample_seed,
        request.question_ids,
    )
    # resolved here, not at run: a null recorded as the procedure means "nobody said",
    # and what it resolves to has already moved once under rows that read identically
    rerank = resolve_rerank(request.rerank)
    exp = Experiment(
        name=request.name,
        kind=request.kind,
        status=ExperimentStatus.draft,
        # a rejudge names its source run here: the column is not null, and the row is
        # flushed below, before the branch that used to assign it
        dataset=request.dataset or request.source_run,
        sample_size=request.sample_size,
        sample_seed=request.sample_seed,
        question_ids=ids,
        data_prep=request.data_prep,
        procedure={
            "pipeline": request.pipeline.value,
            "rerank": rerank,
            "language": request.language,
        },
        param=request.param,
        param_values=request.param_values,
        axes=request.axes,
    )
    session.add(exp)
    await session.flush()

    if request.kind == ExperimentKind.rejudge:
        arms = retrieval_compare.arms(request.axes)
        base = request.name or f"rejudge_{int(time.time())}"
        names = [f"{base}_{retrieval_compare.arm_name(a)}" for a in arms]
        # a pinned version the seed never loaded dies inside the judge, one log at a time,
        # while the job still reports done. Knowable here with one query
        missing = await run_in_threadpool(rejudge.unseeded_prompt_versions, request.axes)
        if missing:
            raise HTTPException(
                status_code=400, detail=f"no such judge prompt versions: {missing}"
            )
        # an unregistered judge would be pulled and deferred every thirty seconds for ever,
        # stranding the experiment: the arm names a judge, it does not order one
        unready = await run_in_threadpool(rejudge.judges_not_ready, request.axes)
        if unready:
            raise HTTPException(
                status_code=400, detail=f"these judges are not pulled and ready: {unready}"
            )
        try:
            await run_in_threadpool(
                rejudge.refuse_oversized_fanout, request.source_run, len(arms)
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        # the whole fan-out is one transaction: half-made copies commit while the
        # experiment rolls back, and the names are then burned for the identical retry
        try:
            await run_in_threadpool(rejudge.copy_runs, request.source_run, names)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        # a rejudge obeys none of these: its answers were produced by the source run, and
        # a record naming a pipeline it did not run is the defect the digests exist against
        exp.sample_size = None
        exp.question_ids = None
        exp.procedure = {
            "source_run": request.source_run,
            "base": base,
            # which arm each run name holds, written down rather than left to the order
            # the grid happens to produce: an arm added later moves that order
            "arms": rejudge.stored_arms(list(zip(arms, names, strict=True))),
        }
        exp.run_names = names
        exp.status = ExperimentStatus.running
        exp.started_at = datetime.now(timezone.utc)
        for arm_name, arm in zip(names, arms, strict=True):
            job_queue.add_job(
                session, "judge_answers", rejudge.arm_options(arm, arm_name)
            )
        try:
            await session.commit()
        except BaseException:
            # BaseException, not Exception: a client disconnect raises CancelledError,
            # which walks past `except Exception` and leaves the copies behind under names
            # `_refuse_bad_pair` then refuses for ever
            await run_in_threadpool(rejudge.delete_runs, names)
            raise
        await session.refresh(exp)
        return exp

    if request.kind == ExperimentKind.retrieval:
        # declared is not measurable: a policy in the config with no rows in the table
        # gives a confident delta over nothing. A query, so it goes to a thread
        empty = await run_in_threadpool(_variants_without_rows, request.axes.get("variant", []))
        if empty:
            raise HTTPException(
                status_code=400,
                detail=f"these variants are declared but hold no chunks: {empty}",
            )
        # one job for the whole grid: an arm is minutes and needs no card, and a job per
        # arm would have them writing the same results field from several workers
        exp.run_names = [retrieval_compare.arm_name(a) for a in retrieval_compare.arms(request.axes)]
        exp.status = ExperimentStatus.running
        exp.started_at = datetime.now(timezone.utc)
        job_queue.add_job(session, "compare_retrieval", {"experiment_id": exp.id})
        return await commit_and_refresh(session, exp)

    base = request.name or f"{request.dataset}_{request.pipeline.value}_{int(time.time())}"
    run_names = []
    for value in request.param_values:
        run_name = f"{base}_{request.param}_{value_suffix(value)}"
        job_queue.add_job(
            session,
            "eval_run",
            {
                "run_name": run_name,
                "set_name": request.dataset if ids is None else None,
                "question_ids": ids,
                "rerank": rerank,
                "pipeline": request.pipeline.value,
                "language": request.language,
                # the swept value wins: `request.param: value` comes after, so a sweep
                # over `variant` is not overwritten by the pinned one
                "variant": request.variant,
                request.param: value,
                "experiment_id": exp.id,
            },
        )
        run_names.append(run_name)

    exp.run_names = run_names
    exp.status = ExperimentStatus.running
    exp.started_at = datetime.now(timezone.utc)
    return await commit_and_refresh(session, exp)


class ArmsAdd(BaseModel):
    # arms one by one rather than a grid: the caller knows which arm it wants, and the
    # product of the widened axes would also name every arm the experiment already ran
    arms: list[dict] = Field(min_length=1)


@router.post("/{id}/arms", response_model=ExperimentResponse)
async def add_arms(
    id: int, request: ArmsAdd, session: AsyncSession = Depends(get_session)
):
    # locked because the status is read and then written, and the worker that aggregates
    # this experiment writes the same column from its own session
    exp = (
        await session.scalars(
            select(Experiment).where(Experiment.id == id).with_for_update()
        )
    ).first()
    if exp is None:
        raise HTTPException(status_code=404, detail=f"no experiment {id}")
    if exp.kind != ExperimentKind.rejudge:
        raise HTTPException(
            status_code=409,
            detail=f"only a rejudge gains arms without a rerun, this one is '{exp.kind}'",
        )
    # from `aggregated` only, and said as such rather than through `can_advance`, which
    # also admits `draft` and `failed`. A failed experiment has unjudged rows, so it would
    # take the arms, go back to `running` and never complete its series again
    if exp.status != ExperimentStatus.aggregated:
        raise HTTPException(
            status_code=409,
            detail=f"cannot add arms to an experiment in status '{exp.status}'"
            " (needs 'aggregated')",
        )
    try:
        rejudge.validate_arms(request.arms)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    base = (exp.procedure or {}).get("base") or exp.name or f"rejudge_{exp.id}"
    names = [f"{base}_{retrieval_compare.arm_name(a)}" for a in request.arms]
    taken = sorted(set(names) & set(exp.run_names))
    if taken:
        raise HTTPException(
            status_code=400, detail=f"the experiment already has these arms: {taken}"
        )
    if len(exp.run_names) + len(names) > retrieval_compare.GRID_CAP:
        raise HTTPException(
            status_code=400,
            detail=f"{len(exp.run_names)} arms plus {len(names)} is over the cap of"
            f" {retrieval_compare.GRID_CAP}",
        )
    added = rejudge.folded_axes(request.arms)
    missing = await run_in_threadpool(rejudge.unseeded_prompt_versions, added)
    if missing:
        raise HTTPException(status_code=400, detail=f"no such judge prompt versions: {missing}")
    unready = await run_in_threadpool(rejudge.judges_not_ready, added)
    if unready:
        raise HTTPException(
            status_code=400, detail=f"these judges are not pulled and ready: {unready}"
        )
    source_run = (exp.procedure or {}).get("source_run")
    if not source_run:
        raise HTTPException(
            status_code=409,
            detail=f"experiment {id} does not record the run its arms copy",
        )
    try:
        await run_in_threadpool(
            rejudge.refuse_oversized_fanout, source_run, len(names), len(exp.run_names)
        )
        await run_in_threadpool(rejudge.copy_runs, source_run, names)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    pairs = rejudge.paired_arms(exp) + list(zip(request.arms, names, strict=True))
    # reassigned rather than mutated: these are JSONB columns, and an in-place append
    # leaves the session with nothing to flush
    exp.procedure = (exp.procedure or {}) | {"arms": rejudge.stored_arms(pairs)}
    exp.axes = {
        name: values + [v for v in added.get(name, []) if v not in values]
        for name, values in ({**added, **exp.axes}).items()
    }
    exp.param_values = list(exp.axes.get(exp.param, exp.param_values))
    exp.run_names = exp.run_names + names
    # the old report stays: the new one overwrites it when the arms are judged, and an arm
    # that never finishes would otherwise leave no report at all. `finished_at` going to
    # null is what says it is not current
    exp.status = ExperimentStatus.running
    exp.finished_at = None
    exp.elapsed = None
    # elapsed is counted from here, or an experiment that gains an arm the next day
    # reports a day of work it did not do
    exp.started_at = datetime.now(timezone.utc)
    for arm, run_name in zip(request.arms, names, strict=True):
        job_queue.add_job(session, "judge_answers", rejudge.arm_options(arm, run_name))
    try:
        await session.commit()
    except BaseException:
        # BaseException: a disconnected client cancels the task, and CancelledError walks
        # past `except Exception` leaving copies under names nothing can reuse
        await run_in_threadpool(rejudge.delete_runs, names)
        raise
    await session.refresh(exp)
    return exp


# the list gives the shape of each experiment, never its contents: a retrieval record
# carries per-question rows for every arm, so a page of them is megabytes nobody asked for
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
    # a record now holds a row per question per arm, so the megabytes must not even
    # reach the ORM: the response model alone would load them and then drop them
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
