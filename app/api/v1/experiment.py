import math
import time
from datetime import datetime, timezone
from typing import Literal

import config
import job_queue
import limits
from crud import get_or_404
from evals import sampling
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.concurrency import run_in_threadpool
from models.eval import Question
from models.experiment import Experiment, ExperimentKind, ExperimentStatus, can_advance
from models.registry import Pipeline
from orm.async_db import commit_and_refresh, get_session
from pydantic import BaseModel, Field, model_validator
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import defer
from use_cases import rejudge, retrieval_compare
from use_cases.chat import resolve_rerank

from api.v1.eval import validate_axis_values, validate_param_values, value_suffix

router = APIRouter(prefix="/experiment", tags=["experiments"])

# the closed set a generation experiment may sweep; `variant` joined it with the corpus sweep
GENERATION_PARAMS = frozenset({"k", "max_hops", "model", "variant"})

# one arm is one pass over the dataset: a bill of hours the door reads before enqueueing
MAX_ARMS = retrieval_compare.GRID_CAP


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
    # an upper bound: what was measured is in results.procedure.questions
    sample_size: int | None = Field(default=None, ge=1, le=limits.MAX_QUESTION_IDS)
    sample_seed: int = 0
    question_ids: list[int] | None = Field(
        default=None, max_length=limits.MAX_QUESTION_IDS
    )
    # the corpus every arm reads unless `variant` is the swept parameter
    variant: str | None = None
    data_prep: dict = Field(default_factory=dict)
    rerank: bool | None = None
    pipeline: Pipeline = Pipeline.single_shot
    language: Literal["ru", "en"] | None = None
    # the generation kind keeps its closed set; a comparison names one of its own axes
    param: str = "k"
    param_values: list[int | str] = Field(default_factory=list, max_length=MAX_ARMS)
    # param names the axis it is reported along, and param_values is filled from it
    axes: dict[str, list] = Field(default_factory=dict)
    # a rejudge holds the answers still: this names the run whose rows every arm copies
    source_run: str | None = None
    # a rejudge with no arm reproducing the source's judge reads its drift as an effect
    unpaired: bool = False
    # arms one by one, for the shape `axes` cannot describe: the grid is their cross product
    arms: list[dict] | None = Field(default=None, max_length=MAX_ARMS)
    # a control does not need every row: about a third of an arm's wall clock on three axes
    control_sample: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def _check(self):
        rejudge_only = {"unpaired": False, "arms": None, "control_sample": None}
        if self.kind != ExperimentKind.rejudge:
            named = [k for k, empty in rejudge_only.items() if getattr(self, k) != empty]
            if named:
                raise ValueError(f"{sorted(named)} apply to a rejudge, and this is a {self.kind}")
        if self.kind == ExperimentKind.rejudge:
            if not self.source_run:
                raise ValueError("a rejudge needs source_run: the answers it re-reads")
            if self.sample_size is not None and self.question_ids:
                raise ValueError("a rejudge takes either a sample size or ids, not both")
            if self.arms is not None:
                if self.axes:
                    raise ValueError("a rejudge takes either axes or arms, not both")
                rejudge.validate_arms(self.arms)
            else:
                rejudge.validate_axes(self.axes)
                refuse_oversized_grid(self.axes)
                rejudge.refuse_repeated_names(retrieval_compare.arms(self.axes))
            # named arms are checked against what they move, not against the `axes` they leave empty
            axes = rejudge.folded_axes(self.arms) if self.arms is not None else self.axes
            if self.param not in axes:
                raise ValueError(
                    f"param must name one of the axes, got {self.param!r}"
                    f" against {sorted(axes)}"
                )
            self.param_values = list(axes[self.param])
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
            if self.axes:
                raise ValueError("axes belong to a retrieval comparison or a rejudge")
            return self
        if not self.axes:
            raise ValueError("a retrieval comparison needs axes")
        unknown = sorted(set(self.axes) - retrieval_compare.AXES)
        if unknown:
            raise ValueError(f"unknown axes: {unknown}, known: {sorted(retrieval_compare.AXES)}")
        empty = sorted(name for name, values in self.axes.items() if not values)
        if empty:
            raise ValueError(f"axes with no values: {empty}")
        # names were checked and values were not, so an undeclared variant reached aggregated
        for name, values in self.axes.items():
            validate_axis_values(name, values)
        if self.param not in self.axes:
            raise ValueError(
                f"param must name one of the axes, got {self.param!r} against {sorted(self.axes)}"
            )
        # the job refuses these too, after the row reached `running` and three retries
        if self.param == "source":
            raise ValueError("source stratifies a comparison, it cannot be the axis of record")
        refuse_oversized_grid(self.axes)
        rejudge.refuse_repeated_names(retrieval_compare.arms(self.axes))
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
        .order_by(sampling.by_id_and_seed(Question.id, sample_seed))
        .limit(sample_size)
    )
    return list(await session.scalars(stmt))


@router.post("", response_model=ExperimentResponse)
async def create_experiment(
    request: ExperimentCreate, session: AsyncSession = Depends(get_session)
):
    if request.kind == ExperimentKind.generation:
        # `max_hops` is agent-only: without the pipeline the allow-list admitted a refused value
        validate_param_values(request.param, request.param_values, request.pipeline)
        # the pinned corpus is checked like a swept one, and it is knowable here
        if request.variant:
            validate_axis_values("variant", [request.variant])
    # a rejudge draws from the source run: `dataset=None` compiled to `set_name IS NULL`
    ids = (
        None
        if request.kind == ExperimentKind.rejudge
        else await _resolve_sample(
            session,
            request.dataset,
            request.sample_size,
            request.sample_seed,
            request.question_ids,
        )
    )
    # resolved here, not at run: a null in the procedure means "nobody said"
    rerank = resolve_rerank(request.rerank)
    exp = Experiment(
        name=request.name,
        kind=request.kind,
        status=ExperimentStatus.draft,
        # the column is not null and the row is flushed below, before the branch that assigned it
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
        arms = request.arms if request.arms is not None else retrieval_compare.arms(request.axes)
        axes = rejudge.folded_axes(arms) if request.arms is not None else request.axes
        # sampled from the source's own rows: a rejudge names a run, not a question set
        sample = request.question_ids or (
            await run_in_threadpool(
                rejudge.sample_of, request.source_run, request.sample_size, request.sample_seed
            )
            if request.sample_size is not None
            else None
        )
        exp.question_ids = sample
        base = request.name or f"rejudge_{int(time.time())}"
        names = [f"{base}_{retrieval_compare.arm_name(a)}" for a in arms]
        try:
            retrieval_compare.refuse_long_names(names)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        # a pinned version the seed never loaded dies inside the judge, one log at a time
        missing = await run_in_threadpool(rejudge.unseeded_prompt_versions, axes)
        if missing:
            raise HTTPException(
                status_code=400, detail=f"no such judge prompt versions: {missing}"
            )
        # an unregistered judge is deferred for ever: the arm names a judge, it does not order one
        unready = await run_in_threadpool(rejudge.judges_not_ready, axes)
        if unready:
            raise HTTPException(
                status_code=400, detail=f"these judges are not pulled and ready: {unready}"
            )
        try:
            await run_in_threadpool(
                rejudge.refuse_oversized_fanout,
                request.source_run,
                len(arms),
                existing=0,
                question_ids=sample,
            )
            await run_in_threadpool(
                rejudge.refuse_unpaired_rejudge, request.source_run, arms, request.unpaired
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        # one transaction: half-made copies would burn the names for the identical retry
        try:
            await run_in_threadpool(rejudge.copy_runs, request.source_run, names, sample)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        # a rejudge obeys none of these: its answers were produced by the source run
        exp.procedure = {
            "source_run": request.source_run,
            "base": base,
            # which arm each run name holds, rather than the order the grid happens to produce
            "arms": rejudge.stored_arms(list(zip(arms, names, strict=True))),
            # an arm added later is judged the way these were, and this is where that is written
            "control_sample": request.control_sample,
            "control_seed": request.sample_seed,
        }
        exp.run_names = names
        # the axes the arms actually moved: a diagonal folded back into axes reads as a grid
        exp.axes = axes
        exp.param_values = list(axes.get(exp.param, exp.param_values))
        exp.status = ExperimentStatus.running
        exp.started_at = datetime.now(timezone.utc)
        for arm_name, arm in zip(names, arms, strict=True):
            job_queue.add_job(
                session,
                "judge_answers",
                rejudge.arm_options(
                    arm, arm_name, request.control_sample, request.sample_seed
                ),
            )
        try:
            await session.commit()
        except BaseException:
            # BaseException: a client disconnect raises CancelledError and leaves the copies behind
            await run_in_threadpool(rejudge.delete_runs, names)
            raise
        await session.refresh(exp)
        return exp

    if request.kind == ExperimentKind.retrieval:
        # declared is not measurable: a policy with no rows gives a confident delta over nothing
        empty = await run_in_threadpool(_variants_without_rows, request.axes.get("variant", []))
        if empty:
            raise HTTPException(
                status_code=400,
                detail=f"these variants are declared but hold no chunks: {empty}",
            )
        # one job for the whole grid: a job per arm would write one results field from several
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
                # the swept value wins: a sweep over `variant` is not overwritten by the pinned one
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
    # arms one by one: the product of the widened axes would name the arms already run
    arms: list[dict] = Field(min_length=1, max_length=MAX_ARMS)
    unpaired: bool = False


@router.post("/{id}/arms", response_model=ExperimentResponse)
async def add_arms(
    id: int, request: ArmsAdd, session: AsyncSession = Depends(get_session)
):
    # locked: the status is read and then written, and the aggregating worker writes it too
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
    # from `aggregated` only: `can_advance` admits it from `draft` and `failed` too, both unjudged
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
    try:
        retrieval_compare.refuse_long_names(names)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
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
    # the arms already on the experiment were built on these
    sample = exp.question_ids
    control_sample = (exp.procedure or {}).get("control_sample")
    # a row from before the writer carries no key at all; a recorded 0 is a seed, not a gap
    recorded_seed = (exp.procedure or {}).get("control_seed")
    control_seed = 0 if recorded_seed is None else recorded_seed
    try:
        await run_in_threadpool(
            rejudge.refuse_oversized_fanout,
            source_run,
            len(names),
            len(exp.run_names),
            sample,
        )
        # the arms already on it count as controls: one may be the repeat this call would add
        await run_in_threadpool(
            rejudge.refuse_unpaired_rejudge,
            source_run,
            request.arms + [a["arm"] for a in (exp.procedure or {}).get("arms") or []],
            request.unpaired,
        )
        await run_in_threadpool(rejudge.copy_runs, source_run, names, sample)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    # the copies are committed by now: every exit before the new commit must take them back
    try:
        pairs = rejudge.paired_arms(exp) + list(zip(request.arms, names, strict=True))
        # reassigned, not mutated: an in-place append on JSONB leaves nothing to flush
        exp.procedure = (exp.procedure or {}) | {"arms": rejudge.stored_arms(pairs)}
        exp.axes = {
            name: values + [v for v in added.get(name, []) if v not in values]
            for name, values in ({**added, **exp.axes}).items()
        }
        if exp.param not in exp.axes:
            raise HTTPException(
                status_code=409,
                detail=f"the experiment reads '{exp.param}' and the merged arms do not move it",
            )
        exp.param_values = list(exp.axes[exp.param])
        exp.run_names = exp.run_names + names
        # the old report stays until the new arms are judged; `finished_at` null says so
        exp.status = ExperimentStatus.running
        exp.finished_at = None
        exp.elapsed = None
        # or an experiment that gains an arm later reports work it did not do
        exp.started_at = datetime.now(timezone.utc)
        for arm, run_name in zip(request.arms, names, strict=True):
            job_queue.add_job(
                session,
                "judge_answers",
                rejudge.arm_options(arm, run_name, control_sample, control_seed),
            )
        await session.commit()
    except BaseException:
        # BaseException: a cancelled task leaves copies under names nothing can reuse
        await run_in_threadpool(rejudge.delete_runs, names)
        raise
    await session.refresh(exp)
    return exp


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
