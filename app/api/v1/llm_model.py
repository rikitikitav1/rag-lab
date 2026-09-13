import engines
import job_queue
import samplers
from crud import get_or_404
from engines import answer_parsers, vllm
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.concurrency import run_in_threadpool
from models.jobs import Job
from models.registry import (
    MAX_MODEL_NAME,
    MODEL_NAME_RE,
    Engine,
    EngineKind,
    Model,
    ModelRole,
    Placement,
    Status,
    Weights,
    refuse_shared_cache_dir,
    refuse_unknown_registry,
)
from orm.async_db import commit_and_refresh, get_session
from pydantic import BaseModel, ConfigDict, Field, field_validator
from query_utils import Page, apply_in_filters, apply_sort_limit_offset
from sqlalchemy import exists, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from use_cases import weights_rules

from api.v1.job import JobResponse

router = APIRouter(prefix="/model", tags=["models"])


class ModelResponse(BaseModel):
    id: int
    name: str
    status: Status
    engine_id: int
    # two rows may share a name now, and a list that shows only the name cannot tell them apart
    engine: str
    # read from the server on the pull, never typed: absent means nobody asked, not "the same"
    quant: str | None = None
    size_bytes: int | None = None
    # one base set behind two rows, which is what a comparison across engines joins on
    weights: str | None = None
    answer_parser: str | None = None
    options: dict = {}

    model_config = {"from_attributes": True}

    @classmethod
    def of(cls, model: Model, engine: str, weights: str | None = None) -> "ModelResponse":
        return cls(
            id=model.id,
            name=model.name,
            status=model.status,
            engine_id=model.engine_id,
            engine=engine,
            quant=model.quant,
            size_bytes=model.size_bytes,
            weights=weights,
            answer_parser=model.answer_parser,
            options=model.options or {},
        )


# ollama's num_ctx and vLLM's max-model-len count the output too: a budget the size of the window leaves no input
def refuse_a_budget_over_the_window(engine_id: int, name: str, options: dict) -> None:
    budget = (options or {}).get("max_tokens")
    spec = engines.spec_of_id(engine_id) if budget else None
    if spec is None:
        return
    window = engines.window_or_configured(spec, name)
    if window is not None and budget >= window:
        raise ValueError(f"max_tokens {budget} fills the {window}-token window of {name} on {spec.name}: no room for the input")


def _refuse_remote(engine: Engine, name: str) -> None:
    try:
        weights_rules.refuse_remote(engine.kind, name)
    except engines.NotSupported as e:
        raise HTTPException(status_code=501, detail=str(e)) from e


async def _engine_of(session: AsyncSession, model: Model) -> Engine:
    return await get_or_404(Engine, model.engine_id, session)


@router.get("", response_model=list[ModelResponse])
async def list_models(
    id: list[int] | None = Query(default=None),
    name: list[str] | None = Query(default=None),
    status: list[Status] | None = Query(default=None),
    page: Page = Depends(),
    session: AsyncSession = Depends(get_session),
):
    stmt = apply_in_filters(
        select(Model, Engine.name, Weights.name)
        .join(Engine, Engine.id == Model.engine_id)
        .outerjoin(Weights, Weights.id == Model.weights_id),
        {Model.id: id, Model.name: name, Model.status: status},
    )

    stmt = apply_sort_limit_offset(
        stmt=stmt,
        sort_map={
            "id": Model.id,
            "name": Model.name,
            "status": Model.status,
            "engine": Engine.name,
        },
        sort_by=page.sort_by,
        sort_order=page.sort_order,
        limit=page.limit,
        offset=page.offset,
    )

    result = await session.execute(stmt)
    return [ModelResponse.of(model, engine, weights) for model, engine, weights in result.all()]


@router.get("/{id}", response_model=ModelResponse)
async def show_model(id: int, session: AsyncSession = Depends(get_session)):
    model = await get_or_404(Model, id, session)
    weights = await session.scalar(select(Weights.name).where(Weights.id == model.weights_id))
    return ModelResponse.of(model, (await _engine_of(session, model)).name, weights)


class ModelCreateRequest(BaseModel):
    name: str = Field(max_length=MAX_MODEL_NAME, pattern=MODEL_NAME_RE.pattern)
    # left out while one ollama engine holds everything; either key names the same engine
    engine_id: int | None = None
    engine: str | None = None

    @field_validator("name")
    @classmethod
    def _check_registry_host(cls, v: str) -> str:
        # a job that registers a model is a second door onto the same pull, refusing the same names
        refuse_unknown_registry(v)
        return v


# zero engines is a broken installation, two is a question only the caller can answer
async def _engine_for(
    session: AsyncSession, engine_id: int | None, name: str | None = None
) -> Engine:
    if name and engine_id is None:
        found = await session.scalar(select(Engine).where(Engine.name == name))
        if found is None:
            raise HTTPException(status_code=404, detail=f"no engine named {name}")
        return found
    if engine_id is None:
        known = list(await session.scalars(select(Engine.id).where(Engine.kind == EngineKind.ollama)))
        if not known:
            raise HTTPException(status_code=500, detail="no engine is registered")
        if len(known) > 1:
            raise HTTPException(status_code=400, detail="several engines; name the engine")
        engine_id = known[0]
    return await get_or_404(Engine, engine_id, session)


@router.post("", response_model=ModelResponse)
async def create_model(
    request: ModelCreateRequest,
    session: AsyncSession = Depends(get_session),
):
    engine = await _engine_for(session, request.engine_id, request.engine)
    taken = await session.scalar(
        select(exists().where(Model.engine_id == engine.id, Model.name == request.name))
    )
    if taken:
        raise HTTPException(
            status_code=409, detail=f"{request.name} is already registered on {engine.name}"
        )

    # an engine that does not pull is asked whether it already serves the name, before the row
    status = Status.available
    if engine.kind is not EngineKind.ollama:
        serves = await run_in_threadpool(_serves, engine, request.name)
        if serves is None:
            raise HTTPException(status_code=503, detail=f"{engine.name} does not answer")
        if not serves:
            # the same code as `/load`: the ask is well formed, and the server has no such model
            raise HTTPException(
                status_code=422, detail=f"{engine.name} does not serve {request.name}"
            )
        status = Status.ready

    model = Model(name=request.name, engine_id=engine.id, status=status)
    if engine.kind is EngineKind.vllm:
        # the bootstrap read these off the weights and the door did not, so its rows carried nulls
        seen = await run_in_threadpool(vllm.artifact_of, request.name)
        model.quant, model.size_bytes = seen.get("quant"), seen.get("size_bytes")
    session.add(model)
    if engine.kind is EngineKind.ollama:
        job_queue.add_job(
            session, "pull_llm_model", {"name": request.name, "engine_id": engine.id}, queue="io"
        )
    try:
        await commit_and_refresh(session, model)
    except IntegrityError as e:
        await session.rollback()
        raise HTTPException(
            status_code=409, detail=f"{request.name} is already registered on {engine.name}"
        ) from e
    return ModelResponse.of(model, engine.name)


# None when the server did not answer: a server that is down is a 503, not a model it lacks
def _serves(engine: Engine, name: str) -> bool | None:
    spec = engines.EngineSpec(
        engine.id, engine.name, engine.kind, engine.env_prefix, engine.placement
    )
    # seconds, not the completion client's two minutes and a retry
    try:
        seen = engines.served_models(spec)
    except engines.Unconfigured:
        return None
    return None if seen is None else name in seen


class LoadQueuedResponse(JobResponse):
    engine: str
    model: str

    model_config = {"protected_namespaces": ()}


# one road to the card, the queue: a load from this process met an awake judge with CUDA OOM
@router.post("/{id}/load", response_model=LoadQueuedResponse, status_code=202)
async def load_model(id: int, session: AsyncSession = Depends(get_session)):
    model = await get_or_404(Model, id, session)
    engine = await _engine_of(session, model)
    _refuse_remote(engine, model.name)
    if model.status != Status.ready:
        raise HTTPException(status_code=409, detail=f"{model.name} is {model.status.value}, not ready")
    if engine.kind is EngineKind.vllm:
        # on the processor it holds its model while it lives: a 202 that does nothing was a lie
        if engine.placement is Placement.cpu:
            raise HTTPException(status_code=409, detail=f"{engine.name} runs on the cpu: nothing to load")
        # a vLLM process serves one model, and a load of another would wake the wrong one with a 202
        serves = await run_in_threadpool(_serves, engine, model.name)
        if serves is None:
            raise HTTPException(status_code=503, detail=f"{engine.name} does not answer")
        if not serves:
            raise HTTPException(status_code=422, detail=f"{engine.name} does not serve {model.name}")
    # a load already waiting answers a second ask
    job_id = await run_in_threadpool(
        job_queue.pending_handover, engine.id, model.name
    ) or await run_in_threadpool(
        job_queue.enqueue, "hand_card", {"engine_id": engine.id, "model": model.name}
    )
    row = JobResponse.model_validate(await session.get(Job, job_id)).model_dump()
    return LoadQueuedResponse.model_validate({**row, "engine": engine.name, "model": model.name})


# what the server cannot say about its weights: fixed by hand, and a key only the hub vouches for
class ModelPatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    quant: str | None = Field(
        default=None, min_length=1, max_length=32, pattern=r"^[A-Za-z0-9_.+-]+$"
    )
    # each part starts with a letter or digit: `../..` and `a/..` are no hub repository
    weights: str | None = Field(
        default=None, max_length=200,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*$",
    )

    # `..` names no repository, and `--` shares another's cache directory
    @field_validator("weights")
    @classmethod
    def _one_hub_directory(cls, v: str | None) -> str | None:
        if v is not None:
            refuse_shared_cache_dir(v)
            if ".." in v:
                raise ValueError("weights names a hub repository: org/name, without `..`")
        return v

    # how this model's answers are cut on this engine, one of the parsers the stand has
    answer_parser: str | None = Field(default=None, max_length=100)
    # the model's own sampler over the role's, whole: `{}` clears it
    options: dict | None = None

    @field_validator("options")
    @classmethod
    def _sampler_keys_only(cls, v: dict | None) -> dict | None:
        return v if v is None else samplers.check(v)

    @field_validator("answer_parser")
    @classmethod
    def _a_known_parser(cls, v: str | None) -> str | None:
        if v is not None:
            answer_parsers.refuse_unknown(v)
        return v


@router.patch("/{id}", response_model=ModelResponse)
async def patch_model(
    id: int, request: ModelPatchRequest, session: AsyncSession = Depends(get_session)
):
    if request.quant is None and request.weights is None and request.answer_parser is None and request.options is None:
        raise HTTPException(status_code=422, detail="nothing to change: name a quant, weights, answer_parser or options")
    model = await get_or_404(Model, id, session)
    engine = await _engine_of(session, model)
    if request.quant is not None:
        model.quant = request.quant
    if request.answer_parser is not None:
        model.answer_parser = request.answer_parser
    if request.options is not None:
        try:
            await run_in_threadpool(refuse_a_budget_over_the_window, model.engine_id, model.name, request.options)
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e)) from e
        model.options = request.options
    # a quant-only patch answered `weights: null` for a row that had them
    weights = await session.scalar(select(Weights.name).where(Weights.id == model.weights_id))
    if request.weights is not None:
        # a key spelled by hand is free text again unless the hub knows the repository
        known = await run_in_threadpool(vllm.repo_exists, request.weights)
        if known is None:
            raise HTTPException(status_code=502, detail="the hub did not answer; try again")
        if not known:
            raise HTTPException(status_code=422, detail=f"{request.weights} is not on the hub")
        row = await session.scalar(select(Weights).where(Weights.name == request.weights))
        if row is None:
            row = Weights(name=request.weights)
            session.add(row)
            try:
                await session.flush()
            except IntegrityError as e:
                # two patches naming one new repository at once: the second one is asked to repeat
                await session.rollback()
                raise HTTPException(status_code=409, detail="the weights row was just made; repeat") from e
        model.weights_id = row.id
        weights = row.name
    await commit_and_refresh(session, model)
    return ModelResponse.of(model, engine.name, weights)


# only a 404 says "not on the hub": a 401, a 429 or a 5xx is a hub that did not answer the question
@router.delete("/{id}", response_model=ModelResponse)
async def delete_model(id: int, session: AsyncSession = Depends(get_session)):
    model = await get_or_404(Model, id, session)

    assigned = await session.scalar(select(exists().where(ModelRole.model_id == id)))
    if assigned:
        raise HTTPException(
            status_code=409,
            detail="model is assigned to a role; reassign the role first",
        )

    engine = await _engine_of(session, model)
    # the worker deletes vLLM weights too; only a remote engine keeps nothing here to delete
    _refuse_remote(engine, model.name)
    # the worker's own rule, asked before the row goes: a 200 here and a failed job later told two stories
    try:
        await run_in_threadpool(weights_rules.refuse_if_the_weights_are_shared, model.name, engine.id)
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e
    try:
        await run_in_threadpool(engines.driver(engine.kind).refuse_delete, model.name)
    except vllm.StillServed as e:
        raise HTTPException(status_code=409, detail=str(e)) from e

    name = model.name
    await session.delete(model)
    job_queue.add_job(
        session,
        "delete_llm_model",
        {"name": name, "engine_id": engine.id},
        queue="io",
    )
    await session.commit()

    return ModelResponse.of(model, engine.name)
