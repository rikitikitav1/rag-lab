import engines
import job_queue
from crud import get_or_404
from engines import ollama
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.concurrency import run_in_threadpool
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
    refuse_unknown_registry,
)
from orm.async_db import commit_and_refresh, get_session
from pydantic import BaseModel, Field, field_validator
from query_utils import Page, apply_in_filters, apply_sort_limit_offset
from sqlalchemy import exists, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

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
        )


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
        if not await run_in_threadpool(_serves, engine, request.name):
            raise HTTPException(
                status_code=400, detail=f"{engine.name} does not serve {request.name}"
            )
        status = Status.ready

    model = Model(name=request.name, engine_id=engine.id, status=status)
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


def _serves(engine: Engine, name: str) -> bool:
    spec = engines.EngineSpec(
        engine.id, engine.name, engine.kind, engine.env_prefix, engine.placement
    )
    try:
        return any(m.id == name for m in engines.client_for(spec).models.list().data)
    except Exception:
        return False


class LoadedResponse(BaseModel):
    model: str
    context_length: int | None

    model_config = {"protected_namespaces": ()}


@router.post("/{id}/load", response_model=LoadedResponse)
async def load_model(id: int, session: AsyncSession = Depends(get_session)):
    model = await get_or_404(Model, id, session)
    engine = await _engine_of(session, model)
    _refuse_unless_ollama(engine.kind, "loading")
    spec = engines.EngineSpec(
        engine.id, engine.name, engine.kind, engine.env_prefix, engine.placement
    )
    try:
        return await run_in_threadpool(ollama.load_into_memory, model.name, spec)
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e)) from e


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
    _refuse_unless_ollama(engine.kind, "deleting")

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


# one refusal, written in the engine layer; every door translates it into its own shape
def _refuse_unless_ollama(kind: EngineKind, doing: str) -> None:
    try:
        ollama.refuse_unless_ollama(_kind_only(kind), doing)
    except engines.NotSupported as e:
        raise HTTPException(status_code=501, detail=str(e)) from e


def _kind_only(kind: EngineKind) -> engines.EngineSpec:
    return engines.EngineSpec(0, "", kind, "", Placement.gpu)
