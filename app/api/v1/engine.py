import engines
from crud import get_or_404
from fastapi import APIRouter, Depends, HTTPException
from fastapi.concurrency import run_in_threadpool
from models.registry import Engine, EngineKind, Model, Placement
from orm.async_db import commit_and_refresh, get_session
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import exists, select
from sqlalchemy.ext.asyncio import AsyncSession

router = APIRouter(prefix="/engine", tags=["engines"])

# a prefix is interpolated into an environment variable name, so it is not free text
PREFIX = r"^[A-Z][A-Z0-9_]{0,31}$"


class EngineResponse(BaseModel):
    id: int
    name: str
    kind: EngineKind
    env_prefix: str
    placement: Placement
    # read from the environment, never from the row, and shown so a reader can tell it is set
    address: str | None
    reachable: bool | None

    model_config = {"from_attributes": True}

    @classmethod
    def of(cls, row: Engine, address: str | None, reachable: bool | None = None):
        return cls(
            id=row.id, name=row.name, kind=row.kind, env_prefix=row.env_prefix,
            placement=row.placement, address=address, reachable=reachable,
        )


def _address(row: Engine) -> str | None:
    try:
        return engines.address_of(_spec(row))
    except engines.Unconfigured:
        return None


def _spec(row: Engine) -> engines.EngineSpec:
    return engines.EngineSpec(row.id, row.name, row.kind, row.env_prefix, row.placement)


@router.get("", response_model=list[EngineResponse])
async def list_engines(session: AsyncSession = Depends(get_session)):
    rows = (await session.scalars(select(Engine).order_by(Engine.id))).all()
    return [EngineResponse.of(row, _address(row)) for row in rows]


class EngineCreateRequest(BaseModel):
    name: str = Field(max_length=64, pattern=r"^[a-z][a-z0-9_-]{0,63}$")
    kind: EngineKind
    env_prefix: str = Field(pattern=PREFIX)
    placement: Placement


@router.post("", response_model=EngineResponse)
async def create_engine(
    request: EngineCreateRequest, session: AsyncSession = Depends(get_session)
):
    if await session.scalar(select(exists().where(Engine.name == request.name))):
        raise HTTPException(status_code=409, detail=f"engine {request.name} already exists")
    row = Engine(**request.model_dump())
    # asked before the insert: a misspelt prefix used to leave a row behind and answer 400
    try:
        address = engines.address_of(engines.EngineSpec(0, row.name, row.kind, row.env_prefix,
                                                        row.placement))
    except engines.Unconfigured as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    session.add(row)
    await commit_and_refresh(session, row)
    return EngineResponse.of(row, address)


# `name` and `env_prefix` never change: stamps name the engine by both; a new address is a new engine
class EnginePatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    placement: Placement


@router.patch("/{id}", response_model=EngineResponse)
async def patch_engine(
    id: int, request: EnginePatchRequest, session: AsyncSession = Depends(get_session)
):
    row = await get_or_404(Engine, id, session)
    # the row describes the process it will start: a live one stays where it is, whatever it says
    if request.placement != row.placement and await run_in_threadpool(_running, _spec(row)):
        raise HTTPException(
            status_code=409,
            detail=f"{row.name} is running as {row.placement.value}: stop it, change the"
            " placement, and start it where the new one says",
        )
    row.placement = request.placement
    await commit_and_refresh(session, row)
    engines.forget_clients(row.id)
    return EngineResponse.of(row, _address(row))


@router.get("/{id}/live", response_model=EngineResponse)
async def probe_engine(id: int, session: AsyncSession = Depends(get_session)):
    row = await get_or_404(Engine, id, session)
    # a synchronous call with a 120 second timeout would hold the loop for every other request
    reachable = await run_in_threadpool(_answers, _spec(row))
    return EngineResponse.of(row, _address(row), reachable)


# seconds, not the completion timeout: a refused connection answers at once, and silence is running
def _running(spec) -> bool:
    import requests

    try:
        requests.get(f"{engines.base_url(spec)}/v1/models", timeout=3)
        return True
    except engines.Unconfigured:
        return False
    except requests.ConnectionError:
        return False
    except Exception:
        return True


def _answers(spec) -> bool:
    try:
        engines.client_for(spec).models.list()
        return True
    except Exception:
        return False


@router.delete("/{id}", response_model=EngineResponse)
async def delete_engine(id: int, session: AsyncSession = Depends(get_session)):
    row = await get_or_404(Engine, id, session)
    if await session.scalar(select(exists().where(Model.engine_id == row.id))):
        raise HTTPException(status_code=409, detail="models still point at this engine")
    answer = EngineResponse.of(row, _address(row))
    await session.delete(row)
    await session.commit()
    engines.forget_clients(answer.id)
    return answer
