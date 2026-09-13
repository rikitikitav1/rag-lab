import engines
from crud import get_or_404
from engines import balances
from fastapi import APIRouter, Depends, HTTPException
from fastapi.concurrency import run_in_threadpool
from models.registry import Engine, EngineKind, Model, Placement
from orm.async_db import commit_and_refresh, get_session
from pydantic import BaseModel, ConfigDict, Field, field_validator
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
    balance_reader: str
    # read from the environment, never from the row, and shown so a reader can tell it is set
    address: str | None
    reachable: bool | None

    model_config = {"from_attributes": True}

    @classmethod
    def of(cls, row: Engine, address: str | None, reachable: bool | None = None):
        return cls(
            id=row.id, name=row.name, kind=row.kind, env_prefix=row.env_prefix,
            placement=row.placement, balance_reader=row.balance_reader or "none",
            address=address, reachable=reachable,
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


class BalanceResponse(BaseModel):
    engine: str
    reader: str
    balance: float | None
    unit: str | None
    why: str | None
    read_at: str


# free to ask: the broker's service route generates nothing
@router.get("/balances", response_model=list[BalanceResponse])
async def broker_balances():
    return await run_in_threadpool(balances.summary)


def _known_reader(name: str | None) -> str | None:
    if name is not None:
        balances.refuse_unknown(name)
    return name


class EngineCreateRequest(BaseModel):
    name: str = Field(max_length=64, pattern=r"^[a-z][a-z0-9_-]{0,63}$")
    kind: EngineKind
    env_prefix: str = Field(pattern=PREFIX)
    placement: Placement
    balance_reader: str = "none"

    _reader = field_validator("balance_reader")(_known_reader)


# a reader on a local engine would never be asked, and the summary would not show it
def _refuse_a_reader_off_the_cloud(kind: EngineKind, reader: str | None) -> None:
    if reader not in (None, "none") and kind is not EngineKind.openai_compatible:
        raise HTTPException(status_code=422, detail=f"only a cloud has a broker to ask; {kind.value} has none")


# a remote engine on `gpu` joined the card engines and every handover waited for it forever
def _refuse_a_placement_the_kind_cannot_have(kind: EngineKind, placement: Placement) -> None:
    remote_kind = kind is EngineKind.openai_compatible
    if remote_kind != (placement is Placement.remote):
        raise HTTPException(
            status_code=422,
            detail=f"{kind.value} engines are placed {'remote' if remote_kind else 'locally'}, not {placement.value}",
        )


@router.post("", response_model=EngineResponse)
async def create_engine(
    request: EngineCreateRequest, session: AsyncSession = Depends(get_session)
):
    if await session.scalar(select(exists().where(Engine.name == request.name))):
        raise HTTPException(status_code=409, detail=f"engine {request.name} already exists")
    _refuse_a_placement_the_kind_cannot_have(request.kind, request.placement)
    _refuse_a_reader_off_the_cloud(request.kind, request.balance_reader)
    row = Engine(**request.model_dump())
    # asked before the insert: a misspelt prefix used to leave a row behind and answer 400
    try:
        address = engines.address_of(engines.EngineSpec(0, row.name, row.kind, row.env_prefix,
                                                        row.placement))
    except engines.Unconfigured as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    if row.kind is EngineKind.vllm and row.placement in engines.CARD:
        await run_in_threadpool(_refuse_a_vllm_that_cannot_sleep, row)
    session.add(row)
    await commit_and_refresh(session, row)
    return EngineResponse.of(row, address)


# `name` and `env_prefix` never change: stamps name the engine by both; a new address is a new engine
class EnginePatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    placement: Placement | None = None
    balance_reader: str | None = None

    _reader = field_validator("balance_reader")(_known_reader)


@router.patch("/{id}", response_model=EngineResponse)
async def patch_engine(
    id: int, request: EnginePatchRequest, session: AsyncSession = Depends(get_session)
):
    if request.placement is None and request.balance_reader is None:
        raise HTTPException(status_code=422, detail="nothing to change: name a placement or balance_reader")
    row = await get_or_404(Engine, id, session)
    if request.placement is not None:
        _refuse_a_placement_the_kind_cannot_have(row.kind, request.placement)
        # the row describes the process it will start: a live one stays where it is, whatever it says
        if request.placement != row.placement and await run_in_threadpool(_running, _spec(row)):
            raise HTTPException(
                status_code=409,
                detail=f"{row.name} is running as {row.placement.value}: stop it, change the"
                " placement, and start it where the new one says",
            )
        row.placement = request.placement
    if request.balance_reader is not None:
        _refuse_a_reader_off_the_cloud(row.kind, request.balance_reader)
        row.balance_reader = request.balance_reader
    await commit_and_refresh(session, row)
    engines.forget_clients(row.id)
    return EngineResponse.of(row, _address(row))


@router.get("/{id}/live", response_model=EngineResponse)
async def probe_engine(id: int, session: AsyncSession = Depends(get_session)):
    row = await get_or_404(Engine, id, session)
    # a synchronous call with a 120 second timeout would hold the loop for every other request
    reachable = await run_in_threadpool(_answers, _spec(row))
    return EngineResponse.of(row, _address(row), reachable)


# without VLLM_SERVER_DEV_MODE the sleep routes answer 404, and it would hold the card for good
def _refuse_a_vllm_that_cannot_sleep(row: Engine) -> None:
    from engines import vllm

    # None is a server not up yet, as a service under a profile: the start flags are asked later
    if vllm.has_sleep_routes(_spec(row)) is False:
        raise HTTPException(
            status_code=422,
            detail=f"{row.name} has no sleep routes: start it with VLLM_SERVER_DEV_MODE=1 and"
            " --enable-sleep-mode, or the card can never be handed away from it",
        )


# seconds, not the completion timeout: a refused connection answers at once, and silence is running
def _running(spec) -> bool:
    import requests

    try:
        requests.get(f"{engines.base_url(spec)}/v1/models", timeout=3)
        return True
    except engines.Unconfigured:
        return False
    # first: a ConnectTimeout is a ConnectionError too, and a host that did not answer may be running
    except requests.Timeout:
        return True
    except requests.ConnectionError:
        return False
    except Exception:
        return True


# seconds, with the key: the completion client waited two minutes and retried once
def _answers(spec) -> bool:
    try:
        return engines.served_models(spec) is not None
    except engines.Unconfigured:
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
