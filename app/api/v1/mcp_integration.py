from datetime import datetime
from typing import Annotated, Literal

import job_queue
from crud import get_or_404
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.concurrency import run_in_threadpool
from models.mcp_integration import TOOL_NAME_RE, McpIntegration, McpStatus, can_switch
from orm.async_db import commit_and_refresh, get_session
from pydantic import BaseModel, Field
from query_utils import (
    Page,
    apply_in_filters,
    apply_sort_limit_offset,
    refuse_backwards_range,
)
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from use_cases import mcp_integration as mcp_uc

router = APIRouter(prefix="/mcp_integration", tags=["mcp_integrations"])

_NAME_PATTERN = r"^[a-z][a-z0-9]*(_[a-z0-9]+)*$"
_ENV_PATTERN = r"^[A-Z][A-Z0-9_]*$"
_HEADER_PATTERN = r"^[A-Za-z0-9-]+$"
_URL_PATTERN = r"^https?://\S+$"


class BearerAuth(BaseModel):
    type: Literal["bearer"]
    token_env: str = Field(max_length=64, pattern=_ENV_PATTERN)


class HeaderAuth(BaseModel):
    type: Literal["header"]
    header: str = Field(max_length=64, pattern=_HEADER_PATTERN)
    value_env: str = Field(max_length=64, pattern=_ENV_PATTERN)


Auth = Annotated[BearerAuth | HeaderAuth, Field(discriminator="type")]
ToolList = list[Annotated[str, Field(max_length=128, pattern=TOOL_NAME_RE.pattern)]]


class McpIntegrationResponse(BaseModel):
    id: int
    name: str
    url: str
    status: McpStatus
    allowed_tools: list[str]
    tool_schemas: dict
    auth: dict | None
    timeout_s: int
    max_result_chars: int
    last_checked_at: datetime | None
    last_error: str | None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


@router.get("", response_model=list[McpIntegrationResponse])
async def list_integrations(
    id: list[int] | None = Query(default=None),
    name: list[str] | None = Query(default=None),
    status: list[McpStatus] | None = Query(default=None),
    url_like: str | None = Query(default=None, max_length=256),
    tool: str | None = Query(default=None, max_length=128, pattern=TOOL_NAME_RE.pattern),
    has_error: bool | None = Query(default=None),
    checked_before: datetime | None = Query(default=None),
    checked_after: datetime | None = Query(default=None),
    created_before: datetime | None = Query(default=None),
    created_after: datetime | None = Query(default=None),
    page: Page = Depends(),
    session: AsyncSession = Depends(get_session),
):
    stmt = select(McpIntegration)

    in_filters = {
        McpIntegration.id: id,
        McpIntegration.name: name,
        McpIntegration.status: status,
    }
    stmt = apply_in_filters(stmt, in_filters)
    refuse_backwards_range(checked_after, checked_before, "checked_after", "checked_before")
    refuse_backwards_range(created_after, created_before, "created_after", "created_before")

    if url_like is not None:
        stmt = stmt.where(McpIntegration.url.ilike(f"%{url_like}%"))
    if tool is not None:
        stmt = stmt.where(McpIntegration.allowed_tools.contains([tool]))
    if has_error is True:
        stmt = stmt.where(McpIntegration.last_error.isnot(None))
    elif has_error is False:
        stmt = stmt.where(McpIntegration.last_error.is_(None))
    if checked_before is not None:
        stmt = stmt.where(McpIntegration.last_checked_at < checked_before)
    if checked_after is not None:
        stmt = stmt.where(McpIntegration.last_checked_at > checked_after)
    if created_before is not None:
        stmt = stmt.where(McpIntegration.created_at < created_before)
    if created_after is not None:
        stmt = stmt.where(McpIntegration.created_at > created_after)

    stmt = apply_sort_limit_offset(
        stmt=stmt,
        sort_map={
            "id": McpIntegration.id,
            "name": McpIntegration.name,
            "status": McpIntegration.status,
            "last_checked_at": McpIntegration.last_checked_at,
            "created_at": McpIntegration.created_at,
        },
        sort_by=page.sort_by,
        sort_order=page.sort_order,
        limit=page.limit,
        offset=page.offset,
    )

    result = await session.scalars(stmt)
    return result.all()


@router.get("/{id}", response_model=McpIntegrationResponse)
async def show_integration(id: int, session: AsyncSession = Depends(get_session)):
    return await get_or_404(McpIntegration, id, session)


class McpIntegrationCreateRequest(BaseModel):
    name: str = Field(max_length=64, pattern=_NAME_PATTERN)
    url: str = Field(max_length=512, pattern=_URL_PATTERN)
    auth: Auth | None = None
    allowed_tools: ToolList = []
    timeout_s: int = Field(default=30, ge=1, le=300)
    max_result_chars: int = Field(default=4000, ge=100, le=100_000)


@router.post("", response_model=McpIntegrationResponse)
async def create_integration(
    request: McpIntegrationCreateRequest,
    session: AsyncSession = Depends(get_session),
):
    existing = await session.scalar(
        select(McpIntegration).where(McpIntegration.name == request.name)
    )
    if existing:
        raise HTTPException(status_code=409, detail=f"name '{request.name}' is taken")

    integration = McpIntegration(
        name=request.name,
        url=request.url,
        auth=request.auth.model_dump() if request.auth else None,
        allowed_tools=request.allowed_tools,
    )
    integration.timeout_s = request.timeout_s
    integration.max_result_chars = request.max_result_chars
    session.add(integration)
    try:
        await session.flush()
    except IntegrityError:
        raise HTTPException(
            status_code=409, detail=f"name '{request.name}' is taken"
        ) from None
    job_queue.add_job(
        session, "check_mcp_health", {"integration_id": integration.id}, queue="io"
    )
    return await commit_and_refresh(session, integration)


class McpIntegrationUpdateRequest(BaseModel):
    url: str = Field(max_length=512, pattern=_URL_PATTERN)
    status: Literal[McpStatus.disabled, McpStatus.active]
    auth: Auth | None = None
    allowed_tools: ToolList = []
    timeout_s: int = Field(default=30, ge=1, le=300)
    max_result_chars: int = Field(default=4000, ge=100, le=100_000)


@router.put("/{id}", response_model=McpIntegrationResponse)
async def update_integration(
    id: int,
    request: McpIntegrationUpdateRequest,
    session: AsyncSession = Depends(get_session),
):
    integration = await get_or_404(McpIntegration, id, session)

    if request.status != integration.status and not can_switch(
        integration.status, request.status
    ):
        raise HTTPException(
            status_code=409,
            detail=f"cannot switch status {integration.status} -> {request.status}",
        )

    for field, value in request.model_dump().items():
        setattr(integration, field, value)
    job_queue.add_job(
        session, "check_mcp_health", {"integration_id": integration.id}, queue="io"
    )
    return await commit_and_refresh(session, integration)


class DiscoveredTool(BaseModel):
    name: str
    description: str


class DiscoverResponse(BaseModel):
    id: int
    name: str
    error: str | None
    tools: list[DiscoveredTool]


@router.post("/{id}/discover", response_model=DiscoverResponse)
async def discover_integration(id: int, session: AsyncSession = Depends(get_session)):
    await get_or_404(McpIntegration, id, session)
    result = await run_in_threadpool(mcp_uc.discover, id)
    if result is None:
        raise HTTPException(status_code=404, detail=f"McpIntegration with id={id} not found")
    return result


class ProbeResponse(BaseModel):
    id: int
    name: str
    alive: bool
    status: McpStatus
    elapsed: float | None
    error: str | None


@router.post("/{id}/probe", response_model=ProbeResponse)
async def probe_integration(id: int, session: AsyncSession = Depends(get_session)):
    await get_or_404(McpIntegration, id, session)
    result = await run_in_threadpool(mcp_uc.check_health, id)
    if result is None:
        raise HTTPException(status_code=404, detail=f"McpIntegration with id={id} not found")
    return result


class HealthResponse(BaseModel):
    id: int
    name: str
    status: McpStatus
    last_checked_at: datetime | None
    last_error: str | None

    model_config = {"from_attributes": True}


@router.get("/{id}/health", response_model=HealthResponse)
async def health_integration(id: int, session: AsyncSession = Depends(get_session)):
    return await get_or_404(McpIntegration, id, session)


@router.delete("/{id}", response_model=McpIntegrationResponse)
async def delete_integration(id: int, session: AsyncSession = Depends(get_session)):
    integration = await get_or_404(McpIntegration, id, session)
    if integration.status != McpStatus.disabled:
        raise HTTPException(
            status_code=409,
            detail=f"integration is {integration.status}; disable it first",
        )
    await session.delete(integration)
    await session.commit()
    return integration
