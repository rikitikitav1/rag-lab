from datetime import datetime

from crud import ensure_not_active, get_or_404
from fastapi import APIRouter, Depends, HTTPException, Query
from models.registry import Prompt, Purpose
from orm.async_db import commit_and_refresh, get_session
from pydantic import BaseModel
from query_utils import Page, apply_sort_limit_offset
from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

router = APIRouter(prefix="/prompt", tags=["prompts"])


class PromptResponse(BaseModel):
    id: int
    purpose: Purpose
    version: int
    template: str
    active: bool
    created_at: datetime

    model_config = {"from_attributes": True}


@router.get("", response_model=list[PromptResponse])
async def list_prompts(
    id: list[int] | None = Query(default=None),
    purpose: list[Purpose] | None = Query(default=None),
    active: list[bool] | None = Query(default=None),
    template: list[str] | None = Query(default=None),
    created_from: datetime | None = Query(default=None),
    created_to: datetime | None = Query(default=None),
    page: Page = Depends(),
    session: AsyncSession = Depends(get_session),
):

    if (
        created_from is not None
        and created_to is not None
        and created_from > created_to
    ):
        raise HTTPException(
            status_code=400,
            detail="created_from must be earlier than created_to",
        )

    stmt = select(Prompt)

    filters = {
        Prompt.id: id,
        Prompt.purpose: purpose,
        Prompt.template: template,
        Prompt.active: active,
    }

    for column, value in filters.items():
        if value is not None:
            stmt = stmt.where(column.in_(value))

    if created_from is not None:
        stmt = stmt.where(Prompt.created_at >= created_from)

    if created_to is not None:
        stmt = stmt.where(Prompt.created_at <= created_to)

    stmt = apply_sort_limit_offset(
        stmt=stmt,
        sort_map={
            "id": Prompt.id,
            "created_at": Prompt.created_at,
            "purpose": Prompt.purpose,
            "version": Prompt.version,
        },
        sort_by=page.sort_by,
        sort_order=page.sort_order,
        limit=page.limit,
        offset=page.offset,
    )

    result = await session.scalars(stmt)
    return result.all()


@router.get("/{id}", response_model=PromptResponse)
async def show_prompt(
    id: int,
    session: AsyncSession = Depends(get_session),
):
    return await get_or_404(Prompt, id, session)


@router.post("/{id}/activate", response_model=PromptResponse)
async def activate_prompt(
    id: int,
    session: AsyncSession = Depends(get_session),
):
    prompt = await get_or_404(Prompt, id, session)
    await session.execute(
        update(Prompt)
        .where(Prompt.purpose == prompt.purpose, Prompt.active)
        .values(active=False)
    )
    prompt.active = True
    return await commit_and_refresh(session, prompt)


@router.delete("/{id}", response_model=PromptResponse)
async def delete_prompt(
    id: int,
    session: AsyncSession = Depends(get_session),
):
    prompt = await get_or_404(Prompt, id, session)
    ensure_not_active(prompt)
    await session.delete(prompt)
    await session.commit()

    return prompt


class PromptCreateRequest(BaseModel):
    purpose: Purpose
    template: str


@router.post("", response_model=PromptResponse)
async def create_prompt(
    request: PromptCreateRequest,
    session: AsyncSession = Depends(get_session),
):
    stmt = select(func.max(Prompt.version)).where(Prompt.purpose == request.purpose)

    max_version = await session.scalar(stmt)

    version = (max_version or 0) + 1

    prompt = Prompt(
        purpose=request.purpose,
        version=version,
        template=request.template,
    )

    session.add(prompt)

    return await commit_and_refresh(session, prompt)
