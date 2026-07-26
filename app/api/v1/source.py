from crud import get_or_404
from fastapi import APIRouter, Depends
from models.corpus import DataChunk, DataSource
from orm.async_db import get_session
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

router = APIRouter(prefix="/source", tags=["sources"])


class SourceResponse(BaseModel):
    id: int
    name: str
    kind: str
    active: bool
    chunks: int

    model_config = {"from_attributes": True}


class SourceActiveRequest(BaseModel):
    active: bool


@router.get("", response_model=list[SourceResponse])
async def list_sources(session: AsyncSession = Depends(get_session)):
    counts = dict(
        (
            await session.execute(
                select(DataChunk.source_id, func.count()).group_by(DataChunk.source_id)
            )
        ).all()
    )
    sources = (await session.scalars(select(DataSource).order_by(DataSource.name))).all()
    return [
        SourceResponse(
            id=s.id,
            name=s.name,
            kind=s.kind,
            active=s.active,
            chunks=counts.get(s.id, 0),
        )
        for s in sources
    ]


@router.put("/{id}", response_model=SourceResponse)
async def set_source_active(
    id: int,
    request: SourceActiveRequest,
    session: AsyncSession = Depends(get_session),
):
    source = await get_or_404(DataSource, id, session)
    source.active = request.active
    await session.commit()
    count = await session.scalar(
        select(func.count()).select_from(DataChunk).where(DataChunk.source_id == id)
    )
    return SourceResponse(
        id=source.id,
        name=source.name,
        kind=source.kind,
        active=source.active,
        chunks=count or 0,
    )
