import job_queue
from crud import get_or_404
from fastapi import APIRouter, Depends, HTTPException, Query
from models.registry import Model, ModelRole, Status
from orm.async_db import commit_and_refresh, get_session
from pydantic import BaseModel, Field
from query_utils import apply_sort_limit_offset
from sqlalchemy import exists, select
from sqlalchemy.ext.asyncio import AsyncSession

router = APIRouter(prefix="/model", tags=["models"])


class ModelResponse(BaseModel):
    id: int
    name: str
    status: Status

    model_config = {"from_attributes": True}


@router.get("", response_model=list[ModelResponse])
async def list_models(
    id: list[int] | None = Query(default=None),
    name: list[str] | None = Query(default=None),
    status: list[Status] | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
    sort_by: str = Query(default="id"),
    sort_order: str = Query(default="asc"),
    session: AsyncSession = Depends(get_session),
):
    stmt = select(Model)

    filters = {Model.id: id, Model.name: name, Model.status: status}
    for column, value in filters.items():
        if value is not None:
            stmt = stmt.where(column.in_(value))

    stmt = apply_sort_limit_offset(
        stmt=stmt,
        sort_map={"id": Model.id, "name": Model.name, "status": Model.status},
        sort_by=sort_by,
        sort_order=sort_order,
        limit=limit,
        offset=offset,
    )

    result = await session.scalars(stmt)
    return result.all()


@router.get("/{id}", response_model=ModelResponse)
async def show_model(id: int, session: AsyncSession = Depends(get_session)):
    return await get_or_404(Model, id, session)


class ModelCreateRequest(BaseModel):
    name: str = Field(pattern=r"^[a-zA-Z0-9][a-zA-Z0-9._-]*(:[a-zA-Z0-9._-]+)?$")


@router.post("", response_model=ModelResponse)
async def create_model(
    request: ModelCreateRequest,
    session: AsyncSession = Depends(get_session),
):
    model = Model(name=request.name)
    session.add(model)
    job_queue.add_job(session, "pull_llm_model", {"name": request.name})
    return await commit_and_refresh(session, model)


@router.delete("/{id}", response_model=ModelResponse)
async def delete_model(id: int, session: AsyncSession = Depends(get_session)):
    model = await get_or_404(Model, id, session)

    assigned = await session.scalar(select(exists().where(ModelRole.model_id == id)))
    if assigned:
        raise HTTPException(
            status_code=409,
            detail="model is assigned to a role; reassign the role first",
        )

    name = model.name
    await session.delete(model)
    job_queue.add_job(session, "delete_llm_model", {"name": name})
    await session.commit()

    return model
