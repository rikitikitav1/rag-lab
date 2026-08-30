
import job_queue
import llm
from crud import get_or_404
from fastapi import APIRouter, Depends, HTTPException, Query
from models.registry import (
    MAX_MODEL_NAME,
    MODEL_NAME_RE,
    Model,
    ModelRole,
    Status,
    refuse_unknown_registry,
)
from orm.async_db import commit_and_refresh, get_session
from pydantic import BaseModel, Field, field_validator
from query_utils import Page, apply_sort_limit_offset
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
    page: Page = Depends(),
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
        sort_by=page.sort_by,
        sort_order=page.sort_order,
        limit=page.limit,
        offset=page.offset,
    )

    result = await session.scalars(stmt)
    return result.all()


@router.get("/{id}", response_model=ModelResponse)
async def show_model(id: int, session: AsyncSession = Depends(get_session)):
    return await get_or_404(Model, id, session)




class ModelCreateRequest(BaseModel):
    name: str = Field(max_length=MAX_MODEL_NAME, pattern=MODEL_NAME_RE.pattern)

    @field_validator("name")
    @classmethod
    def _check_registry_host(cls, v: str) -> str:
        # the rule itself lives beside the name pattern: a job that registers a model is a
        # second door onto the same pull, and it has to refuse the same names
        refuse_unknown_registry(v)
        return v


@router.post("", response_model=ModelResponse)
async def create_model(
    request: ModelCreateRequest,
    session: AsyncSession = Depends(get_session),
):
    model = Model(name=request.name)
    session.add(model)
    job_queue.add_job(session, "pull_llm_model", {"name": request.name}, queue="io")
    return await commit_and_refresh(session, model)


class LoadedResponse(BaseModel):
    model: str
    context_length: int | None

    model_config = {"protected_namespaces": ()}


@router.post("/{id}/load", response_model=LoadedResponse)
async def load_model(id: int, session: AsyncSession = Depends(get_session)):
    model = await get_or_404(Model, id, session)
    try:
        return llm.load_into_memory(model=model.name)
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

    name = model.name
    await session.delete(model)
    job_queue.add_job(session, "delete_llm_model", {"name": name}, queue="io")
    await session.commit()

    return model
