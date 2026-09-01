from fastapi import APIRouter, Depends, HTTPException
from fastapi.concurrency import run_in_threadpool
from models.registry import Model, ModelRole, Role
from orm.async_db import commit_and_refresh, get_session
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from use_cases import model_acceptance

router = APIRouter(prefix="/role", tags=["roles"])


class RoleResponse(BaseModel):
    role: Role
    model_id: int

    model_config = {"from_attributes": True}


@router.get("", response_model=list[RoleResponse])
async def list_roles(session: AsyncSession = Depends(get_session)):
    result = await session.scalars(select(ModelRole))
    return result.all()


class RoleAssignRequest(BaseModel):
    model_id: int
    # a server can be wrong or new, so the way past is named and a record shows it was taken
    anyway: bool = False


@router.put("/{role}", response_model=RoleResponse)
async def assign_role(
    role: Role,
    request: RoleAssignRequest,
    session: AsyncSession = Depends(get_session),
):
    model = await session.get(Model, request.model_id)
    if model is None:
        raise HTTPException(
            status_code=404,
            detail=f"Model with id={request.model_id} not found",
        )
    # asked here: a model that cannot do its role fails per row while the job reports done
    if not request.anyway:
        try:
            await run_in_threadpool(model_acceptance.refuse_unfit_model, role, model.name)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"{e}. Pass anyway=true to insist") from e

    assignment = await session.get(ModelRole, role)
    if assignment is None:
        assignment = ModelRole(role=role, model_id=request.model_id)
        session.add(assignment)
    else:
        assignment.model_id = request.model_id

    return await commit_and_refresh(session, assignment)
