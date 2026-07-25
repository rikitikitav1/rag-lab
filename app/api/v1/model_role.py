from fastapi import APIRouter, Depends, HTTPException
from models.registry import Model, ModelRole, Role
from orm.async_db import commit_and_refresh, get_session
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

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


@router.put("/{role}", response_model=RoleResponse)
async def assign_role(
    role: Role,
    request: RoleAssignRequest,
    session: AsyncSession = Depends(get_session),
):
    if await session.get(Model, request.model_id) is None:
        raise HTTPException(
            status_code=404,
            detail=f"Model with id={request.model_id} not found",
        )

    assignment = await session.get(ModelRole, role)
    if assignment is None:
        assignment = ModelRole(role=role, model_id=request.model_id)
        session.add(assignment)
    else:
        assignment.model_id = request.model_id

    return await commit_and_refresh(session, assignment)
