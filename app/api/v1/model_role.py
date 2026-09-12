import job_queue
from fastapi import APIRouter, Depends, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
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


class SeatQueuedResponse(BaseModel):
    job_id: int
    detail: str


@router.put("/{role}", response_model=RoleResponse, responses={
    202: {"model": SeatQueuedResponse,
          "description": "an asleep vLLM: a `hand_card` job wakes it, probes and then seats the role"},
    503: {"description": "the model's engine does not answer"},
})
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
            await run_in_threadpool(
                model_acceptance.refuse_unfit_model, role, model.name, model.engine_id
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"{e}. Pass anyway=true to insist") from e
        except model_acceptance.EngineDown as e:
            raise HTTPException(status_code=503, detail=str(e)) from e
        except model_acceptance.NeedsProbe as e:
            # the stand can wake it and ask, as `/load` does; the role is seated once the probe says so
            held = await session.get(ModelRole, role)
            asked = {"engine_id": model.engine_id, "model": model.name, "seat": role.value}
            # a seat already waiting answers a second ask, as a load does
            job_id = await run_in_threadpool(
                job_queue.pending_handover, model.engine_id, model.name, role.value
            ) or await run_in_threadpool(
                job_queue.enqueue, "hand_card",
                {**asked, "seat_over": held.model_id if held else None},
            )
            queued = SeatQueuedResponse(job_id=job_id, detail=str(e))
            return JSONResponse(status_code=202, content=queued.model_dump())

    assignment = await session.get(ModelRole, role)
    if assignment is None:
        assignment = ModelRole(role=role, model_id=request.model_id)
        session.add(assignment)
    else:
        assignment.model_id = request.model_id

    return await commit_and_refresh(session, assignment)
