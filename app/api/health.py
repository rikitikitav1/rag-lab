from engines import ollama
from fastapi import APIRouter, Depends, HTTPException
from fastapi.concurrency import run_in_threadpool
from orm.async_db import get_session
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
from use_cases import stand_health

router = APIRouter(tags=["health"])
# the probes are unversioned by design; the stand read gets its prefix at include time
v1 = APIRouter(prefix="/health", tags=["health"])


@router.get("/liveness")
def liveness():
    return "OK"


@router.get("/readiness")
async def readiness(session: AsyncSession = Depends(get_session)):
    checks = {}

    try:
        await session.execute(text("SELECT 1"))
        checks["postgres"] = "ok"
    except Exception:
        checks["postgres"] = "down"

    try:
        await run_in_threadpool(ollama.list_models)
        checks["ollama"] = "ok"
    except Exception:
        checks["ollama"] = "down"

    if checks["postgres"] != "ok":
        raise HTTPException(status_code=503, detail=checks)

    # the chat may still answer, so not a 503; but a role with no engine is named, not guessed
    try:
        down = await run_in_threadpool(stand_health.roles_down)
    except Exception as e:
        down = [f"cannot read the roles: {str(e)[:80]}"]
    if down:
        checks["status"] = "degraded"
        checks["roles_down"] = down
    return checks


# a route, not the preflight: the tree, the worker's age and its imports are invisible in here
@v1.get("/stand")
async def stand():
    return await run_in_threadpool(stand_health.stand)
