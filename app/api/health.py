import llm
from fastapi import APIRouter, Depends, HTTPException
from fastapi.concurrency import run_in_threadpool
from orm.async_db import get_session
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

router = APIRouter(tags=["health"])


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
        await run_in_threadpool(llm.list_models)
        checks["ollama"] = "ok"
    except Exception:
        checks["ollama"] = "down"

    if checks["postgres"] != "ok":
        raise HTTPException(status_code=503, detail=checks)

    return checks
