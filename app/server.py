import os
from contextlib import AsyncExitStack, asynccontextmanager

import bootstrap
import logging_setup
from api import health
from api.v1 import (
    agent,
    categories,
    chat,
    eval,
    experiment,
    job,
    llm_model,
    model_role,
    prompt,
    question_log,
    questions,
    source,
)
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from mcp_server import mcp

logging_setup.configure(os.getenv("LOG_LEVEL", "INFO"))

MAX_BODY_BYTES = 6 * 1024 * 1024

mcp_app = mcp.http_app(path="/", stateless_http=True)


@asynccontextmanager
async def lifespan(app):
    async with AsyncExitStack() as stack:
        bootstrap.bootstrap_models()
        await stack.enter_async_context(mcp_app.lifespan(app))
        yield


app = FastAPI(lifespan=lifespan)

app.mount("/mcp", mcp_app)


@app.middleware("http")
async def _limit_body_size(request, call_next):
    is_mcp = request.url.path.startswith("/mcp")
    if not is_mcp and "transfer-encoding" in request.headers:
        return JSONResponse({"detail": "content-length required"}, status_code=411)
    length = request.headers.get("content-length")
    if length is not None:
        try:
            size = int(length)
        except ValueError:
            return JSONResponse({"detail": "invalid content-length"}, status_code=400)
        if size < 0:
            return JSONResponse({"detail": "invalid content-length"}, status_code=400)
        if size > MAX_BODY_BYTES:
            return JSONResponse({"detail": "request body too large"}, status_code=413)
    return await call_next(request)


app.include_router(health.router)
app.include_router(chat.router, prefix="/v1")
app.include_router(agent.router, prefix="/v1")
app.include_router(categories.router, prefix="/v1")
app.include_router(llm_model.router, prefix="/v1")
app.include_router(model_role.router, prefix="/v1")
app.include_router(prompt.router, prefix="/v1")
app.include_router(question_log.router, prefix="/v1")
app.include_router(eval.router, prefix="/v1")
app.include_router(source.router, prefix="/v1")
app.include_router(questions.router, prefix="/v1")
app.include_router(job.router, prefix="/v1")
app.include_router(experiment.router, prefix="/v1")
