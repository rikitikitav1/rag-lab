import os
from contextlib import asynccontextmanager

import bootstrap
import logging_setup
from api import health
from api.v1 import (
    agent,
    categories,
    chat,
    eval,
    job,
    llm_model,
    model_role,
    prompt,
    question_log,
    questions,
    source,
)
from fastapi import FastAPI

logging_setup.configure(os.getenv("LOG_LEVEL", "INFO"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    bootstrap.bootstrap_models()
    yield


app = FastAPI(lifespan=lifespan)

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
