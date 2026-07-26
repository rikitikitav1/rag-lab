import code

import bootstrap
import config
import crud
import job_queue
import llm
import prompt_repo
import seed
import sources.factory
import use_cases.agent
import use_cases.chat as chat
import use_cases.index
import use_cases.judge
import worker
from orm.base import Base
from orm.sync_db import Session
from sqlalchemy import delete, exists, func, insert, select, text, update

import db


def start() -> None:
    session = Session()
    entities = {mapper.class_.__name__: mapper.class_ for mapper in Base.registry.mappers}
    namespace = {
        "config": config,
        "Session": Session,
        "session": session,
        "select": select,
        "insert": insert,
        "update": update,
        "delete": delete,
        "exists": exists,
        "func": func,
        "text": text,
        "job_queue": job_queue,
        "worker": worker,
        "bootstrap": bootstrap,
        "seed": seed,
        "crud": crud,
        "prompt_repo": prompt_repo,
        "chat": chat,
        "db": db,
        "llm": llm,
        "index": use_cases.index,
        "judge": use_cases.judge,
        "agent": use_cases.agent,
        "sources": sources.factory,
        **entities,
    }
    code.interact(local=namespace)
