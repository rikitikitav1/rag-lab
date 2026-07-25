from models.registry import Model, ModelRole, Role, Status
from orm.sync_db import Session
from sqlalchemy import select

HANDLERS = {}


class Deferred(Exception):
    def __init__(self, delay_seconds: int = 10):
        self.delay_seconds = delay_seconds


def register(job_type):
    def deco(fn):
        HANDLERS[job_type] = fn
        return fn

    return deco


def require_role_ready(role) -> None:
    with Session() as session:
        model = session.scalar(
            select(Model)
            .join(ModelRole, ModelRole.model_id == Model.id)
            .where(ModelRole.role == role)
        )
    if model is None or model.status != Status.ready:
        raise Deferred(10)


def require_embedder_ready() -> None:
    require_role_ready(Role.embedding)
