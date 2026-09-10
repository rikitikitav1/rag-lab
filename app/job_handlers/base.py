from models.registry import Model, ModelRole, Role, Status, refuse_unknown_registry
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


# a model named by a job: registered and pulled if new, and the job waits rather than fails
def require_model_ready(name: str) -> None:
    import engines
    import job_queue

    found = engines.find_model(name)
    if found is None:
        # the same refusal the HTTP door makes: this is a second way to have a name pulled
        refuse_unknown_registry(name)
        spec = engines.seeded_ollama()
        with Session() as session:
            session.add(Model(name=name, engine_id=spec.id))
            session.commit()
        job_queue.enqueue("pull_llm_model", {"name": name, "engine_id": spec.id}, queue="io")
        raise Deferred(30)

    with Session() as session:
        status = session.scalar(
            select(Model.status).where(
                Model.engine_id == found.engine.id, Model.name == found.name
            )
        )
    if status != Status.ready:
        # a model that never arrives would re-defer for the life of the process, holding its lane
        raise Deferred(30)
