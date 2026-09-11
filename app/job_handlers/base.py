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


# a job on a card engine waits for the card, and asks for it once rather than once per retry
def require_card(role: str, model: str | None = None, asked_by: str | None = None) -> None:
    import engines
    import job_queue
    import llm
    from engines import card

    picked = llm.resolve_for(role, model)
    spec = picked.engine
    if spec.placement not in engines.CARD or card.holds_for(spec):
        return
    if not job_queue.pending_of_type("hand_card", engine_id=spec.id):
        job_queue.enqueue(
            "hand_card", {"engine_id": spec.id, "model": picked.name, "asked_by": asked_by}
        )
    # short: `reschedule` moves the waiter on by this much after the card has already changed hands
    raise Deferred(5)


def require_role_ready(role, take_card: bool = True) -> None:
    with Session() as session:
        model = session.scalar(
            select(Model)
            .join(ModelRole, ModelRole.model_id == Model.id)
            .where(ModelRole.role == role)
        )
    if model is None or model.status != Status.ready:
        raise Deferred(10)
    # every role that answers on the card asks for it here, so no handler can forget to
    if take_card:
        require_card(role.value, asked_by="judge_answers" if role is Role.judging else None)


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
