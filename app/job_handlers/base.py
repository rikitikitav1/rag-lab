from errors import Final as Final
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


# taken in the job's own turn: a handover queued apart ping-ponged the card with whoever came between
def require_card(role: str, model: str | None = None, allow_spill: bool = False) -> None:
    import engines
    import llm

    from .card import take

    picked = llm.resolve_for(role, model)
    if picked.engine.placement in engines.CARD:
        # a run's `allow_cpu`: a fresh load half on the processor then measures the cpu on purpose
        take(picked.engine, picked.name, allow_spill=allow_spill)


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
        require_card(role.value)


def require_embedder_ready() -> None:
    require_role_ready(Role.embedding)


# a model named by a job: registered and pulled if new, and the job waits rather than fails
def require_model_ready(name: str, role: str | None = None) -> None:
    import engines
    import job_queue
    import llm

    # the role's engine lacks it: still ambiguous, and never a new name to register and pull
    found = engines.find_model_on(name, lambda: llm.resolve(role).engine.id if role else None)
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
        failed = _failed_pull(found.name, found.engine.id)
        if failed:
            raise Final(f"model {found.name} on {found.engine.name} did not pull: {failed}")
        # a model that never arrives would re-defer for the life of the process, holding its lane
        raise Deferred(30)


# a pull that gave up says so here, or the run waiting on it deferred for an hour
def _failed_pull(name: str, engine_id: int) -> str | None:
    from models.jobs import Job, JobStatus

    with Session() as session:
        last = session.execute(
            select(Job.status, Job.error).where(
                Job.type == "pull_llm_model", Job.options["name"].astext == name,
                Job.options["engine_id"].as_integer() == engine_id,
            ).order_by(Job.id.desc()).limit(1)
        ).first()
    if last is None or last.status != JobStatus.error:
        return None
    return str((last.error or {}).get("error") or "the pull failed")
