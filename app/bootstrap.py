import config
import job_queue
import llm
import logging_setup
from models.registry import Model, ModelRole, Role, Status
from orm.sync_db import Session
from sqlalchemy import exists, select

log = logging_setup.get_logger(__name__)


def bootstrap_models() -> None:
    _ensure_models()
    _ensure_roles()
    _reconcile_with_ollama()
    _ensure_index()
    _ensure_vector_indexes()
    _repair_served_vector_index()
    _ensure_question_embeddings()


def _ensure_models() -> None:
    with Session() as session:
        existing = set(session.scalars(select(Model.name)).all())
        for name in config.settings.llm.pull_models:
            if name not in existing:
                session.add(Model(name=name))
        session.commit()


def _ensure_roles() -> None:
    with Session() as session:
        assigned = set(session.scalars(select(ModelRole.role)).all())
        for role, cfg in config.settings.llm.roles.items():
            if Role(role) in assigned:
                continue
            model = session.scalar(select(Model).where(Model.name == cfg.model))
            if model is not None:
                session.add(ModelRole(role=Role(role), model_id=model.id))
        session.commit()


def _reconcile_with_ollama() -> None:
    try:
        pulled = set(llm.add_tags(llm.list_models()))
    except Exception as e:
        log.error("bootstrap.ollama_unreachable", error=str(e))
        pulled = set()

    to_pull = []
    with Session() as session:
        for model in session.scalars(select(Model)).all():
            if llm.add_tags([model.name])[0] in pulled:
                model.status = Status.ready
            else:
                model.status = Status.loading
                to_pull.append(model.name)
        session.commit()

    for name in to_pull:
        job_queue.enqueue("pull_llm_model", {"name": name}, queue="io")
        log.info("bootstrap.pull_enqueued", name=name)


# an empty named variant is a deliberate next step, not a reason to spend the card on its own
def _ensure_index() -> None:
    import db

    variant = config.settings.corpus.variant
    if db.corpus_variants():
        if db.is_empty(variant=variant):
            log.info("bootstrap.variant_empty", variant=variant, hint="index it deliberately")
        return
    job_queue.enqueue("index_data", {"source": "all", "variant": variant})
    log.info("bootstrap.index_enqueued", variant=variant)


# every variant that has rows gets its index, so no migration ever names one and
# db/schema.sql stays a function of the migrations rather than of what is indexed. Queued,
# never built here: the API and the worker both wait for this service to finish, an hnsw
# build takes tens of minutes, and a migration that drops an index would otherwise hold
# the whole stack down while it comes back
def _ensure_vector_indexes() -> None:
    import use_cases.index

    import db

    for row in db.corpus_variants():
        # the API and the worker both gate on this service finishing, so one variant whose
        # index cannot even be asked about must not keep the stack down
        try:
            present = use_cases.index.has_vector_index(row["variant"])
        except Exception as e:
            log.error("bootstrap.index_check_failed", variant=row["variant"], error=str(e))
            continue
        if present:
            continue
        log.info("bootstrap.vector_index_missing", variant=row["variant"])
        _queue_index_build(row["variant"])



# the served variant without its index answers every question with a sequential scan over
# every vector: right answers at a quietly different scale, and a migration can drop one.
# Refusing to boot would deadlock the repair, because the worker runs that job and the
# worker waits for this service, so the job is queued again on every start until it lands
def _repair_served_vector_index() -> None:
    import use_cases.index

    import db

    served = config.settings.corpus.variant
    try:
        missing = not db.is_empty(variant=served) and not use_cases.index.has_vector_index(
            served
        )
    except Exception as e:
        log.error("bootstrap.served_index_check_failed", variant=served, error=str(e))
        return
    if not missing:
        return
    log.error(
        "bootstrap.served_variant_has_no_index",
        variant=served,
        index=use_cases.index.vector_index_name(served),
    )
    _queue_index_build(served)


# the build takes tens of minutes and bootstrap runs on every start, so a variant that is
# already waiting must not be queued again: the failing path and the served-variant repair
# both point at the same index
def _queue_index_build(variant: str) -> None:
    if job_queue.pending_of_type("build_vector_index", variant=variant):
        log.info("bootstrap.index_build_already_queued", variant=variant)
        return
    job_queue.enqueue("build_vector_index", {"variant": variant})


def _ensure_question_embeddings() -> None:
    from models.eval import Question

    with Session() as session:
        pending = session.scalar(select(exists().where(Question.embedding.is_(None))))
    if not pending:
        return
    if job_queue.pending_of_type("embed_questions"):
        log.info("bootstrap.embed_questions_already_queued")
        return
    job_queue.enqueue("embed_questions", {})
    log.info("bootstrap.embed_questions_enqueued")


if __name__ == "__main__":
    logging_setup.configure("INFO")
    bootstrap_models()
