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
    _ensure_question_embeddings()
    llm.warn_if_models_do_not_fit()


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


def _ensure_index() -> None:
    import db

    if db.is_empty():
        job_queue.enqueue("index_data", {"source": "all"})
        log.info("bootstrap.index_enqueued")


def _ensure_question_embeddings() -> None:
    from models.eval import Question

    with Session() as session:
        pending = session.scalar(select(exists().where(Question.embedding.is_(None))))
    if pending:
        job_queue.enqueue("embed_questions", {})
        log.info("bootstrap.embed_questions_enqueued")
