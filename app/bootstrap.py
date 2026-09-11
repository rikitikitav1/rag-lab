import config
import engines
import job_queue
import llm
import logging_setup
from engines import ollama, vllm
from models.registry import Engine, EngineKind, Model, ModelRole, Role, Status
from orm.sync_db import Session
from sqlalchemy import exists, select
from use_cases import model_acceptance

log = logging_setup.get_logger(__name__)


def bootstrap_models() -> None:
    _put_vllm_to_sleep()
    seeded = _seeded()
    if seeded is not None:
        _ensure_models(seeded)
    # a role naming its own engine does not need the seeded one, and that is the point of naming it
    _ensure_roles(seeded)
    # every ollama: the rows on `ollama-cpu` got neither a status nor a pull while only one was read
    for spec in engines.registered():
        if spec.kind is EngineKind.ollama:
            _reconcile_with_ollama(spec, pull_when_silent=spec == seeded)
    _fill_vllm_rows()
    _ensure_index()
    _ensure_vector_indexes()
    _repair_served_vector_index()
    _ensure_question_embeddings()


# vLLM takes the card first when the stack comes up, so it sleeps before ollama loads any role
def _put_vllm_to_sleep() -> None:
    for spec in engines.card_engines(EngineKind.vllm):
        awake = vllm.is_sleeping(spec) is False
        if not awake:
            # an engine that does not answer holds no card, and asleep is where it should be
            log.info("bootstrap.vllm_not_awake", engine=spec.name)
            continue
        # awake and refusing would leave every ollama role half on the cpu, and nothing would say so
        vllm.sleep(spec)
        log.info("bootstrap.vllm_asleep", engine=spec.name)


# `pull_models` come through `/api/pull`, so their engine is chosen by kind, not by being the only one
def _seeded() -> engines.EngineSpec | None:
    try:
        return engines.seeded_ollama()
    except engines.Unnamed as e:
        # a second ollama is a thing to fix, not a reason to stop the stack from booting
        log.error("bootstrap.no_seeded_engine", error=str(e))
        return None


def _ensure_models(spec) -> None:
    with Session() as session:
        existing = set(
            session.scalars(select(Model.name).where(Model.engine_id == spec.id)).all()
        )
        for name in config.settings.llm.pull_models:
            if name not in existing:
                session.add(Model(name=name, engine_id=spec.id))
        session.commit()


# a role that names its engine is seated there or nowhere; silence still means the seeded ollama
def _engine_of_role(role: str, cfg, seeded):
    if cfg.engine is None:
        if seeded is None:
            log.error("bootstrap.role_has_no_engine", role=role, model=cfg.model)
        return seeded
    spec = engines.spec_of_name(cfg.engine)
    if spec is None:
        log.error("bootstrap.role_engine_unknown", role=role, engine=cfg.engine)
    return spec


def _ensure_roles(seeded) -> None:
    with Session() as session:
        assigned = set(session.scalars(select(ModelRole.role)).all())
        for role, cfg in config.settings.llm.roles.items():
            if Role(role) in assigned:
                continue
            spec = _engine_of_role(role, cfg, seeded)
            if spec is None:
                continue
            model = session.scalar(
                select(Model).where(Model.engine_id == spec.id, Model.name == cfg.model)
            )
            if model is None and spec.kind is EngineKind.vllm:
                model = _register_what_vllm_serves(session, spec, role, cfg.model)
            if model is None:
                log.error("bootstrap.role_model_absent", role=role, model=cfg.model,
                          engine=spec.name)
                continue
            # the same gate `PUT /v1/role` runs: an empty database is the usual way roles are set
            try:
                model_acceptance.refuse_unfit_model(Role(role), cfg.model)
            except ValueError as e:
                log.error("bootstrap.role_refused", role=role, model=cfg.model, error=str(e))
                continue
            session.add(ModelRole(role=Role(role), model_id=model.id))
        session.commit()


# the model is named twice, in compose and in the config: a server serving another is loud, not a row
def _register_what_vllm_serves(session, spec, role: str, name: str):
    try:
        served = vllm.served(spec)
    except Exception as e:
        # a service under a profile starts after the bootstrap, and its weights answer for it
        if not vllm.weights_intact(name):
            log.error("bootstrap.vllm_unreachable", engine=spec.name, error=str(e))
            return None
        log.info("bootstrap.vllm_registered_from_disk", engine=spec.name, model=name)
        served = [name]
    if name not in served:
        log.error("bootstrap.vllm_serves_another", role=role, engine=spec.name, asked=name,
                  served=served)
        return None
    model = Model(name=name, engine_id=spec.id, status=Status.ready)
    session.add(model)
    session.flush()
    return model


# like ollama's reconcile: weights on disk and intact are ready, absent or broken ones are pulled
def _fill_vllm_rows() -> None:
    to_pull = []
    with Session() as session:
        rows = session.execute(
            select(Model, Engine.id)
            .join(Engine, Engine.id == Model.engine_id)
            .where(Engine.kind == EngineKind.vllm)
        ).all()
        for model, engine_id in rows:
            if not vllm.weights_intact(model.name):
                log.error("bootstrap.vllm_weights_not_intact", model=model.name,
                          broken=vllm.broken_weights(model.name)[:5])
                model.status = Status.loading
                to_pull.append((model.name, engine_id))
                continue
            model.status = Status.ready
            if model.quant is None:
                seen = vllm.artifact_of(model.name)
                model.quant = seen.get("quant")
                model.size_bytes = model.size_bytes or seen.get("size_bytes")
        session.commit()
    for name, engine_id in to_pull:
        if not job_queue.pending_of_type("pull_llm_model", name=name, engine_id=engine_id):
            job_queue.enqueue("pull_llm_model", {"name": name, "engine_id": engine_id})


def _reconcile_with_ollama(spec, pull_when_silent: bool = True) -> None:
    try:
        pulled = set(ollama.add_tags(ollama.list_models(spec)))
    except Exception as e:
        log.error("bootstrap.ollama_unreachable", engine=spec.name, error=str(e))
        # a second ollama that is down says nothing about its disk, and a pull on it only fails
        if not pull_when_silent:
            return
        pulled = set()

    to_pull, to_fill = [], []
    with Session() as session:
        # only ollama's own rows: another engine's model is absent from `/api/tags` by design
        rows = session.execute(
            select(Model, Engine.id)
            .join(Engine, Engine.id == Model.engine_id)
            .where(Engine.id == spec.id)
        ).all()
        for model, engine_id in rows:
            if ollama.add_tags([model.name])[0] in pulled:
                model.status = Status.ready
                if None in (model.weights_id, model.quant, model.size_bytes):
                    to_fill.append(model.name)
            else:
                model.status = Status.loading
                to_pull.append((model.name, engine_id))
        session.commit()

    for name, engine_id in to_pull:
        job_queue.enqueue("pull_llm_model", {"name": name, "engine_id": engine_id}, queue="io")
        log.info("bootstrap.pull_enqueued", name=name, engine_id=engine_id)
    _fill_from_the_server(spec, to_fill)


# rows registered before `/api/show` was read carry no quant and no weights, and nobody types those
def _fill_from_the_server(spec, names: list[str]) -> None:
    from job_handlers.model_ops import record_what_the_server_holds

    for name in names:
        record_what_the_server_holds(engines.Resolved(name, spec))


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


# queued, never built here: an hnsw build takes tens of minutes and the stack waits for boot
def _ensure_vector_indexes() -> None:
    import use_cases.index

    import db

    for row in db.corpus_variants():
        # one variant whose index cannot even be asked about must not keep the stack down
        try:
            present = use_cases.index.has_vector_index(row["variant"])
        except Exception as e:
            log.error("bootstrap.index_check_failed", variant=row["variant"], error=str(e))
            continue
        if present:
            continue
        log.info("bootstrap.vector_index_missing", variant=row["variant"])
        _queue_index_build(row["variant"])



# without its index the served variant scans: right answers at a quietly different scale
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


# bootstrap runs on every start, so a variant already waiting must not be queued twice
def _queue_index_build(variant: str) -> None:
    if job_queue.pending_of_type("build_vector_index", variant=variant):
        log.info("bootstrap.index_build_already_queued", variant=variant)
        return
    job_queue.enqueue("build_vector_index", {"variant": variant})


def _ensure_question_embeddings() -> None:
    from models.eval import Question

    missing = Question.embedding.is_(None)
    try:
        # a role moved to another embedder leaves every question vector foreign to the new one
        missing = missing | Question.embedded_by.is_distinct_from(llm.embedder_label())
    except Exception as e:
        log.error("bootstrap.embedder_unknown", error=str(e))
    with Session() as session:
        pending = session.scalar(select(exists().where(missing)))
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
