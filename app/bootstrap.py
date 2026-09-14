import config
import engines
import job_queue
import llm
import logging_setup
from engines import ollama, vllm
from models.registry import (
    Engine,
    EngineKind,
    Model,
    ModelRole,
    Role,
    Status,
    refuse_unknown_registry,
)
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
    # every ollama with a row: the rows on `ollama-cpu` got neither a status nor a pull while one was read
    for spec in engines.registered():
        if spec.kind is EngineKind.ollama and _holds_models(spec):
            _reconcile_with_ollama(spec)
    _fill_vllm_rows()
    _ensure_index()
    _ensure_vector_indexes()
    _repair_served_vector_index()
    _ensure_question_embeddings()


# vLLM takes the card first when the stack comes up, so it sleeps before ollama loads any role
def _put_vllm_to_sleep() -> None:
    from engines import card

    # `compose --profile ... up` reruns the bootstrap, and a sleep under a running pass cost its row
    if job_queue.running_in_lane("default"):
        log.info("bootstrap.card_left_to_the_running_job")
        return
    card.sleep_every_vllm()


# `pull_models` come through `/api/pull`, so their engine is chosen by kind, not by being the only one
def _seeded() -> engines.EngineSpec | None:
    try:
        return engines.seeded_ollama()
    except engines.Unnamed as e:
        # a second ollama is a thing to fix, not a reason to stop the stack from booting
        log.error("bootstrap.no_seeded_engine", error=str(e))
        return None


def _holds_models(spec) -> bool:
    with Session() as session:
        return bool(session.scalar(select(exists().where(Model.engine_id == spec.id))))


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
            # a misspelt role in the file died here as a bare ValueError and took the whole boot down
            if role not in {r.value for r in Role}:
                log.error("bootstrap.role_unknown", role=role, known=sorted(r.value for r in Role))
                continue
            if Role(role) in assigned:
                continue
            spec = _engine_of_role(role, cfg, seeded)
            if spec is None:
                continue
            model = session.scalar(
                select(Model).where(Model.engine_id == spec.id, Model.name == cfg.model)
            )
            fresh = None
            if model is None and spec.kind is EngineKind.vllm:
                model = fresh = _register_what_vllm_serves(session, spec, role, cfg.model)
            # a role on an ollama the pull list does not cover: its row here, pulled by the reconcile
            new_row = model is None and spec.kind is EngineKind.ollama
            if model is None and not new_row:
                log.error("bootstrap.role_model_absent", role=role, model=cfg.model,
                          engine=spec.name)
                continue
            # the gates `POST /v1/model` and `PUT /v1/role` run: an empty database is how roles get set
            try:
                if new_row:
                    refuse_unknown_registry(cfg.model)
                model_acceptance.refuse_unfit_model(Role(role), cfg.model, spec.id)
            except ValueError as e:
                log.error("bootstrap.role_refused", role=role, model=cfg.model, error=str(e))
                # a refused model registered a line above stayed behind, a row with no role
                if fresh is not None:
                    session.delete(fresh)
                continue
            # a boot cannot wait: a stopped engine is named by `/readiness`, a probe asked on its turn
            except (model_acceptance.EngineDown, model_acceptance.NeedsProbe) as e:
                log.warning("bootstrap.role_seated_unasked", role=role, model=cfg.model, why=str(e))
            if new_row:
                model = Model(name=cfg.model, engine_id=spec.id)
                session.add(model)
                session.flush()
            session.add(ModelRole(role=Role(role), model_id=model.id))
        session.commit()


# the model is named twice, in compose and in the config: a server serving another is loud, not a row
def _register_what_vllm_serves(session, spec, role: str, name: str):
    try:
        served = vllm.served(spec)
    except Exception as e:
        # a service under a profile starts after the bootstrap, and its weights answer for it
        try:
            intact = vllm.weights_intact(name)
        except ValueError as refused:
            # a name the cache cannot hold apart stops this role, not the boot
            log.error("bootstrap.vllm_name_refused", model=name, error=str(refused))
            return None
        if not intact:
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


# every vLLM reads this host's HF cache: intact weights are ready, others pulled; a remote one is not
def _fill_vllm_rows() -> None:
    to_pull, to_fill = [], {}
    with Session() as session:
        rows = session.execute(
            select(Model, Engine.id)
            .join(Engine, Engine.id == Model.engine_id)
            .where(Engine.kind == EngineKind.vllm)
        ).all()
        # one hashing pass per repository: 8 GB were read up to three times per boot, per row
        checked = {}
        for model, engine_id in rows:
            if model.name not in checked:
                try:
                    checked[model.name] = vllm.weights_check(model.name)
                except ValueError as e:
                    # a name the cache cannot hold apart stops this row, not the boot
                    log.error("bootstrap.vllm_name_refused", model=model.name, error=str(e))
                    checked[model.name] = None
            broken = checked[model.name]
            if broken is None:
                continue
            if broken:
                log.error("bootstrap.vllm_weights_not_intact", model=model.name, broken=broken[:5])
                model.status = Status.loading
                to_pull.append((model.name, engine_id))
                continue
            model.status = Status.ready
            # the pull's own record fills each field on its own, and the weights row with them
            if None in (model.weights_id, model.quant, model.size_bytes):
                to_fill.setdefault(engine_id, []).append(model.name)
        session.commit()
    for engine_id, names in to_fill.items():
        _fill_from_the_server(engines.spec_of_id(engine_id), names)
    for name, engine_id in to_pull:
        if not job_queue.pending_of_type("pull_llm_model", name=name, engine_id=engine_id):
            job_queue.enqueue("pull_llm_model", {"name": name, "engine_id": engine_id})


def _reconcile_with_ollama(spec) -> None:
    try:
        pulled = set(ollama.add_tags(ollama.list_models(spec)))
    except Exception as e:
        # silence says nothing about its disk and a pull on it only fails: without a card, every boot
        log.warning("bootstrap.ollama_silent_rows_kept", engine=spec.name, error=str(e))
        return

    to_pull, to_fill = [], []
    with Session() as session:
        # only ollama's own rows: another engine's model is absent from `/api/tags` by design
        rows = session.execute(
            select(Model, Engine.id)
            .join(Engine, Engine.id == Model.engine_id)
            .where(Engine.id == spec.id)
        ).all()
        ready = []
        for model, engine_id in rows:
            if ollama.add_tags([model.name])[0] in pulled:
                model.status = Status.ready
                ready.append(model.name)
                if None in (model.weights_id, model.quant, model.size_bytes):
                    to_fill.append(model.name)
            else:
                model.status = Status.loading
                to_pull.append((model.name, engine_id))
        session.commit()

    # recreating a model unloads it, and a job running on it would lose its model mid-pass
    held_by_a_job = job_queue.running_in_lane("default")
    if held_by_a_job:
        log.warning("bootstrap.repetition_penalty_left_to_next_boot", models=ready)
    # bases before the windowed tags made from them; a model pulled by hand never got the penalty
    for name in [] if held_by_a_job else sorted(ready, key=lambda n: ollama.windowed(n) is not None):
        try:
            ollama.hold_repetition_penalty(name, spec)
        except Exception as e:
            log.warning("bootstrap.repetition_penalty_not_held", model=name, error=str(e))

    for name, engine_id in to_pull:
        # a pull still waiting from the last boot answers this one, as it does for vLLM
        if job_queue.pending_of_type("pull_llm_model", name=name, engine_id=engine_id):
            continue
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
    from use_cases.index import question_needs_embedding

    label = None
    try:
        # a role moved to another embedder leaves every question vector foreign to the new one
        label = llm.embedder_label()
    except Exception as e:
        log.error("bootstrap.embedder_unknown", error=str(e))
    missing = question_needs_embedding(label)
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
