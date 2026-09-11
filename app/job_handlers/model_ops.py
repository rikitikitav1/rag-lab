import engines
import logging_setup
from engines import ollama, vllm
from models.registry import Engine, EngineKind, Model, ModelRole, Status, Weights
from orm.sync_db import Session
from sqlalchemy import select

from .base import register

log = logging_setup.get_logger(__name__)


# a name alone stopped addressing one row when uniqueness moved onto the pair
def _pair(name: str, engine_id: int | None) -> engines.Resolved:
    found = engines.find_model(name, engine_id)
    if found is None:
        raise ValueError(f"no model {name} on engine {engine_id}" if engine_id else
                         f"model {name} is not registered")
    _refuse_remote(found.engine, name)
    return found


# weights on a paid engine are the provider's; local engines each keep their own and are asked
def _refuse_remote(spec: engines.EngineSpec, name: str) -> None:
    if spec.kind is EngineKind.openai_compatible:
        raise engines.NotSupported(f"managing {name} is not applicable to {spec.kind}")


@register("pull_llm_model")
def pull_llm_model(options: dict) -> None:
    found = _pair(options["name"], options.get("engine_id"))
    if found.engine.kind is EngineKind.vllm:
        engines.refuse_if_tight(_size_seen_before(found), found.name, vllm.weights_cache())
        vllm.pull_weights(found.name)
    else:
        engines.refuse_if_tight(_size_seen_before(found), found.name)
        ollama.pull_model(found.name, found.engine)
    record_what_the_server_holds(found)


# a first pull knows no size, but the same name on another engine already recorded one
def _size_seen_before(found) -> int | None:
    with Session() as session:
        mine = session.scalar(
            select(Model.size_bytes).where(
                Model.engine_id == found.engine.id, Model.name == found.name
            )
        )
        if mine is not None:
            return mine
        seen = session.scalars(
            select(Model.size_bytes)
            .where(Model.name == found.name, Model.size_bytes.isnot(None))
        ).first()
    # nobody has pulled this name yet, so the registry is the only one who knows what it costs
    if seen:
        return seen
    if found.engine.kind is EngineKind.vllm:
        return vllm.repo_size(found.name)
    return ollama.registry_size(found.name)


# the size is unknown until the weights are here, and then it is what the next estimate reads
def record_what_the_server_holds(found) -> None:
    try:
        if found.engine.kind is EngineKind.vllm:
            seen = vllm.artifact_of(found.name)
        else:
            seen = ollama.artifact_of(found.name, found.engine)
    except Exception as e:
        log.warning("pull.artifact_unread", model=found.name, error=str(e))
        seen = {}
    with Session() as session:
        model = session.scalars(
            select(Model).where(Model.engine_id == found.engine.id, Model.name == found.name)
        ).first()
        if model is None:
            return
        model.status = Status.ready
        model.quant = model.quant or seen.get("quant")
        model.size_bytes = model.size_bytes or seen.get("size_bytes")
        if model.weights_id is None and seen.get("family"):
            model.weights_id = _weights_row(session, seen)
        session.commit()


# one row per base model, so the closing comparison joins on a key nobody spells by hand
def _weights_row(session, seen: dict) -> int:
    name = f"{seen['family']}-{(seen.get('params') or '?').lower()}"
    row = session.scalars(select(Weights).where(Weights.name == name)).first()
    if row is None:
        row = Weights(name=name, params=seen.get("params"))
        session.add(row)
        session.flush()
    return row.id


# the typed door deletes the row before queueing, so the engine is resolved by id, not by the row
def _engine_for_delete(name: str, engine_id: int | None) -> engines.EngineSpec:
    if engine_id:
        # a named engine that does not resolve is a refusal: falling through deletes another's row
        spec = engines.spec_of_id(engine_id)
        if spec is None:
            raise ValueError(f"engine {engine_id} is not registered")
    else:
        found = engines.find_model(name)
        if found is None:
            raise ValueError(f"model {name} is not registered")
        spec = found.engine
    _refuse_remote(spec, name)
    return spec


# reached past the typed door, the weights vanished from under a role still pointing at them
@register("delete_llm_model")
def delete_llm_model(options: dict) -> None:
    name = options["name"]
    spec = _engine_for_delete(name, options.get("engine_id"))
    with Session() as session:
        model = session.scalars(
            select(Model).where(Model.engine_id == spec.id, Model.name == name)
        ).first()
        # no row means the typed door removed it; a row means we were reached past that door
        if model is not None:
            assigned = session.scalar(select(ModelRole).where(ModelRole.model_id == model.id))
            if assigned is not None:
                raise ValueError(f"{name} is assigned to role {assigned.role}; reassign it first")
            session.delete(model)
            session.commit()
    _refuse_if_another_row_needs_these_weights(name)
    if spec.kind is EngineKind.vllm:
        _refuse_if_served(spec, name)
        vllm.delete_weights(name)
    else:
        ollama.delete_model(name, spec)


# a running vLLM reads its weights from this directory, and deleting it under the server breaks it
def _refuse_if_served(spec: engines.EngineSpec, name: str) -> None:
    try:
        served = vllm.served(spec)
    except Exception:
        return
    if name in served:
        raise ValueError(f"{name} is served by {spec.name} right now; stop that server first")


# two rows can name one artifact, and two engines can share a volume: the row is not the unit
def _refuse_if_another_row_needs_these_weights(name: str) -> None:
    wanted = ollama.add_tags([name])[0]
    with Session() as session:
        held = session.execute(
            select(Model.name, Engine.name, ModelRole.role)
            .join(ModelRole, ModelRole.model_id == Model.id)
            .join(Engine, Engine.id == Model.engine_id)
        ).all()
    for model_name, engine_name, role in held:
        if ollama.add_tags([model_name])[0] == wanted:
            raise ValueError(
                f"{name} is the same artifact as {model_name} on engine {engine_name},"
                f" which role {role} still uses; reassign that role first"
            )
