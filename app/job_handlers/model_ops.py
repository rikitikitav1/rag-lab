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
    refuse_remote(found.engine.kind, name)
    return found


# a paid engine keeps the weights itself; the doors and the worker refuse the same way
def refuse_remote(kind: EngineKind, name: str) -> None:
    if kind is EngineKind.openai_compatible:
        raise engines.NotSupported(f"{name} lives on a remote {kind} engine: no weights here to manage")


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
    refuse_remote(spec.kind, name)
    return spec


# reached past the typed door, the weights vanished from under a role still pointing at them
@register("delete_llm_model")
def delete_llm_model(options: dict) -> None:
    name = options["name"]
    spec = _engine_for_delete(name, options.get("engine_id"))
    # every refusal before the row goes: after it, a refusal left weights with no row to name them
    refuse_if_the_weights_are_shared(name, spec.id)
    if spec.kind is EngineKind.vllm:
        vllm.refuse_if_served(name)
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
    if spec.kind is EngineKind.vllm:
        vllm.delete_weights(name)
    else:
        ollama.delete_model(name, spec)


# engines of one kind read one store (ollama a volume, vLLM the HF cache) the stand cannot see
def refuse_if_the_weights_are_shared(name: str, engine_id: int) -> None:
    with Session() as session:
        kind = session.scalar(select(Engine.kind).where(Engine.id == engine_id))
        others = session.execute(
            select(Model.name, Engine.name).join(Engine, Engine.id == Model.engine_id)
            .where(Engine.kind == kind)
            .where(~((Model.engine_id == engine_id) & (Model.name == name)))
        ).all()
    same = (lambda n: ollama.add_tags([n])[0]) if kind is EngineKind.ollama else (lambda n: n)
    for model_name, engine_name in others:
        if same(model_name) == same(name):
            raise ValueError(
                f"{name} shares its weights with {model_name} on {engine_name}; deleting them"
                " would leave that row without its files, so delete that row first"
            )
