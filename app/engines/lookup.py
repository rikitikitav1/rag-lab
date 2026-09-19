from dataclasses import dataclass, field

from models.registry import Engine, EngineKind, Model, ModelRole, Placement
from orm.sync_db import Session
from sqlalchemy import select, update

from . import answer_parsers
from .core import SEEDED_PREFIX, Ambiguous, EngineSpec, Unnamed

# the columns an `EngineSpec` takes, in its order: one place, so the two readers cannot drift
COLUMNS = (Engine.id, Engine.name, Engine.kind, Engine.env_prefix, Engine.placement)


@dataclass(frozen=True)
class Resolved:
    name: str
    engine: EngineSpec
    # what cuts this model's answers on this engine; `none` passes the text as it came
    parser: str = answer_parsers.NONE
    # the model's own sampler, over the role's
    options: dict = field(default_factory=dict, hash=False)


def _spec(row) -> EngineSpec:
    return EngineSpec(*row)


def spec_of_role(role: str) -> Resolved:
    with Session() as session:
        row = session.execute(
            select(Model.name, Model.answer_parser, Model.options, *COLUMNS)
            .join(ModelRole, ModelRole.model_id == Model.id)
            .join(Engine, Engine.id == Model.engine_id)
            .where(ModelRole.role == role)
        ).first()
    if row is None:
        raise Unnamed(f"no model assigned to role {role}")
    return Resolved(row[0], _spec(row[3:]), row[1], row[2] or {})


# one name on two engines, as on `ollama` and `ollama-cpu`: the caller's engine answers, read only then
def find_model_on(name: str, engine_id) -> Resolved | None:
    try:
        return find_model(name)
    except Ambiguous:
        preferred = engine_id()
        found = find_model(name, preferred) if preferred is not None else None
        if found is None:
            raise
        return found


# one rule for a bare name: look across every engine unless the caller named one
def find_model(name: str, engine_id: int | None = None) -> Resolved | None:
    stmt = (
        select(Model.answer_parser, Model.options, *COLUMNS)
        .join(Model, Model.engine_id == Engine.id)
        .where(Model.name == name)
    )
    if engine_id is not None:
        stmt = stmt.where(Engine.id == engine_id)
    with Session() as session:
        rows = session.execute(stmt).all()
    if len(rows) > 1:
        seen = ", ".join(sorted(_spec(r[2:]).name for r in rows))
        raise Ambiguous(f"model {name} lives on engines {seen}; name the engine")
    return Resolved(name, _spec(rows[0][2:]), rows[0][0], rows[0][1] or {}) if rows else None


# a role may name its engine, and a name that matches nothing is a refusal rather than a default
def spec_of_name(name: str) -> EngineSpec | None:
    with Session() as session:
        row = session.execute(select(*COLUMNS).where(Engine.name == name)).first()
    return _spec(row) if row else None


def spec_of_id(engine_id: int) -> EngineSpec | None:
    with Session() as session:
        row = session.execute(select(*COLUMNS).where(Engine.id == engine_id)).first()
    return _spec(row) if row else None


# the config names weights and not a place to put them, and `pull_models` are ollama's by definition
def seeded_ollama() -> EngineSpec:
    with Session() as session:
        rows = session.execute(select(*COLUMNS).where(Engine.kind == EngineKind.ollama)).all()
    specs = [_spec(r) for r in rows]
    if len(specs) == 1:
        return specs[0]
    # a second ollama on the cpu does not unseat the first: the seeded one keeps the old prefix
    seeded = [s for s in specs if s.env_prefix == SEEDED_PREFIX]
    if len(seeded) != 1:
        raise Unnamed(f"{len(specs)} ollama engines registered and none is {SEEDED_PREFIX}")
    return seeded[0]


# None when the table cannot be read at all: an empty set would read as "every engine was deleted"
def registered_names() -> set[str] | None:
    try:
        with Session() as session:
            return set(session.scalars(select(Engine.name)).all())
    except Exception:
        return None


# what can hold the card: the one gpu engine at a time is found among these, never assumed
CARD = (Placement.gpu, Placement.gpu_and_cpu)


def registered() -> list[EngineSpec]:
    with Session() as session:
        return [_spec(r) for r in session.execute(select(*COLUMNS).order_by(Engine.id)).all()]


def card_engines(kind: EngineKind | None = None) -> list[EngineSpec]:
    stmt = select(*COLUMNS).where(Engine.placement.in_(CARD)).order_by(Engine.id)
    if kind is not None:
        stmt = stmt.where(Engine.kind == kind)
    with Session() as session:
        return [_spec(r) for r in session.execute(stmt).all()]


# the one record of a tool probe: the worker, the door and the bootstrap run in three processes
def record_tool_probe(engine_id: int, model: str, started: str, probed: bool) -> None:
    with Session() as session:
        session.execute(
            update(Model).where(Model.engine_id == engine_id, Model.name == model)
            .values(tool_probe=probed, tool_probe_start=started)
        )
        session.commit()


# an answer from an earlier process start says nothing: a restart may have changed the flags
def recorded_tool_probe(engine_id: int, model: str, started: str) -> bool | None:
    with Session() as session:
        seen = session.execute(
            select(Model.tool_probe, Model.tool_probe_start)
            .where(Model.engine_id == engine_id, Model.name == model)
        ).first()
    return seen.tool_probe if seen and seen.tool_probe_start == started else None
