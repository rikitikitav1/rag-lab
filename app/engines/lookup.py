from dataclasses import dataclass

from models.registry import Engine, EngineKind, Model, ModelRole
from orm.sync_db import Session
from sqlalchemy import select

from .core import Ambiguous, EngineSpec, Unnamed

# the columns an `EngineSpec` takes, in its order: one place, so the two readers cannot drift
COLUMNS = (Engine.id, Engine.name, Engine.kind, Engine.env_prefix, Engine.placement)


@dataclass(frozen=True)
class Resolved:
    name: str
    engine: EngineSpec


def _spec(row) -> EngineSpec:
    return EngineSpec(*row)


def spec_of_role(role: str) -> Resolved:
    with Session() as session:
        row = session.execute(
            select(Model.name, *COLUMNS)
            .join(ModelRole, ModelRole.model_id == Model.id)
            .join(Engine, Engine.id == Model.engine_id)
            .where(ModelRole.role == role)
        ).first()
    if row is None:
        raise Unnamed(f"no model assigned to role {role}")
    return Resolved(row[0], _spec(row[1:]))


# one rule for a bare name: look across every engine unless the caller named one
def find_model(name: str, engine_id: int | None = None) -> Resolved | None:
    stmt = select(*COLUMNS).join(Model, Model.engine_id == Engine.id).where(Model.name == name)
    if engine_id is not None:
        stmt = stmt.where(Engine.id == engine_id)
    with Session() as session:
        rows = session.execute(stmt).all()
    if len(rows) > 1:
        seen = ", ".join(sorted(_spec(r).name for r in rows))
        raise Ambiguous(f"model {name} lives on engines {seen}; name the engine")
    return Resolved(name, _spec(rows[0])) if rows else None


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
    if len(rows) != 1:
        raise Unnamed(f"{len(rows)} ollama engines registered; name the engine")
    return _spec(rows[0])


# None when the table cannot be read at all: an empty set would read as "every engine was deleted"
def registered_names() -> set[str] | None:
    try:
        with Session() as session:
            return set(session.scalars(select(Engine.name)).all())
    except Exception:
        return None
