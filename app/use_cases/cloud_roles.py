import engines
from errors import Refusal
from models.registry import Engine, Model, ModelRole, Placement, Role, Status
from orm.sync_db import Session
from sqlalchemy import select
from use_cases import model_acceptance


# a role on a cloud broker moves to another cloud model; a local one holds the card and is a person's to move
def seat(role: str, model: str, engine: str, anyway: bool = False) -> dict:
    try:
        wanted = Role(role)
    except ValueError as e:
        raise Refusal("invalid", f"no role {role}; roles: {sorted(r.value for r in Role)}") from e
    with Session() as session:
        held = session.get(ModelRole, wanted)
        now = session.get(Model, held.model_id) if held else None
        now_on = session.get(Engine, now.engine_id) if now else None
        if now_on is None or now_on.placement is not Placement.remote:
            where = now_on.name if now_on else "no engine"
            raise Refusal("invalid", f"{role} runs on {where}, not a cloud broker; local roles are changed by a person")
        target = session.scalar(select(Engine).where(Engine.name == engine))
        if target is None or target.placement is not Placement.remote:
            clouds = sorted(session.scalars(select(Engine.name).where(Engine.placement == Placement.remote)))
            raise Refusal("invalid", f"{engine} is not a cloud broker; cloud brokers: {clouds}")
        served = engines.served_models(engines.EngineSpec(
            target.id, target.name, target.kind, target.env_prefix, target.placement))
        if served is None:
            raise Refusal("busy", f"{engine} does not answer")
        if model not in served:
            raise Refusal("missing", f"{engine} does not serve {model}; it serves {sorted(served)[:20]}")
        row = session.scalar(select(Model).where(Model.name == model, Model.engine_id == target.id))
        if row is None:
            row = Model(name=model, engine_id=target.id, status=Status.ready)
            session.add(row)
            session.flush()
        if not anyway:
            try:
                model_acceptance.refuse_unfit_model(wanted, model, target.id)
            except ValueError as e:
                raise Refusal("invalid", f"{e}; anyway=true insists") from e
        was = f"{now.name} on {now_on.name}"
        held.model_id = row.id
        session.commit()
    return {"role": role, "model": model, "engine": engine, "was": was}
