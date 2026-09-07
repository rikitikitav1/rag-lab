import llm
from models.registry import Model, Status
from orm.sync_db import Session
from sqlalchemy import select

from .base import register


@register("pull_llm_model")
def pull_llm_model(options: dict) -> None:
    name = options["name"]
    llm.pull_model(name)
    with Session() as session:
        model = session.scalars(select(Model).where(Model.name == name)).first()
        if model:
            model.status = Status.ready
            session.commit()


# reached past the typed door, the weights vanished from under a role still pointing at them
@register("delete_llm_model")
def delete_llm_model(options: dict) -> None:
    from models.registry import ModelRole

    name = options["name"]
    with Session() as session:
        model = session.scalars(select(Model).where(Model.name == name)).first()
        if model is not None:
            assigned = session.scalar(select(ModelRole).where(ModelRole.model_id == model.id))
            if assigned is not None:
                raise ValueError(f"{name} is assigned to role {assigned.role}; reassign it first")
            session.delete(model)
            session.commit()
    llm.delete_model(name)
