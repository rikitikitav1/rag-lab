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


@register("delete_llm_model")
def delete_llm_model(options: dict) -> None:
    llm.delete_model(options["name"])
