import engines
from models.registry import Engine, EngineKind, Model
from orm.sync_db import Session
from sqlalchemy import select


# a paid engine keeps the weights itself; the doors and the worker refuse the same way
def refuse_remote(kind: EngineKind, name: str) -> None:
    if kind is EngineKind.openai_compatible:
        raise engines.NotSupported(f"{name} lives on a remote {kind} engine: no weights here to manage")


# engines of one kind read one store (ollama a volume, vLLM the HF cache) the stand cannot see
def refuse_if_the_weights_are_shared(name: str, engine_id: int) -> None:
    with Session() as session:
        kind = session.scalar(select(Engine.kind).where(Engine.id == engine_id))
        others = session.execute(
            select(Model.name, Engine.name).join(Engine, Engine.id == Model.engine_id)
            .where(Engine.kind == kind)
            .where(~((Model.engine_id == engine_id) & (Model.name == name)))
        ).all()
    same = engines.driver(kind).weights_key
    for model_name, engine_name in others:
        if same(model_name) == same(name):
            raise ValueError(
                f"{name} shares its weights with {model_name} on {engine_name}; deleting them"
                " would leave that row without its files, so delete that row first"
            )
