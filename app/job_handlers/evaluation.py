import job_queue
import logging_setup
from evals import runner
from models.registry import Model, Role, Status
from orm.sync_db import Session
from sqlalchemy import select

from .base import Deferred, register, require_role_ready

log = logging_setup.get_logger(__name__)


def _require_model_ready(name: str) -> None:
    with Session() as session:
        model = session.scalar(select(Model).where(Model.name == name))
        if model is None:
            session.add(Model(name=name))
            session.commit()
            job_queue.enqueue("pull_llm_model", {"name": name}, queue="io")
            log.info("eval_run.model_pull_enqueued", model=name)
            raise Deferred(30)
    if model.status != Status.ready:
        raise Deferred(30)


@register("eval_run")
def eval_run(options: dict) -> None:
    model = options.get("model")
    if model:
        _require_model_ready(model)
    else:
        require_role_ready(Role.generation)
    require_role_ready(Role.embedding)
    answered = runner.run(
        run_name=options["run_name"],
        set_name=options.get("set_name"),
        question_ids=options.get("question_ids"),
        use_rerank=options.get("rerank"),
        pipeline=options.get("pipeline", "single_shot"),
        language=options.get("language"),
        k=options.get("k"),
        max_hops=options.get("max_hops"),
        model=model,
        fallback_policy=options.get("fallback_policy"),
        gate_signal=options.get("gate_signal"),
        weak_distance=options.get("weak_distance"),
        orchestrator=options.get("orchestrator"),
        allow_cpu=bool(options.get("allow_cpu")),
        topic_threshold=options.get("topic_threshold"),
        job_id=options.get("_job_id"),
    )
    log.info("eval_run.done", run_name=options["run_name"], answered=answered)
