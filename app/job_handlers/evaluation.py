import logging_setup
from evals import runner
from models.registry import Role

from .base import register, require_role_ready

log = logging_setup.get_logger(__name__)


@register("eval_run")
def eval_run(options: dict) -> None:
    require_role_ready(Role.generation)
    require_role_ready(Role.embedding)
    answered = runner.run(
        run_name=options["run_name"],
        set_name=options.get("set_name"),
        question_ids=options.get("question_ids"),
        use_rerank=options.get("rerank"),
        pipeline=options.get("pipeline", "single_shot"),
    )
    log.info("eval_run.done", run_name=options["run_name"], answered=answered)
