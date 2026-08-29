import job_queue
import logging_setup
from evals import build_paraphrased
from models.registry import Role

from .base import register, require_role_ready

log = logging_setup.get_logger(__name__)


@register("paraphrase_questions")
def paraphrase_questions(options: dict) -> None:
    require_role_ready(Role.paraphrasing)
    made = build_paraphrased.build(
        options.get("limit", 100),
        source=options.get("source"),
        set_name=options.get("set_name", "paraphrased"),
        seed=options.get("seed", ""),
        per_source=options.get("per_source"),
        grow=options.get("grow", False),
        originals=options.get("originals"),
    )
    log.info("paraphrase_questions.done", made=made)
    job_queue.enqueue("embed_questions", {})
