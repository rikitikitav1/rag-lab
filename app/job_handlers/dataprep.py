import contextlib

import job_queue
import llm
import logging_setup
from evals import build_paraphrased, build_veto
from models.registry import Role

from .base import register, require_role_ready

log = logging_setup.get_logger(__name__)


# `keep_alive` is Forever, so a role raised for one job holds the card until ollama dies
@contextlib.contextmanager
def _released(role: Role):
    try:
        yield
    finally:
        # `llm.unload` swallows and logs its own failures, so a job never dies here
        llm.unload(role=str(role))


@register("paraphrase_questions")
def paraphrase_questions(options: dict) -> None:
    require_role_ready(Role.paraphrasing)
    with _released(Role.paraphrasing):
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


# generated, not imported: the gold cannot name a section the cut merged away
@register("build_veto_set")
def build_veto_set(options: dict) -> None:
    require_role_ready(Role.paraphrasing)
    with _released(Role.paraphrasing):
        counted = build_veto.build(
            seed=options.get("seed", ""),
            set_name=options.get("set_name", "veto_v1"),
            variants=options.get("variants"),
            cut_from=options.get("cut_from", "clean_1024"),
            quotas=options.get("quotas"),
        )
    log.info("build_veto_set.done", **counted)
    job_queue.enqueue("embed_questions", {})
