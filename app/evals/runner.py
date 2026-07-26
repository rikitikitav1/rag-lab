import sys
import time

import job_queue
import logging_setup
from models.eval import Question
from models.registry import Pipeline
from orm.sync_db import Session
from sqlalchemy import select
from use_cases import agent, chat

log = logging_setup.get_logger(__name__)


def _target_texts(set_name: str | None, question_ids: list[int] | None) -> list[str]:
    with Session() as session:
        stmt = select(Question.original_text)
        if question_ids:
            stmt = stmt.where(Question.id.in_(question_ids))
        elif set_name:
            stmt = stmt.where(Question.set_name == set_name)
        return list(session.scalars(stmt))


def _answer_one(
    text: str,
    run_name: str,
    use_rerank: bool | None,
    pipeline: str,
) -> None:
    if pipeline == Pipeline.agent:
        agent.run(text, run_name=run_name)
    elif pipeline == Pipeline.single_shot:
        chat.answer(
            text, add_context=True, run_name=run_name, use_rerank=use_rerank
        )
    else:
        raise ValueError(f"unknown pipeline: {pipeline}")


def run(
    run_name: str,
    set_name: str | None = None,
    question_ids: list[int] | None = None,
    use_rerank: bool | None = None,
    pipeline: str = "single_shot",
) -> int:
    texts = _target_texts(set_name, question_ids)
    answered = 0
    for text in texts:
        try:
            _answer_one(text, run_name, use_rerank, pipeline)
            answered += 1
        except Exception as e:
            log.error("eval_run.answer_failed", run_name=run_name, error=str(e))
    job_queue.enqueue("judge_answers", {"run_name": run_name})
    log.info(
        "eval_run.answered", run_name=run_name, answered=answered, total=len(texts)
    )
    return answered


if __name__ == "__main__":
    set_name = sys.argv[1] if len(sys.argv) > 1 else "curated"
    run_name = sys.argv[2] if len(sys.argv) > 2 else f"{set_name}_{int(time.time())}"
    n = run(run_name, set_name=set_name)
    print(f"run: {run_name} | set: {set_name} | answered: {n} | judging enqueued")
