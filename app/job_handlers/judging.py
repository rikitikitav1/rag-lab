import logging_setup
import use_cases.judge as judge
from models.eval import QuestionLog
from models.registry import Role
from orm.sync_db import Session
from sqlalchemy import select

from .base import register, require_role_ready

log = logging_setup.get_logger(__name__)


def _target_log_ids(session, options) -> list[int]:
    stmt = select(QuestionLog.id).where(QuestionLog.answered.is_(True))
    if options.get("log_ids"):
        stmt = stmt.where(QuestionLog.id.in_(options["log_ids"]))
    else:
        stmt = stmt.where(QuestionLog.relevance.is_(None))
        if options.get("run_name"):
            stmt = stmt.where(QuestionLog.run_name == options["run_name"])
    return list(session.scalars(stmt))


@register("judge_answers")
def judge_answers(options: dict) -> None:
    require_role_ready(Role.judging)
    with Session() as session:
        log_ids = _target_log_ids(session, options)

    judged = 0
    for log_id in log_ids:
        try:
            _judge_log(log_id)
            judged += 1
        except Exception as e:
            log.error("judge.log_failed", log_id=log_id, error=str(e))
    log.info(
        "judge_answers.done",
        run_name=options.get("run_name"),
        judged=judged,
        total=len(log_ids),
    )


def _judge_log(log_id: int) -> None:
    with Session() as session:
        ql = session.get(QuestionLog, log_id)
        if ql is None or not ql.answered:
            return

        question = ql.question.original_text
        metrics = dict(ql.metrics)

        rv = judge.relevance_verdict(question, ql.answer)
        ql.relevance = rv.verdict.value
        metrics["relevance"] = {"reason": rv.reason, "elapsed": rv.elapsed}

        if ql.context:
            fv = judge.faithful_verdict(question, ql.answer, ql.context)
            ql.faithfulness = fv.verdict.value
            metrics["faithfulness"] = {"reason": fv.reason, "elapsed": fv.elapsed}

        metrics["judge_model"] = rv.model
        ql.metrics = metrics
        session.commit()
