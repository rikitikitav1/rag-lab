import logging_setup
import use_cases.judge as judge
from models.eval import Question, QuestionLog
from models.registry import Role
from orm.sync_db import Session
from sqlalchemy import and_, or_, select

from .base import register, require_role_ready

log = logging_setup.get_logger(__name__)


def _target_log_ids(session, options) -> list[int]:
    stmt = select(QuestionLog.id).where(QuestionLog.answered.is_(True))
    if options.get("log_ids"):
        stmt = stmt.where(QuestionLog.id.in_(options["log_ids"]))
    else:
        stmt = stmt.join(Question, QuestionLog.question_id == Question.id).where(
            or_(
                QuestionLog.relevance.is_(None),
                and_(
                    QuestionLog.completeness.is_(None),
                    Question.reference_answer.isnot(None),
                ),
            )
        )
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
        reference = ql.question.reference_answer
        metrics = dict(ql.metrics)

        if ql.relevance is None:
            v = _run_axis(log_id, "relevance", judge.relevance_verdict, question, ql.answer)
            if v:
                ql.relevance = v.verdict.value
                metrics["relevance"] = _axis_metric(v)

        if ql.faithfulness is None and ql.context:
            v = _run_axis(
                log_id, "faithfulness", judge.faithful_verdict, question, ql.answer, ql.context
            )
            if v:
                ql.faithfulness = v.verdict.value
                metrics["faithfulness"] = _axis_metric(v)

        if ql.completeness is None and reference:
            v = _run_axis(
                log_id, "completeness", judge.completeness_verdict, question, ql.answer, reference
            )
            if v:
                ql.completeness = v.verdict.value
                metrics["completeness"] = _axis_metric(v)

        ql.metrics = metrics
        session.commit()


def _run_axis(log_id, axis, verdict_fn, *args):
    try:
        return verdict_fn(*args)
    except Exception as e:
        log.error("judge.axis_failed", axis=axis, log_id=log_id, error=str(e))
        return None


def _axis_metric(verdict) -> dict:
    return {"reason": verdict.reason, "elapsed": verdict.elapsed, "model": verdict.model}
