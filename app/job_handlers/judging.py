import logging_setup
from models.eval import Question, QuestionLog
from models.registry import Role
from orm.sync_db import Session
from sqlalchemy import and_, or_, select
from use_cases import judge

from .base import register, require_role_ready

log = logging_setup.get_logger(__name__)

_MAX_JUDGE_ATTEMPTS = 3


def _not_capped(axis: str):
    attempts = QuestionLog.metrics[(axis, "attempts")].as_integer()
    return or_(attempts.is_(None), attempts < _MAX_JUDGE_ATTEMPTS)


def _target_log_ids(session, options) -> list[int]:
    stmt = select(QuestionLog.id).where(QuestionLog.answered.is_(True))
    if options.get("log_ids"):
        return list(session.scalars(stmt.where(QuestionLog.id.in_(options["log_ids"]))))

    stmt = stmt.join(Question, QuestionLog.question_id == Question.id).where(
        or_(
            and_(QuestionLog.relevance.is_(None), _not_capped("relevance")),
            and_(
                QuestionLog.faithfulness.is_(None),
                QuestionLog.context.isnot(None),
                QuestionLog.context != "",
                _not_capped("faithfulness"),
            ),
            and_(
                QuestionLog.completeness.is_(None),
                Question.reference_answer.isnot(None),
                Question.reference_answer != "",
                _not_capped("completeness"),
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

    force = bool(options.get("log_ids"))
    judged = 0
    for log_id in log_ids:
        try:
            if _judge_log(log_id, force=force):
                judged += 1
        except Exception as e:
            log.error("judge.log_failed", log_id=log_id, error=str(e))
    log.info(
        "judge_answers.done",
        run_name=options.get("run_name"),
        judged=judged,
        total=len(log_ids),
    )
    run_name = options.get("run_name")
    if run_name:
        from use_cases import experiment

        experiment.try_aggregate_for_run(run_name)


def _judge_log(log_id: int, force: bool = False) -> bool:
    with Session() as session:
        ql = session.get(QuestionLog, log_id)
        if ql is None or not ql.answered:
            return False

        question = ql.question.original_text
        reference = ql.question.reference_answer
        metrics = dict(ql.metrics)

        wrote = _judge_axis(
            ql, metrics, "relevance", ql.relevance is None, force,
            judge.relevance_verdict, (question, ql.answer),
        )
        wrote |= _judge_axis(
            ql, metrics, "faithfulness", ql.faithfulness is None and bool(ql.context), force,
            judge.faithful_verdict, (question, ql.answer, ql.context),
        )
        wrote |= _judge_axis(
            ql, metrics, "completeness", ql.completeness is None and bool(reference), force,
            judge.completeness_verdict, (question, ql.answer, reference),
        )

        ql.metrics = metrics
        session.commit()
        return wrote


def _judge_axis(ql, metrics, axis, precondition, force, verdict_fn, args) -> bool:
    if not precondition or (not force and _errored(metrics, axis)):
        return False
    v, err = _run_axis(ql.id, axis, verdict_fn, *args)
    if v:
        setattr(ql, axis, str(v.score))
        metrics[axis] = _axis_metric(v)
        return True
    metrics[axis] = _errored_metric(metrics, axis, err)
    return False


def _errored(metrics: dict, axis: str) -> bool:
    return metrics.get(axis, {}).get("attempts", 0) >= _MAX_JUDGE_ATTEMPTS


def _errored_metric(metrics: dict, axis: str, err: str) -> dict:
    attempts = metrics.get(axis, {}).get("attempts", 0) + 1
    return {"error": err, "attempts": attempts}


def _run_axis(log_id, axis, verdict_fn, *args):
    try:
        return verdict_fn(*args), None
    except Exception as e:
        log.error("judge.axis_failed", axis=axis, log_id=log_id, error=str(e))
        return None, type(e).__name__


def _axis_metric(verdict) -> dict:
    return {"reason": verdict.reason, "elapsed": verdict.elapsed, "model": verdict.model}
