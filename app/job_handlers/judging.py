from dataclasses import dataclass

import logging_setup
from models.eval import Question, QuestionLog
from models.registry import Purpose, Role
from orm.sync_db import Session
from sqlalchemy import and_, or_, select
from use_cases import experiment, judge, rejudge

from .base import register, require_model_ready, require_role_ready

log = logging_setup.get_logger(__name__)

_MAX_JUDGE_ATTEMPTS = 3


def _not_capped(axis: str):
    attempts = QuestionLog.metrics[(axis, "attempts")].as_integer()
    return or_(attempts.is_(None), attempts < _MAX_JUDGE_ATTEMPTS)


# the one predicate for "this row still has a verdict coming", so the series cannot call itself
# complete while a row the judge would still pick up is unjudged. It used to be spelled here and
# again in `experiment._run_pending`, and the second spelling looked at faithfulness alone
def still_to_judge():
    return or_(
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


def _target_log_ids(session, options) -> list[int]:
    stmt = select(QuestionLog.id).where(QuestionLog.answered.is_(True))
    if options.get("log_ids"):
        return list(session.scalars(stmt.where(QuestionLog.id.in_(options["log_ids"]))))

    stmt = stmt.join(Question, QuestionLog.question_id == Question.id).where(still_to_judge())
    if options.get("run_name"):
        stmt = stmt.where(QuestionLog.run_name == options["run_name"])
    return list(session.scalars(stmt))


@register("judge_answers")
def judge_answers(options: dict) -> None:
    bench = _bench_from(options)
    if bench.model:
        # the arm's override, not the role's model: `require_role_ready` looks at what the
        # role points to, so a mistyped or unpulled tag passed the gate and then failed
        # once per log while the job still reported done
        require_model_ready(bench.model)
    else:
        require_role_ready(Role.judging)
    # resolved once, before any log: a missing version raises here and fails the job, where
    # inside the loop it was swallowed per axis and left the experiment waiting for ever
    for purpose in _PURPOSES:
        bench.template(purpose)

    run_name = options.get("run_name")
    if run_name:
        # before the work, not after: a retry of a failed arm judged its rows and then
        # found the experiment refusing to aggregate from `failed`
        experiment.revive_for_run(run_name)

    with Session() as session:
        log_ids = _target_log_ids(session, options)

    force = bool(options.get("log_ids"))
    judged = 0
    for log_id in log_ids:
        try:
            if _judge_log(log_id, force=force, bench=bench):
                judged += 1
        except Exception as e:
            log.error("judge.log_failed", log_id=log_id, error=str(e))
    log.info(
        "judge_answers.done",
        run_name=options.get("run_name"),
        judged=judged,
        total=len(log_ids),
    )
    if run_name:
        experiment.try_aggregate_for_run(run_name)


# derived from the axes rather than listed, so a fourth axis does not need a second edit
_PURPOSES = tuple(Purpose[f"judge_{axis}"] for axis in rejudge.AXES)


def _bench_from(options: dict) -> judge.Bench:
    # an arm names its judge instead of switching the stand's active one: the role and the
    # active prompt are shared with every other reader, a live answer included
    return rejudge.arm_bench(
        {"judge_model": options.get("judge_model"), **(options.get("judge_prompts") or {})}
    )


def _judge_log(log_id: int, force: bool = False, bench=None) -> bool:
    with Session() as session:
        ql = session.get(QuestionLog, log_id)
        if ql is None or not ql.answered:
            return False

        question = ql.question.original_text
        reference = ql.question.reference_answer
        snapshot = _Snapshot(dict(ql.metrics), dict(ql.prompts), dict(ql.models))
        bench = bench or judge.ACTIVE

        wrote = _judge_axis(
            ql, snapshot, "relevance", ql.relevance is None, force,
            judge.relevance_verdict, (question, ql.answer, bench),
        )
        wrote |= _judge_axis(
            ql, snapshot, "faithfulness", ql.faithfulness is None and bool(ql.context), force,
            judge.faithful_verdict, (question, ql.answer, ql.context, bench),
        )
        wrote |= _judge_axis(
            ql, snapshot, "completeness", ql.completeness is None and bool(reference), force,
            judge.completeness_verdict, (question, ql.answer, reference, bench),
        )

        ql.metrics = snapshot.metrics
        ql.prompts = snapshot.prompts
        ql.models = snapshot.models
        session.commit()
        return wrote


@dataclass
class _Snapshot:
    metrics: dict
    prompts: dict
    models: dict


def _judge_axis(ql, snapshot, axis, precondition, force, verdict_fn, args) -> bool:
    metrics = snapshot.metrics
    if not precondition or (not force and _errored(metrics, axis)):
        return False
    v, err = _run_axis(ql.id, axis, verdict_fn, *args)
    if v:
        setattr(ql, axis, str(v.score))
        metrics[axis] = _axis_metric(v)
        # the row has to say what scored it, not only what answered: two verdicts on the
        # same answer differ by the judge or by its prompt, and a rejudge that cannot name
        # the difference is a comparison of two instruments
        snapshot.models[rejudge.JUDGE_MODEL_KEY] = v.model
        if v.purpose is not None:
            snapshot.prompts[v.purpose.name] = v.prompt_version
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
