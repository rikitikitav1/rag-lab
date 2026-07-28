from datetime import datetime, timezone

import logging_setup
from evals import generation_metrics, retrieval_metrics
from models.eval import QuestionLog
from models.experiment import Experiment, ExperimentStatus
from orm.sync_db import Session
from sqlalchemy import func, select, update

log = logging_setup.get_logger(__name__)

_RRF_K = 60
_AXES = ("faithfulness", "relevance", "completeness")


def _run_pending(session, run_name: str) -> int:
    return session.scalar(
        select(func.count())
        .select_from(QuestionLog)
        .where(
            QuestionLog.run_name == run_name,
            QuestionLog.answered.is_(True),
            QuestionLog.context.isnot(None),
            QuestionLog.context != "",
            QuestionLog.faithfulness.is_(None),
        )
    ) or 0


def _run_count(session, run_name: str) -> int:
    return session.scalar(
        select(func.count()).select_from(QuestionLog).where(QuestionLog.run_name == run_name)
    ) or 0


def _series_complete(session, run_names: list[str]) -> bool:
    return all(
        _run_count(session, rn) > 0 and _run_pending(session, rn) == 0 for rn in run_names
    )


def _rrf(per_value: dict) -> dict[str, float]:
    scores = {v: 0.0 for v in per_value}
    for axis in _AXES:
        ranked = sorted(
            (v for v in per_value if per_value[v].get(axis) is not None),
            key=lambda v: per_value[v][axis],
            reverse=True,
        )
        for rank, v in enumerate(ranked, 1):
            scores[v] += 1 / (_RRF_K + rank)
    return scores


def compute_results(param: str, param_values: list, run_names: list[str]) -> dict:
    per_value = {}
    for value, run_name in zip(param_values, run_names, strict=True):
        gen = generation_metrics.evaluate(run_name)
        ret = retrieval_metrics.evaluate(run_name)
        per_value[str(value)] = {
            "run_name": run_name,
            "faithfulness": gen["faithfulness"],
            "relevance": gen["relevance"],
            "completeness": gen["completeness"],
            "hit_at_k": ret["hit_at_k"],
            "mrr": ret["mrr"],
            "refusal_accuracy": gen["refusal_accuracy"],
            "n_scored": gen["n_scored"],
        }
    scores = _rrf(per_value)
    ranking = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return {
        "param": param,
        "per_value": per_value,
        "composite": {
            "method": "rrf",
            "k": _RRF_K,
            "axes": list(_AXES),
            "ranking": [{"value": v, "rrf": round(s, 5)} for v, s in ranking],
            "winner": ranking[0][0] if ranking else None,
        },
    }


def _finalize(session, exp: Experiment, finished_at: datetime) -> None:
    exp.results = compute_results(exp.param, exp.param_values, exp.run_names)
    exp.finished_at = finished_at
    if exp.started_at:
        exp.elapsed = round((finished_at - exp.started_at).total_seconds(), 1)


def aggregate(experiment_id: int) -> bool:
    with Session() as session:
        exp = session.get(Experiment, experiment_id)
        if exp is None or exp.status != ExperimentStatus.running:
            return False
        if not _series_complete(session, exp.run_names):
            return False
        won = session.execute(
            update(Experiment)
            .where(
                Experiment.id == experiment_id,
                Experiment.status == ExperimentStatus.running,
            )
            .values(status=ExperimentStatus.aggregated)
        ).rowcount
        if not won:
            session.rollback()
            return False
        _finalize(session, exp, datetime.now(timezone.utc))
        session.commit()
        log.info("experiment.aggregated", id=experiment_id, winner=exp.results["composite"]["winner"])
        return True


def try_aggregate_for_run(run_name: str) -> None:
    with Session() as session:
        ids = list(
            session.scalars(
                select(Experiment.id).where(
                    Experiment.status == ExperimentStatus.running,
                    Experiment.run_names.contains([run_name]),
                )
            )
        )
    for experiment_id in ids:
        aggregate(experiment_id)
