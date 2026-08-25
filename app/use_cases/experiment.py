from datetime import datetime, timezone

import logging_setup
import numpy as np
from evals import generation_metrics, retrieval_metrics
from evals.loaders import load_logs
from models.eval import QuestionLog
from models.experiment import Experiment, ExperimentStatus
from orm.sync_db import Session
from scipy.stats import wilcoxon
from sqlalchemy import func, select, update

log = logging_setup.get_logger(__name__)

_RRF_K = 60
_BOOTSTRAP_N = 10_000
_AXES = ("faithfulness", "relevance", "completeness")
_COMPOSITE_AXES = (*_AXES, "off_domain_refusal_rate", "supported_rate")


def _run_pending(session, run_name: str) -> int:
    return (
        session.scalar(
            select(func.count())
            .select_from(QuestionLog)
            .where(
                QuestionLog.run_name == run_name,
                QuestionLog.answered.is_(True),
                QuestionLog.context.isnot(None),
                QuestionLog.context != "",
                QuestionLog.faithfulness.is_(None),
            )
        )
        or 0
    )


def _run_count(session, run_name: str) -> int:
    return (
        session.scalar(
            select(func.count()).select_from(QuestionLog).where(QuestionLog.run_name == run_name)
        )
        or 0
    )


def _series_complete(session, run_names: list[str]) -> bool:
    return all(_run_count(session, rn) > 0 and _run_pending(session, rn) == 0 for rn in run_names)


def _rrf(per_value: dict) -> dict[str, float]:
    scores = {v: 0.0 for v in per_value}
    for axis in _COMPOSITE_AXES:
        ranked = sorted(
            (v for v in per_value if per_value[v].get(axis) is not None),
            key=lambda v: per_value[v][axis],
            reverse=True,
        )
        for rank, v in enumerate(ranked, 1):
            scores[v] += 1 / (_RRF_K + rank)
    return scores


def _score(value):
    return None if value is None else int(value)


def _paired_logs(set_a: list, set_b: list) -> list:
    by_id_b = {ql.question_id: ql for ql in set_b}
    return [(a, by_id_b[a.question_id]) for a in set_a if a.question_id in by_id_b]


def _axis_deltas(pairs: list, axis: str) -> list:
    deltas = []
    for a, b in pairs:
        va, vb = _score(getattr(a, axis)), _score(getattr(b, axis))
        if va is not None and vb is not None:
            deltas.append(vb - va)
    return deltas


def _delta_stats(deltas: list, rng) -> dict:
    arr = np.array(deltas, dtype=float)
    boot_means = rng.choice(arr, size=(_BOOTSTRAP_N, arr.size), replace=True).mean(axis=1)
    p = 1.0 if np.all(arr == 0) else float(wilcoxon(arr).pvalue)
    return {
        "mean_delta": round(float(arr.mean()), 3),
        "ci95": [
            round(float(np.percentile(boot_means, 2.5)), 3),
            round(float(np.percentile(boot_means, 97.5)), 3),
        ],
        "p": round(p, 4),
        "n": int(arr.size),
    }


def _compare_question_sets(set_a: list, set_b: list) -> dict:
    pairs = _paired_logs(set_a, set_b)
    rng = np.random.default_rng(42)
    out = {}
    for axis in _AXES:
        deltas = _axis_deltas(pairs, axis)
        out[axis] = _delta_stats(deltas, rng) if deltas else None
    return out


def pairwise_stats(run_a: str, run_b: str) -> dict:
    return _compare_question_sets(load_logs(run_a), load_logs(run_b))


def _annotate_significance(comparisons: dict, alpha: float = 0.05) -> dict:
    tests = [s for axes in comparisons.values() for s in axes.values() if s is not None]
    threshold = alpha / len(tests) if tests else None
    for s in tests:
        s["significant_raw"] = s["p"] < alpha
        s["significant_bonferroni"] = s["p"] < threshold
    return {
        "comparisons": comparisons,
        "method": "bonferroni",
        "alpha": alpha,
        "tests": len(tests),
        "threshold": round(threshold, 5) if threshold else None,
    }


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
            "remote_grounding": gen["remote_grounding"],
            "remote_relevance": gen["remote_relevance"],
            "refusal_accuracy": gen["refusal_accuracy"],
            "off_domain_refusal": gen["off_domain_refusal"],
            "off_domain_grounding": gen["off_domain_grounding"],
            "off_domain_refusal_rate": gen["off_domain_refusal_rate"],
            "supported_rate": gen["supported_rate"],
            "n_off_domain_scored": gen["n_off_domain_scored"],
            "unsupported_external": gen["unsupported_external"],
            "unsupported_off_domain": gen["unsupported_off_domain"],
            "narrated_calls": gen["narrated_calls"],
            "outcomes": gen["outcomes"],
            "false_refusal": gen["false_refusal"],
            "answer_rate": gen["answer_rate"],
            "answered_via_remote": gen["answered_via_remote"],
            "n_scored": gen["n_scored"],
            "n_remote_scored": gen["n_remote_scored"],
        }
    scores = _rrf(per_value)
    ranking = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    winner = ranking[0][0] if ranking else None

    run_by_value = {str(v): rn for v, rn in zip(param_values, run_names, strict=True)}
    comparisons = {}
    if winner and len(run_by_value) > 1:
        winner_logs = load_logs(run_by_value[winner])
        for value, run_name in run_by_value.items():
            if value == winner:
                continue
            comparisons[f"{winner}_vs_{value}"] = _compare_question_sets(
                load_logs(run_name), winner_logs
            )

    return {
        "param": param,
        "per_value": per_value,
        "composite": {
            "method": "rrf",
            "k": _RRF_K,
            "axes": list(_COMPOSITE_AXES),
            "ranking": [{"value": v, "rrf": round(s, 5)} for v, s in ranking],
            "winner": winner,
            "pairwise": _annotate_significance(comparisons),
        },
    }


def _finalize(exp: Experiment, finished_at: datetime) -> None:
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
        _finalize(exp, datetime.now(timezone.utc))
        session.commit()
        log.info(
            "experiment.aggregated", id=experiment_id, winner=exp.results["composite"]["winner"]
        )
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
