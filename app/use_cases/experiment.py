from datetime import datetime, timezone

import logging_setup
import numpy as np
from evals import generation_metrics, retrieval_metrics
from evals.loaders import load_logs
from evals.stats import annotate_holm, deltas_over, score_of
from evals.stats import delta_stats as _delta_stats
from models.eval import Question, QuestionLog
from models.experiment import Experiment, ExperimentKind, ExperimentStatus, can_advance
from orm.sync_db import Session
from sqlalchemy import func, select, update
from use_cases import rejudge

log = logging_setup.get_logger(__name__)

_RRF_K = 60
_AXES = rejudge.AXES
_COMPOSITE_AXES = (*_AXES, "off_domain_refusal_rate", "supported_rate")


# the judge's own predicate: counting faithfulness-with-context read a run as fully judged
def _run_pending(session, run_name: str) -> int:
    from job_handlers.judging import still_to_judge

    return (
        session.scalar(
            select(func.count())
            .select_from(QuestionLog)
            .join(Question, QuestionLog.question_id == Question.id)
            .where(
                QuestionLog.run_name == run_name,
                QuestionLog.answered.is_(True),
                still_to_judge(),
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


def _paired_logs(set_a: list, set_b: list) -> list:
    by_id_b = {ql.question_id: ql for ql in set_b}
    return [(a, by_id_b[a.question_id]) for a in set_a if a.question_id in by_id_b]


def _axis_deltas(pairs: list, axis: str) -> list:
    before = {a.question_id: score_of(getattr(a, axis)) for a, _ in pairs}
    after = {b.question_id: score_of(getattr(b, axis)) for _, b in pairs}
    return deltas_over(before, after, [a.question_id for a, _ in pairs])


def _compare_question_sets(set_a: list, set_b: list) -> dict:
    pairs = _paired_logs(set_a, set_b)
    rng = np.random.default_rng(42)
    out = {}
    for axis in _AXES:
        deltas = _axis_deltas(pairs, axis)
        out[axis] = _delta_stats(deltas, rng) if deltas else None
    return out


def _annotate_significance(comparisons: dict, alpha: float = 0.05) -> dict:
    tests = [s for axes in comparisons.values() for s in axes.values() if s is not None]
    # a reader who declared a narrower family before the run corrects over that one, and says so
    family = annotate_holm(tests, "every pair of the grid on every axis", alpha)
    return {"comparisons": comparisons, **family}


# 1 before the field; 2 Bonferroni; 3 Holm, the family named, `answered_ungrounded` read
SCHEMA = 3


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
        "schema": SCHEMA,
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


def for_reading(results: dict) -> dict:
    pairwise = (results.get("composite") or {}).get("pairwise") or {}
    return {
        "source_run": None,
        # the arms answer the same questions, and every axis is compared over that pairing
        "pairing": "by question",
        "multiplicity": {k: v for k, v in pairwise.items() if k != "comparisons"},
        "ranking": {k: v for k, v in (results.get("composite") or {}).items()
                    if k in ("method", "winner", "ranking", "axes")},
        "arms": {
            value: {k: v for k, v in body.items()
                    if k in ("run_name", "n_scored", "mrr", "hit_at_k", *_COMPOSITE_AXES)}
            for value, body in (results.get("per_value") or {}).items()
        },
        "deltas": pairwise.get("comparisons") or {},
    }


# nothing held open: minutes of bootstrap would hold a write lock for that long
def _report(exp: Experiment) -> dict:
    if exp.kind == ExperimentKind.rejudge:
        # arms of a rejudge share their answers, so an rrf over their retrieval ranks noise
        return rejudge.compute_results(
            exp.procedure.get("source_run"), exp.param, rejudge.paired_arms(exp)
        )
    return compute_results(exp.param, exp.param_values, exp.run_names)


def _report_of(experiment_id: int) -> dict:
    with Session() as session:
        return _report(session.get(Experiment, experiment_id))


def aggregate(experiment_id: int) -> bool:
    # computed with nothing held: holding the row's lock blocks every reader of the experiment
    with Session() as session:
        exp = session.get(Experiment, experiment_id)
        if exp is None or exp.status != ExperimentStatus.running:
            return False
        if not _series_complete(session, exp.run_names):
            return False
        started_at, kind = exp.started_at, exp.kind
    results = _report_of(experiment_id)

    if not record_report(experiment_id, results, started_at):
        # the row moved while the report was computed; whoever moved it aggregates in its own turn
        log.info("experiment.moved_while_aggregating", id=experiment_id)
        return False
    # a rejudge has no winner: its arms are one set of answers read twice
    log.info(
        "experiment.aggregated",
        id=experiment_id,
        kind=str(kind),
        winner=(results.get("composite") or {}).get("winner"),
    )
    return True


# guarded on the status the report was computed for, or an arm added meanwhile takes it back
def record_report(experiment_id: int, results: dict, started_at) -> bool:
    with Session() as session:
        finished = datetime.now(timezone.utc)
        won = session.execute(
            update(Experiment)
            .where(
                Experiment.id == experiment_id,
                Experiment.status == ExperimentStatus.running,
            )
            .values(
                results=results,
                status=ExperimentStatus.aggregated,
                finished_at=finished,
                elapsed=(
                    round((finished - started_at).total_seconds(), 1)
                    if started_at
                    else None
                ),
            )
        ).rowcount
        if not won:
            session.rollback()
            return False
        session.commit()
        return True


# a rejudge arm is a judge job like a generation arm's, and advances the same way
_JUDGED_KINDS = (ExperimentKind.generation, ExperimentKind.rejudge)


# a run out of attempts leaves the experiment `running` for ever; this traverses it
def mark_failed_for_run(run_name: str) -> None:
    with Session() as session:
        won = session.execute(
            update(Experiment)
            .where(
                Experiment.kind.in_(_JUDGED_KINDS),
                Experiment.status == ExperimentStatus.running,
                Experiment.run_names.contains([run_name]),
            )
            .values(status=ExperimentStatus.failed)
        ).rowcount
        if won:
            session.commit()
            log.warning("experiment.failed", run_name=run_name)
        else:
            session.rollback()


def revive_for_run(run_name: str) -> None:
    # a retry arrives with the row `failed`, and aggregating from there is refused
    with Session() as session:
        for exp in session.scalars(
            select(Experiment).where(
                # a rejudge only: a generation arm that died has no answers to come back to
                Experiment.kind == ExperimentKind.rejudge,
                Experiment.status == ExperimentStatus.failed,
                Experiment.run_names.contains([run_name]),
            )
        ):
            if can_advance(exp.status, ExperimentStatus.running):
                exp.status = ExperimentStatus.running
                exp.started_at = datetime.now(timezone.utc)
                log.info("experiment.revived", id=exp.id, run_name=run_name)
        session.commit()


# taken back only when every arm it still names is judged, so a real gap stays visible
def revive_if_complete(session, exp) -> bool:
    return (
        exp is not None
        and exp.status == ExperimentStatus.failed
        and can_advance(exp.status, ExperimentStatus.running)
        and _series_complete(session, exp.run_names)
    )


def try_aggregate_for_run(run_name: str) -> None:
    with Session() as session:
        ids = list(
            session.scalars(
                select(Experiment.id).where(
                    Experiment.kind.in_(_JUDGED_KINDS),
                    # `failed` too: cancelling one arm fails the row, and a sibling would finish into it
                    Experiment.status.in_((ExperimentStatus.running, ExperimentStatus.failed)),
                    Experiment.run_names.contains([run_name]),
                )
            )
        )
    for experiment_id in ids:
        with Session() as session:
            exp = session.get(Experiment, experiment_id)
            if revive_if_complete(session, exp):
                exp.status = ExperimentStatus.running
                session.commit()
                log.info("experiment.revived_by_sibling", id=experiment_id, run_name=run_name)
        aggregate(experiment_id)
