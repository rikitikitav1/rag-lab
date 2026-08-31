from datetime import datetime, timezone

import logging_setup
import numpy as np
from evals import generation_metrics, retrieval_metrics
from evals.loaders import load_logs
from evals.stats import delta_stats as _delta_stats
from models.eval import Question, QuestionLog
from models.experiment import Experiment, ExperimentKind, ExperimentStatus, can_advance
from orm.sync_db import Session
from sqlalchemy import func, select, update
from use_cases import rejudge

log = logging_setup.get_logger(__name__)

_RRF_K = 60
_AXES = ("faithfulness", "relevance", "completeness")
_COMPOSITE_AXES = (*_AXES, "off_domain_refusal_rate", "supported_rate")


# asks the judge's own predicate rather than a second spelling of it: counting only
# faithfulness-with-context let a run whose rows carry no context read as fully judged
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


# 1 is every generation report written before the field existed; readers tell them apart
# by this rather than by the date the row was created
SCHEMA = 2


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


# the report is computed with nothing held open. It is minutes of bootstrap over the rows
# plus a session per arm, and doing that inside the transaction that has just claimed the
# experiment holds a write lock on the row for as long as the arithmetic takes
def _report(exp: Experiment) -> dict:
    if exp.kind == ExperimentKind.rejudge:
        # not the generation report: arms of a rejudge share their answers, so retrieval
        # metrics are identical by construction and an rrf over them ranks noise
        return rejudge.compute_results(
            exp.procedure.get("source_run"), exp.param, rejudge.paired_arms(exp)
        )
    return compute_results(exp.param, exp.param_values, exp.run_names)


def _report_of(experiment_id: int) -> dict:
    with Session() as session:
        return _report(session.get(Experiment, experiment_id))


def aggregate(experiment_id: int) -> bool:
    # the report is computed with nothing held: it is minutes of bootstrap over the rows,
    # and holding the row's lock for that long blocks every reader of the experiment
    with Session() as session:
        exp = session.get(Experiment, experiment_id)
        if exp is None or exp.status != ExperimentStatus.running:
            return False
        if not _series_complete(session, exp.run_names):
            return False
        started_at, kind = exp.started_at, exp.kind
    results = _report_of(experiment_id)

    # then the status and the report land together, guarded on the status the report was
    # computed for, the way the retrieval kind does it. Two writes would let an arm added
    # meanwhile take the row back to `running` and receive a report about fewer arms
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
            # the row moved while the report was being computed, so the report describes a
            # state that is gone. Whoever moved it will be aggregated in its own turn
            session.rollback()
            log.info("experiment.moved_while_aggregating", id=experiment_id)
            return False
        session.commit()
    # a rejudge has no winner: its arms are one set of answers read twice, and the report
    # is the paired delta rather than a ranking
    log.info(
        "experiment.aggregated",
        id=experiment_id,
        kind=str(kind),
        winner=(results.get("composite") or {}).get("winner"),
    )
    return True


# both kinds whose arms are finished by the judge: a rejudge arm is a judge job like a
# generation arm's, so it advances and fails its experiment the same way
_JUDGED_KINDS = (ExperimentKind.generation, ExperimentKind.rejudge)


# a run that exhausted its attempts leaves its experiment `running` for ever: nothing
# aggregates (the series never completes) and nothing moves it on, so the row waits for a
# sibling that is not coming. The transition is declared; this is what traverses it
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
    # a retry arrives with the row left `failed` by the attempt before it, and aggregating
    # from there is refused, so the arms would be judged and thrown away. The retrieval
    # kind already does this in its own handler; the judged kinds had no equivalent
    with Session() as session:
        for exp in session.scalars(
            select(Experiment).where(
                # a rejudge only: its arms are copies, so re-enqueueing the judge is the
                # whole recovery. A generation arm that died has no answers to come back to
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


def try_aggregate_for_run(run_name: str) -> None:
    with Session() as session:
        ids = list(
            session.scalars(
                select(Experiment.id).where(
                    Experiment.kind.in_(_JUDGED_KINDS),
                    Experiment.status == ExperimentStatus.running,
                    Experiment.run_names.contains([run_name]),
                )
            )
        )
    for experiment_id in ids:
        aggregate(experiment_id)
