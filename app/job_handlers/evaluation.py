import logging_setup
from evals import runner
from models.registry import Role
from orm.sync_db import Session
from sqlalchemy import update

from .base import register, require_model_ready, require_role_ready

log = logging_setup.get_logger(__name__)


@register("eval_run")
def eval_run(options: dict) -> None:
    model = options.get("model")
    if model:
        require_model_ready(model)
    else:
        require_role_ready(Role.generation)
    require_role_ready(Role.embedding)
    answered = runner.run(
        run_name=options["run_name"],
        set_name=options.get("set_name"),
        question_ids=options.get("question_ids"),
        # the runner resolves the default: two places deciding one switch is how a null
        # ended up recorded as the procedure of every run
        use_rerank=options.get("rerank"),
        pipeline=options.get("pipeline", "single_shot"),
        language=options.get("language"),
        k=options.get("k"),
        max_hops=options.get("max_hops"),
        model=model,
        fallback_policy=options.get("fallback_policy"),
        gate_signal=options.get("gate_signal"),
        weak_distance=options.get("weak_distance"),
        orchestrator=options.get("orchestrator"),
        allow_cpu=bool(options.get("allow_cpu")),
        topic_threshold=options.get("topic_threshold"),
        job_id=options.get("_job_id"),
        variant=options.get("variant"),
    )
    log.info("eval_run.done", run_name=options["run_name"], answered=answered)


@register("compare_retrieval")
def compare_retrieval(options: dict) -> None:
    from datetime import UTC, datetime

    from models.experiment import Experiment, ExperimentStatus, can_advance
    from use_cases import retrieval_compare
    from use_cases.retrieval_compare import ComparisonPlan

    job_id = options.get("_job_id")
    with Session() as session:
        exp = session.get(Experiment, options["experiment_id"])
        if exp is None:
            raise LookupError(f"no such experiment: {options['experiment_id']}")
        # a retry arrives with the row left `failed` by the attempt before it, and
        # aggregating from there is refused, so the grid would be measured and thrown
        # away. The transition is declared; nothing was traversing it
        if exp.status == ExperimentStatus.failed and can_advance(
            exp.status, ExperimentStatus.running
        ):
            exp.status = ExperimentStatus.running
            # elapsed spanning the failed attempt and its backoff describes a queue
            exp.started_at = datetime.now(UTC)
            session.commit()
        plan = ComparisonPlan(
            axes=exp.axes,
            param=exp.param,
            dataset=exp.dataset,
            sample_size=exp.sample_size,
            question_ids=exp.question_ids,
            job_id=job_id,
        )
        started = exp.started_at

    # measured outside the session: the arms take minutes and a transaction held open for
    # them is a lock nobody is waiting to be told about
    try:
        results = retrieval_compare.run(plan)
    except Exception:
        # the same compare-and-swap the success path uses: a row that moved on must not be
        # marked failed by an attempt that no longer owns it
        with Session() as session:
            session.execute(
                update(Experiment)
                .where(
                    Experiment.id == options["experiment_id"],
                    Experiment.status == ExperimentStatus.running,
                )
                .values(status=ExperimentStatus.failed)
            )
            session.commit()
        raise

    with Session() as session:
        exp = session.get(Experiment, options["experiment_id"])
        if exp is None:
            raise LookupError(f"no such experiment: {options['experiment_id']}")
        # check and write as one operation, the way the generation kind does it
        finished = datetime.now(UTC)
        won = session.execute(
            update(Experiment)
            .where(
                Experiment.id == exp.id,
                Experiment.status == ExperimentStatus.running,
            )
            .values(
                results=results,
                status=ExperimentStatus.aggregated,
                finished_at=finished,
                elapsed=(finished - started).total_seconds() if started else None,
            )
        ).rowcount
        if won:
            session.commit()
            log.info(
                "compare.done", experiment=exp.id, arms=len(results["arms"])
            )
            return
        session.rollback()
        # the row moved on. An hour of measuring is not a log line, but a conclusion names
        # the numbers it was written about, so the grid is kept only where nothing is lost
        session.refresh(exp)
        kept = exp.results is None
        if kept:
            exp.results = results
            session.commit()
        log.warning(
            "compare.not_advanced",
            experiment=options["experiment_id"],
            status=str(exp.status),
            grid_kept=kept,
        )
