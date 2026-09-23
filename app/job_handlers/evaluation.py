import logging_setup
from errors import StandFault
from evals import runner
from models import Job
from models.eval import QuestionLog
from models.registry import Pipeline, Role
from orm.sync_db import Session
from sqlalchemy import func, select, update

from .base import Final, register, require_card, require_model_ready, require_role_ready

log = logging_setup.get_logger(__name__)


def _claims_on(run_name: str, job_id: int | None) -> tuple[int, list[int]]:
    with Session() as session:
        rows = session.scalar(select(func.count()).select_from(QuestionLog).where(QuestionLog.run_name == run_name))
        others = list(session.scalars(
            select(Job.id).where(Job.type == "eval_run", Job.options["run_name"].astext == run_name, Job.id != job_id)
        ))
    return rows or 0, others


@register("eval_run")
def eval_run(options: dict) -> None:
    model = options.get("model")
    if model:
        require_model_ready(model, "generation")
    else:
        require_role_ready(Role.generation, take_card=False)
    # one preamble takes the card: two, on two engines, handed it back and forth and never started
    require_role_ready(Role.embedding, take_card=False)
    require_card("generation", model, allow_spill=bool(options.get("allow_cpu")))
    resume = bool(options.get("resume"))
    if not resume:
        rows, others = _claims_on(options["run_name"], options.get("_job_id"))
        # every door queues here, and only one of them refused a taken name
        if others or (rows and not options.get("attempts")):
            raise Final(
                f"run {options['run_name']} is taken: {rows} rows, eval_run jobs {others[:5]};"
                " pass resume to answer the rest on its own options, or name a new run"
            )
        # a retry of this very job answers what its first attempt left, not every question again
        resume = bool(rows)
    try:
        answered = runner.run(
            run_name=options["run_name"],
            set_name=options.get("set_name"),
            question_ids=options.get("question_ids"),
            # the runner resolves the default: two deciders is how a null became every run's procedure
            use_rerank=options.get("rerank"),
            pipeline=options.get("pipeline", Pipeline.single_shot),
            language=options.get("language"),
            k=options.get("k"),
            max_hops=options.get("max_hops"),
            model=model,
            fallback_policy=options.get("fallback_policy"),
            gate_signal=options.get("gate_signal"),
            restate_tools=bool(options.get("restate_tools")),
            grade_chunks=bool(options.get("grade_chunks")),
            weak_distance=options.get("weak_distance"),
            orchestrator=options.get("orchestrator"),
            allow_cpu=bool(options.get("allow_cpu")),
            topic_threshold=options.get("topic_threshold"),
            job_id=options.get("_job_id"),
            variant=options.get("variant"),
            resume=resume,
            generation_sampler=options.get("generation_sampler"),
            judge=options.get("judge", True),
        )
    # the worker's retry would answer every question again beside the rows already written
    except StandFault as e:
        raise Final(str(e)) from e
    log.info("eval_run.done", run_name=options["run_name"], answered=answered)


@register("compare_retrieval")
def compare_retrieval(options: dict) -> None:
    from datetime import UTC, datetime

    from models.experiment import Experiment, ExperimentStatus, can_advance
    from use_cases import experiment, retrieval_compare
    from use_cases.retrieval_compare import ComparisonPlan

    job_id = options.get("_job_id")
    with Session() as session:
        exp = session.get(Experiment, options["experiment_id"])
        if exp is None:
            raise LookupError(f"no such experiment: {options['experiment_id']}")
        # a retry arrives with the row left `failed`, and aggregating from there is refused
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

    # outside the session: the arms take minutes and a held transaction is an unannounced lock
    try:
        results = retrieval_compare.run(plan)
    except Exception:
        # the same swap the success path uses: a row that moved on is not failed by this attempt
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

    # check and write as one operation, the same one the generation kind lands through
    if experiment.record_report(options["experiment_id"], results, started):
        log.info(
            "compare.done", experiment=options["experiment_id"], arms=len(results["arms"])
        )
        return

    with Session() as session:
        exp = session.get(Experiment, options["experiment_id"])
        if exp is None:
            raise LookupError(f"no such experiment: {options['experiment_id']}")
        # an hour of measuring is not a log line, so the grid is kept where nothing is lost
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


@register("grade_candidates")
def grade_candidates(options: dict) -> None:
    from evals import grade_candidates as bench

    require_role_ready(Role.grading, take_card=False)
    require_card("grading")
    # the worker's retry would grade the whole file again, and the pass writes only at the end
    try:
        said = bench.run(
            options["candidates"],
            form=options.get("form") or "per_chunk",
            top=options.get("top") or 5,
            limit=options.get("limit"),
            sample=options.get("sample"),
            seed=options.get("seed") or 0,
            shuffle=options.get("shuffle"),
            prompt_version=options.get("prompt_version"),
            name=options.get("name"),
            job_id=options.get("_job_id"),
            question_ids=options.get("question_ids"),
        )
    except StandFault as e:
        raise Final(str(e)) from e
    log.info(
        "grade_candidates.done", questions=said["questions"], asked=said["asked"],
        unreadable=said["unreadable_share"], seconds=said["seconds"], where=said["where"],
        cancelled=said["cancelled"],
    )
