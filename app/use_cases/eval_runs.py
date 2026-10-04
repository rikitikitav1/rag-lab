import re

import config
import job_queue
import job_specs
import limits
import vocabulary
from errors import Refusal
from models import Job
from models.eval import Question, QuestionLog
from models.registry import Pipeline, refuse_unknown_registry
from sqlalchemy import func, select
from use_cases import retrieval_compare
from vocabulary import GONE, FallbackPolicy, GateSignal, Orchestrator


# rules live beside `measure`, so AXES and the rules cannot name different sets
def validate_axis_values(axis: str, values: list) -> None:
    rule = retrieval_compare.AXIS_RULES.get(axis)
    if rule is None:
        raise Refusal("invalid", f"no rule for axis {axis!r}")
    bad = [v for v in values if not rule(v)]
    if bad:
        detail = f"{axis} takes {retrieval_compare.AXIS_LIMITS[axis]}, got: {bad}"
        if axis == "variant":
            detail += f", declared: {sorted(config.settings.corpus.variants)}"
        raise Refusal("invalid", detail)


def validate_param_values(param: str, values: list, pipeline: Pipeline | None = None) -> None:
    agent_only = (
        "fallback_policy",
        "max_hops",
        "gate_signal",
        "weak_distance",
        "topic_threshold",
        "orchestrator",
    )
    if param in agent_only and pipeline != Pipeline.agent:
        raise Refusal("invalid", f"{param} only applies to the agent pipeline")
    if param == "model":
        bad = [v for v in values if not isinstance(v, str) or not _known_model(v)]
        if bad:
            raise Refusal("invalid", f"invalid model names: {bad}")
    elif param in ("topic_threshold", "weak_distance"):
        bad = [v for v in values if not isinstance(v, int | float) or not 0 <= v <= 2]
        if bad:
            raise Refusal("invalid", f"{param} must be 0..2: {bad}")
    elif param == "variant":
        # inside the chain: outside it every corpus sweep was refused as not an integer
        validate_axis_values("variant", values)
    elif param in ("fallback_policy", "gate_signal", "orchestrator"):
        enums = {
            "fallback_policy": FallbackPolicy,
            "gate_signal": GateSignal,
            "orchestrator": Orchestrator,
        }
        allowed = {p.value for p in enums[param]}
        if param == "orchestrator":
            allowed -= {o.value for o in GONE}
        bad = [v for v in values if v not in allowed]
        if bad:
            raise Refusal("invalid", f"{param} must be one of {sorted(allowed)}: {bad}")
    elif param == "k":
        bad = [v for v in values if not isinstance(v, int) or not 1 <= v <= limits.MAX_K]
        if bad:
            raise Refusal("invalid", f"k must be 1..{limits.MAX_K}: {bad}")
    elif param == "max_hops":
        bad = [v for v in values if not isinstance(v, int) or not 1 <= v <= vocabulary.MAX_HOPS]
        if bad:
            raise Refusal("invalid", f"max_hops must be 1..{vocabulary.MAX_HOPS}: {bad}")
    else:
        bad = [v for v in values if not isinstance(v, int) or v < 1]
        if bad:
            raise Refusal("invalid", f"{param} values must be positive integers")


def _known_model(name: str) -> bool:
    try:
        refuse_unknown_registry(name)
    except ValueError:
        return False
    return True


def value_suffix(value) -> str:
    if isinstance(value, int):
        return f"{value:02d}"
    return re.sub(r"[^a-zA-Z0-9._-]", "_", str(value))


async def _rows_of(session, run_name: str) -> int:
    return (
        await session.scalar(select(func.count()).select_from(QuestionLog).where(QuestionLog.run_name == run_name)) or 0
    )


async def _question_ids_in(session, ids: list[int]) -> set[int]:
    return set((await session.scalars(select(Question.id).where(Question.id.in_(ids)))).all())


# refused at the door, not an hour in: a run over fewer questions than named reads as the named set
async def refuse_missing_questions(session, ids: list[int]) -> None:
    missing = sorted(set(ids) - await _question_ids_in(session, ids))
    if missing:
        raise Refusal(
            "malformed",
            f"{len(missing)} of {len(ids)} question ids are not in the stand: {missing[:20]}",
        )


async def _eval_runs_named(session, run_name: str) -> list:
    return list(
        (
            await session.scalars(
                select(Job)
                .where(Job.type == "eval_run", Job.options["run_name"].astext == run_name)
                .order_by(Job.id.desc())
            )
        ).all()
    )


# a taken name, by its rows or by a job that stopped before its first row, is resumed, not run twice
async def refuse_a_taken_run(session, run_name: str) -> None:
    rows = await _rows_of(session, run_name)
    jobs = await _eval_runs_named(session, run_name)
    if rows or jobs:
        raise Refusal(
            "taken",
            f"run {run_name} has {rows} rows and {len(jobs)} eval_run jobs: pass resume"
            " to answer the rest on its own options, or name a new run",
        )


# accepted questions of a set, and the sources their golds name that are not in search
async def _set_readiness(session, set_name: str) -> tuple[int, list[str]]:
    from models import DataSource
    from models.eval import READ_BY_RUNS

    golds = (await session.scalars(select(Question.gold["file"].astext)
                                   .where(Question.set_name == set_name, READ_BY_RUNS))).all()
    named = {(g or "").split("/")[0] for g in golds} - {""}
    inactive = (await session.scalars(select(DataSource.name).where(DataSource.name.in_(named),
                                                                     DataSource.active.is_(False)))).all()
    return len(golds), sorted(inactive)


# a set with nothing accepted answers nothing and reads done; a gold in a source out of search scores a miss
async def refuse_an_unready_set(session, set_name: str) -> None:
    accepted, inactive = await _set_readiness(session, set_name)
    if not accepted:
        raise Refusal("malformed", f"set {set_name} has no accepted question; accept or judge its pairs first")
    if inactive:
        raise Refusal("malformed", f"set {set_name} has golds in sources not in search: {inactive}; turn them on")


# REST and MCP queue a run through here: a resume reads the run's own options, a new run refuses a taken name
async def queued_eval_run(session, options: dict):
    import job_queue

    if options.get("resume"):
        extra = sorted(set(options) - {"run_name", "resume"})
        options = await resumed_options(session, options.get("run_name"), extra)
    job = await job_queue.prepared("eval_run", options)
    if not options.get("resume"):
        if options.get("run_name"):
            await refuse_a_taken_run(session, options["run_name"])
        if options.get("question_ids"):
            await refuse_missing_questions(session, options["question_ids"])
        elif options.get("set_name"):
            await refuse_an_unready_set(session, options["set_name"])
    return job


# both doors that queue a run resume through here, or /v1/job finished a run on other options
async def resumed_options(session, run_name: str | None, extra: list[str]) -> dict:
    if not run_name or extra:
        raise Refusal(
            "malformed",
            "resume takes run_name alone: a resumed run changes nothing"
            + (f", and {', '.join(extra)} would" if extra else ""),
        )
    jobs = await _eval_runs_named(session, run_name)
    if not jobs:
        raise Refusal("missing", f"no eval_run named {run_name} to resume")
    if any(job.status in job_queue.ACTIVE for job in jobs):
        raise Refusal("busy", f"run {run_name} is still queued or running")
    # the worker's own bookkeeping belongs to the attempt that stopped, not to the resumed one
    options = {k: v for k, v in jobs[0].options.items() if k not in job_specs.WORKER_KEYS}
    return {**options, "resume": True}


# the stand could compose names its own reading doors refuse: base is capped, the arm was not
def refuse_long_names(names: list[str]) -> None:
    too_long = [n for n in names if len(n) > limits.MAX_RUN_NAME]
    if too_long:
        raise Refusal(
            "invalid", f"run names over {limits.MAX_RUN_NAME} characters: {[n[:60] for n in too_long[:3]]}"
        )


# the names a sweep's arms run under, refused whole when one is too long or taken
async def sweep_names(session, base: str, param: str, values: list) -> list[str]:
    names = [f"{base}_{param}_{value_suffix(value)}" for value in values]
    refuse_long_names(names)
    for name in names:
        await refuse_a_taken_run(session, name)
    return names


# one sweep of runs, an arm a value of one parameter, queued in the caller's transaction; ids win over the set
async def queue_sweep(
    session, base: str, param: str, values: list, *, set_name, question_ids, rerank, pipeline: str, language,
    variant, experiment_id: int | None = None,
) -> list:
    if question_ids:
        await refuse_missing_questions(session, question_ids)
    names = await sweep_names(session, base, param, values)
    jobs = []
    for value, name in zip(values, names, strict=True):
        options = {
            "run_name": name,
            # what the row claims it filtered by: ids win over the set, as the runner reads them
            "set_name": None if question_ids else set_name,
            "question_ids": question_ids,
            "rerank": rerank,
            "pipeline": pipeline,
            "language": language,
            # the swept value wins: a sweep over `variant` is not overwritten by the pinned one
            "variant": variant,
            param: value,
        }
        if experiment_id is not None:
            options["experiment_id"] = experiment_id
        jobs.append(await job_queue.add_job(session, "eval_run", options))
    return jobs

