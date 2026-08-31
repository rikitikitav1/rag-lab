from typing import Annotated

import job_queue
import logging_setup
from evals import compare, generation_metrics, retrieval_metrics
from evals.loaders import load_logs
from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from models import Job, JobStatus
from orm.sync_db import Session
from pydantic import Field
from sqlalchemy import select
from use_cases import experiment as experiment_uc

log = logging_setup.get_logger(__name__)

mcp_ops = FastMCP("rag-lab-ops", mask_error_details=True)


@mcp_ops.tool(
    name="run_metrics",
    description=(
        "Aggregated eval metrics for one run_name: generation quality "
        "(faithfulness/relevance/completeness on a 0-10 numeric judge, plus 0-1 "
        "normalized and refusal_accuracy) and retrieval (hit_at_k, mrr). Numbers "
        "are averages over the run's judged logs."
    ),
    annotations={"readOnlyHint": True},
)
def run_metrics(
    run_name: Annotated[str, Field(description="The run_name to aggregate.")],
) -> dict:
    if not run_name.strip():
        raise ToolError("run_name must not be empty")
    gen = generation_metrics.evaluate(run_name)
    ret = retrieval_metrics.evaluate(run_name)
    return {"run_name": run_name, **gen, **ret}


@mcp_ops.tool(
    name="compare_runs",
    description=(
        "Compare several runs side by side. Returns per_value metrics keyed by "
        "run_name and an RRF composite (k=60) ranking over five axes: the three "
        "judged ones, the off-domain refusal rate and the supported rate "
        "(retrieval hit_at_k/mrr are reported but excluded from the fusion "
        "since hit_at_k is monotonic in k). winner is the top-ranked run."
    ),
    annotations={"readOnlyHint": True},
)
def compare_runs(
    run_names: Annotated[list[str], Field(description="Run names to compare.")],
) -> dict:
    if not run_names:
        raise ToolError("run_names must not be empty")
    return experiment_uc.compute_results("run", run_names, run_names)


@mcp_ops.tool(
    name="compare_pools",
    description=(
        "Compare runs pool by pool: in_corpus, out_of_corpus, off_domain, rejected. Per arm "
        "returns judged counts, the three judged axes, how often the answer "
        "came from a remote tool against the corpus, how often the coverage gate "
        "fired, latency (avg and p50) and the outcome histogram. Per pair of runs "
        "returns a paired Wilcoxon test over the same questions. Use instead of "
        "compare_runs when the question is where a difference comes from, not "
        "which run wins on average."
    ),
    annotations={"readOnlyHint": True},
)
def compare_pools(
    run_names: Annotated[list[str], Field(description="Run names to compare.")],
) -> dict:
    if not run_names:
        raise ToolError("run_names must not be empty")
    runs = {name: load_logs(name) for name in dict.fromkeys(run_names)}
    empty = [name for name, logs in runs.items() if not logs]
    if empty:
        raise ToolError(f"no logs for runs: {empty}")
    return compare.compare(runs)


@mcp_ops.tool(
    name="list_jobs",
    description=(
        "List background jobs, newest first. Optional filters by status, type "
        "and run_name. Use to check whether eval/judge runs are queued, running "
        "or done."
    ),
    annotations={"readOnlyHint": True},
)
def list_jobs(
    status: Annotated[JobStatus | None, Field(description="Filter by status.")] = None,
    type: Annotated[str | None, Field(description="Filter by job type.")] = None,
    run_name: Annotated[str | None, Field(description="Filter by options.run_name.")] = None,
    limit: Annotated[int, Field(description="Max rows (1-100).", ge=1, le=100)] = 20,
) -> list[dict]:
    with Session() as session:
        stmt = select(Job)
        if status is not None:
            stmt = stmt.where(Job.status == status)
        if type is not None:
            stmt = stmt.where(Job.type == type)
        if run_name is not None:
            stmt = stmt.where(Job.options["run_name"].astext == run_name)
        stmt = stmt.order_by(Job.id.desc()).limit(limit)
        return [
            {
                "id": j.id,
                "type": j.type,
                "status": j.status,
                "run_name": (j.options or {}).get("run_name"),
                "elapsed": j.elapsed,
            }
            for j in session.scalars(stmt)
        ]


@mcp_ops.tool(
    name="cancel_job",
    description=(
        "Cancel a job and its dependent judge (matched by run_name for an "
        "eval_run). Only jobs in 'new' or 'running' are affected. Returns the "
        "list of ids actually cancelled."
    ),
    annotations={"idempotentHint": True},
)
def cancel_job(
    id: Annotated[int, Field(description="Job id to cancel.")],
) -> dict:
    with Session() as session:
        job = session.get(Job, id)
        if job is None:
            raise ToolError(f"job {id} not found")
        ids = [job.id]
        run_name = (job.options or {}).get("run_name")
        if job.type == "eval_run" and run_name:
            deps = session.scalars(
                select(Job.id).where(
                    Job.type == "judge_answers",
                    Job.status.in_([JobStatus.new, JobStatus.running]),
                    Job.options["run_name"].astext == run_name,
                )
            ).all()
            ids += list(deps)
    return {"cancelled": job_queue.cancel(ids)}
