from typing import Annotated

import job_queue
import limits
import logging_setup
from evals import (
    compare,
    generation_metrics,
    judge_correlation,
    question_sets,
    retrieval_metrics,
    run_debts,
    stats,
)
from evals.loaders import load_logs
from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from models import Job, JobStatus
from models.experiment import Experiment, ExperimentKind
from orm.sync_db import Session
from pydantic import Field
from sqlalchemy import select
from use_cases import experiment as experiment_uc
from use_cases import rejudge, retrieval_compare

log = logging_setup.get_logger(__name__)

mcp_ops = FastMCP("rag-lab-ops", mask_error_details=True)


@mcp_ops.tool(
    name="run_metrics",
    description=(
        "Aggregated eval metrics for one run_name: generation quality "
        "(faithfulness/relevance/completeness on a 0-10 numeric judge, plus 0-1 "
        "normalized and refusal_accuracy) and retrieval (hit_at_k, mrr). Numbers "
        "are averages over the run's judged logs. n_scored counts the in-corpus rows "
        "our judge scored, so a run holding no corpus question reads 0 there however "
        "well it was judged; debts says what the run still owes and why the rest of "
        "its rows cannot owe it: per guest axis the rows owed, scored and abstained, "
        "which of question_text/answer/contexts/reference the others lack, the seconds "
        "one row of this run costs on that axis, and what finishing the debt would cost. "
        "Read debts before spending a judge or a guest pass on a run."
    ),
    annotations={"readOnlyHint": True},
)
def run_metrics(
    run_name: Annotated[str, Field(description="The run_name to aggregate.")],
) -> dict:
    _named_runs([run_name.strip()] if run_name.strip() else [])
    gen = generation_metrics.evaluate(run_name)
    ret = retrieval_metrics.evaluate(run_name)
    return {"run_name": run_name, **gen, **ret, "debts": run_debts.safely(run_name)}


@mcp_ops.tool(
    name="question_sets",
    description=(
        "What every question set holds, and therefore which axes a run over it can be scored "
        "on. Per set: how many questions, the pools they fall into, their languages, and three "
        "counts that gate the measurement. with_marked_sources is the corpus pool and what "
        "every retrieval axis ranks against; a set with none of it cannot be scored on hit@k "
        "or mrr. with_reference_answer is what the two guest context axes need "
        "(LLMContextPrecisionWithReference, LLMContextRecall); a set with none of it can only "
        "be scored on faithfulness, and the guest pass will abstain on the rest. Read this "
        "before spending a judge or a guest pass on a set."
    ),
    annotations={"readOnlyHint": True},
)
def list_question_sets(
    set_name: Annotated[
        str | None, Field(description="Only this set; omit for every set, largest first.")
    ] = None,
) -> list[dict]:
    found = question_sets.inventory((set_name or "").strip() or None)
    if not found:
        raise ToolError(f"no question set named {set_name!r}")
    return found


def _named_runs(run_names: list[str]) -> list[str]:
    try:
        return compare.named_runs(run_names)
    except ValueError as e:
        raise ToolError(str(e)) from e


@mcp_ops.tool(
    name="judge_correlation",
    description=(
        "Our judge against the standard's on the same rows: spearman of our faithfulness "
        "with ragas faithfulness, the overlap covariate, the partial correlation behind it, "
        "and the strata by code share of the context. Rows our own outcome calls a refusal "
        "are excluded and counted apart. Reports whether each prediction declared before the "
        "count held. Rows carry guest scores only where the guest pass has run."
    ),
    annotations={"readOnlyHint": True},
)
def judge_correlation_report(
    run_name: Annotated[
        str, Field(default="", description="One run_name, or empty for every judged row.")
    ] = "",
) -> dict:
    name = run_name.strip() or None
    if name:
        _named_runs([name])
    rows, counts = judge_correlation.rows_of(name)
    if not rows:
        raise ToolError("no row carries both our faithfulness and the guest's")
    return {"run_name": name, **judge_correlation.report(rows, counts)}


@mcp_ops.tool(
    name="holm_over",
    description=(
        "Correct a family of tests the reader declares, rather than the family one record "
        "happens to hold. Give the p-values with a name each and say what the family is; "
        "returns each test with its holm threshold and whether it survives, plus the family "
        "as written. Use it when the arms being read come from more than one experiment: a "
        "report corrects over its own record, and a wider reading is a wider family."
    ),
    annotations={"readOnlyHint": True},
)
def holm_over(
    tests: Annotated[
        dict[str, float],
        Field(description="Each test by name with its p-value.", max_length=limits.MAX_RUNS),
    ],
    family: Annotated[
        str, Field(min_length=1, description="What this family is, in the reader's words.")
    ],
    alpha: Annotated[float, Field(default=0.05, gt=0, lt=1)] = 0.05,
) -> dict:
    if not tests:
        raise ToolError("a family of no tests corrects nothing")
    named = [{"name": name, "p": p} for name, p in tests.items()]
    return {"tests": named, **stats.annotate_holm(named, family, alpha)}


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
    run_names: Annotated[
        list[str], Field(description="Run names to compare.", max_length=limits.MAX_RUNS)
    ],
) -> dict:
    names = _named_runs(run_names)
    return experiment_uc.compute_results("run", names, names)


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
    run_names: Annotated[
        list[str], Field(description="Run names to compare.", max_length=limits.MAX_RUNS)
    ],
) -> dict:
    runs = {name: load_logs(name) for name in _named_runs(run_names)}
    empty = [name for name, logs in runs.items() if not logs]
    if empty:
        raise ToolError(f"no logs for runs: {empty}")
    return compare.compare(runs)


READING = {
    ExperimentKind.generation: experiment_uc.for_reading,
    ExperimentKind.retrieval: retrieval_compare.for_reading,
    ExperimentKind.rejudge: rejudge.for_reading,
}


@mcp_ops.tool(
    name="experiment_results",
    description=(
        "The report of one experiment: its status and conclusion, the arms with "
        "the judge that scored each and their answers_digest, and the paired "
        "delta of every pair on every axis with its interval, its p and whether "
        "it survives the multiplicity correction over the family this report "
        "holds. For a rejudge, same_answers says the arms judged the same "
        "answers; a false there means the deltas compare two different sets. "
        "Read this instead of the raw record: the record carries halves and "
        "seeds that only the aggregation is meant to read."
    ),
    annotations={"readOnlyHint": True},
)
def experiment_results(
    id: Annotated[int, Field(description="Experiment id.", ge=1)],
    pair: Annotated[
        str | None,
        Field(description="Only this pair, as it is named in the report."),
    ] = None,
) -> dict:
    with Session() as session:
        exp = session.get(Experiment, id)
        if exp is None:
            raise ToolError(f"no experiment {id}")
        # each kind writes its own shape, and only its writer knows which key holds what
        read = READING[ExperimentKind(exp.kind)](exp.results or {})
        out = {
            "id": exp.id,
            "name": exp.name,
            "kind": exp.kind,
            "status": exp.status,
            "conclusion": exp.conclusion,
            **{k: v for k, v in read.items() if k != "deltas"},
        }
        deltas = read["deltas"]
        if pair is not None:
            if pair not in deltas:
                raise ToolError(f"no pair {pair!r}; this report has {sorted(deltas)}")
            deltas = {pair: deltas[pair]}
        # halves are the aggregation's own check and read as four more numbers per axis here
        out["deltas"] = {
            name: {k: v for k, v in body.items() if k != "halves"}
            for name, body in deltas.items()
        }
        return out


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
        if session.get(Job, id) is None:
            raise ToolError(f"job {id} not found")
    return {"cancelled": job_queue.cancel_with_its_judge(id)}
