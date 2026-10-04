from typing import Annotated, Literal

import config
import job_queue
import limits
import logging_setup
from evals import (
    compare,
    generation_metrics,
    judge_correlation,
    pools,
    question_sets,
    retrieval_metrics,
    run_debts,
    run_tokens,
    stats,
    trace,
)
from evals.loaders import load_logs
from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from models import Job, JobStatus
from models.corpus import Stage
from orm.sync_db import Session
from pydantic import Field
from sqlalchemy import select
from use_cases import experiment as experiment_uc
from use_cases import prereg

log = logging_setup.get_logger(__name__)

mcp_ops = FastMCP("rag-lab-ops", mask_error_details=True)


@mcp_ops.tool(
    name="run_metrics",
    description=(
        "Aggregated eval metrics for one run_name: generation quality "
        "(faithfulness/relevance/completeness on a 0-10 numeric judge, plus 0-1 "
        "normalized and refusal_accuracy) and retrieval (hit_at_k, mrr; by_anchor reads the same run on "
        "questions that name a rare identifier of their section and on the rest, each with its n). Numbers "
        "are averages over the run's judged logs. n_scored counts the in-corpus rows "
        "our judge scored, so a run holding no corpus question reads 0 there however "
        "well it was judged; debts says what the run still owes and why the rest of "
        "its rows cannot owe it: per guest axis the rows owed, scored and abstained, "
        "which of question_text/answer/contexts/reference the others lack, the seconds "
        "one row of this run costs on that axis, and what finishing the debt would cost. "
        "Read debts before spending a judge or a guest pass on a run. tokens says what the run cost: "
        + run_tokens.READS
        + "."
    ),
    annotations={"readOnlyHint": True},
)
def run_metrics(
    run_name: Annotated[str, Field(description="The run_name to aggregate.")],
) -> dict:
    _named_runs([run_name.strip()] if run_name.strip() else [])
    gen = generation_metrics.evaluate(run_name)
    # a mistyped name read as a measured zero on retrieval
    if not gen.get("n_logs"):
        raise ToolError(f"no logs for run {run_name!r}")
    ret = retrieval_metrics.evaluate(run_name)
    return {
        "run_name": run_name,
        **gen,
        **ret,
        "debts": run_debts.safely(run_name),
        "tokens": run_tokens.of(run_name),
    }


@mcp_ops.tool(
    name="question_sets",
    description=(
        "What every question set holds, and therefore which axes a run over it can be scored "
        "on. Per set: how many questions, the pools they fall into, their languages, and three "
        "counts that gate the measurement. with_marked_sources is the corpus pool and what "
        "every retrieval axis ranks against; a set with none of it cannot be scored on hit@k "
        "or mrr. with_reference_answer is what the two guest context axes need "
        "(LLMContextPrecisionWithReference, LLMContextRecall); a set with none of it can only "
        "be scored on faithfulness, and the guest pass will abstain on the rest. A generated set "
        "also carries stages: pairs asked and lost by the generator, pairs read again, and by "
        "language who settled what (accepted, refused by the sieve or the judge, candidates) "
        "and under_the_floor. Read this before spending a judge or a guest pass on a set."
    ),
    annotations={"readOnlyHint": True},
)
def list_question_sets(
    set_name: Annotated[str | None, Field(description="Only this set; omit for every set, largest first.")] = None,
) -> list[dict]:
    found = question_sets.inventory((set_name or "").strip() or None)
    if not found:
        raise ToolError(f"no question set named {set_name!r}")
    return found


@mcp_ops.tool(
    name="questions",
    description=(
        "The questions themselves, one row each, for picking the question_ids of a run: id, set, "
        "language, pool, the text, whether a reference answer is there, how many sources are "
        "marked, and which embedder embedded it. The pool is the rule question_sets counts with, "
        "so the rows of a pool add up to its count there. question_sets says what a set holds; "
        "this says which rows."
    ),
    annotations={"readOnlyHint": True},
)
def list_questions(
    set_name: Annotated[str | None, Field(description="Only this set.")] = None,
    language: Annotated[str | None, Field(description="Only this language, as stored.")] = None,
    pool: Annotated[Literal[pools.POOLS] | None, Field(description="Only this pool.")] = None,
    limit: Annotated[int, Field(ge=1, le=1000)] = 100,
    offset: Annotated[int, Field(ge=0)] = 0,
) -> list[dict]:
    named = (set_name or "").strip() or None
    # a mistyped set read as an empty one while picking question_ids
    if named and not question_sets.inventory(named):
        raise ToolError(f"no question set named {set_name!r}")
    return question_sets.rows(named, language, pool, limit, offset)


def _named_runs(run_names: list[str]) -> list[str]:
    try:
        return compare.named_runs(run_names)
    except ValueError as e:
        raise ToolError(str(e)) from e


# a name with no rows came back as a measured zero, and once as the winner
def _logged(run_names: list[str]) -> list[str]:
    names = _named_runs(run_names)
    empty = [name for name in names if not load_logs(name)]
    if empty:
        raise ToolError(f"no logs for runs: {empty}")
    return names


@mcp_ops.tool(
    name="agent_trace",
    description=(
        "What the agent's own record says about one run's hops and nodes, read from the rows "
        "instead of a query per question: rows by the hop they finished on, how many rows reached "
        "each node and how many steps each took, what the gate said per retrieval, how often the "
        "fallback opened, dropped context or announced the tools, failed hops, and the outcome "
        "paired with the hops the row spent. Rows with no trace are named, not counted as zero: "
        "single-shot rows and rows answered before the trace was recorded carry none."
    ),
    annotations={"readOnlyHint": True},
)
def agent_trace(
    run_name: Annotated[str, Field(description="The run_name to read.")],
) -> dict:
    name = run_name.strip()
    _logged([name] if name else [])
    return trace.of_run(name)


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
    run_name: Annotated[str, Field(default="", description="One run_name, or empty for every judged row.")] = "",
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
        # a p that is not a number walks the step-down and comes back `significant_holm: true`
        dict[str, Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]],
        Field(description="Each test by name with its p-value.", max_length=limits.MAX_TESTS),
    ],
    family: Annotated[str, Field(min_length=1, description="What this family is, in the reader's words.")],
    alpha: Annotated[float, Field(gt=0, lt=1)] = stats.ALPHA,
) -> dict:
    if not tests:
        raise ToolError("a family of no tests corrects nothing")
    named = [{"name": name, "p": p} for name, p in tests.items()]
    # the summary's own `tests` is a count, and spreading it used to overwrite the annotated list
    summary = stats.annotate_holm(named, family, alpha)
    return {
        "tests": named,
        "family": summary["family"],
        "method": summary["method"],
        "alpha": summary["alpha"],
        "n": summary["tests"],
    }


@mcp_ops.tool(
    name="compare_runs",
    description=(
        "Compare several runs side by side. Returns per_value metrics keyed by "
        "run_name and an RRF composite "
        f"(k={config.settings.evals.retrieval_compare.rrf_k}) ranking over five axes: the three "
        "judged ones, the off-domain refusal rate and the supported rate "
        "(retrieval hit_at_k/mrr are reported but excluded from the fusion "
        "since hit_at_k is monotonic in k). winner is the top-ranked run. code says which "
        "code each run's rows were written by, and flags when the arms did not share one."
    ),
    annotations={"readOnlyHint": True},
)
def compare_runs(
    run_names: Annotated[list[str], Field(description="Run names to compare.", max_length=limits.MAX_RUNS)],
) -> dict:
    names = _logged(run_names)
    try:
        return experiment_uc.compute_results("run", names, names)
    except pools.Ambiguous as e:
        raise ToolError(str(e)) from e


@mcp_ops.tool(
    name="language_cost",
    description=(
        "What our own axes charge when the answer comes back in the language it was asked in. "
        "Two arms of one experiment, paired by question over the corpus pool, cut two ways: the "
        "cut declared from the record before the change (rows whose earlier answer was in another "
        "language, the one to quote) and the cut the outcome selected (rows whose language moved). "
        "Give floor_against to add the same arm judged across a reload, which is what the contrast "
        "cannot go below. Carries the comparability block, so a reader sees whether one engine, "
        "one judge prompt and one residency stand behind both arms."
    ),
    annotations={"readOnlyHint": True},
)
def language_cost(
    before: Annotated[str, Field(description="The arm as it stood before the change.")],
    after: Annotated[str, Field(description="The arm after it.")],
    floor_against: Annotated[
        str | None,
        Field(description="A third run holding the same answers judged in another residency."),
    ] = None,
) -> dict:
    from evals import language_cost as costs

    _logged([before, after] + ([floor_against] if floor_against else []))
    return costs.measure(before, after, floor_against)


@mcp_ops.tool(
    name="compare_pools",
    description=(
        "Compare runs pool by pool: in_corpus, out_of_corpus, off_domain, rejected. Per arm "
        "returns judged counts, the three judged axes, how often the answer "
        "came from a remote tool against the corpus, how often the coverage gate "
        "fired, latency (avg and p50) and the outcome histogram. Per pair of runs "
        "returns a paired Wilcoxon test over the same questions. For exactly two runs, "
        "verdicts counts the judge's scores that moved on shared questions, per axis, which "
        "a mean hides when moves cancel. Use instead of "
        "compare_runs when the question is where a difference comes from, not "
        "which run wins on average."
    ),
    annotations={"readOnlyHint": True},
)
def compare_pools(
    run_names: Annotated[list[str], Field(description="Run names to compare.", max_length=limits.MAX_RUNS)],
) -> dict:
    runs = {name: load_logs(name) for name in _named_runs(run_names)}
    # the same refusal `_logged` makes, on logs already loaded here
    empty = [name for name, logs in runs.items() if not logs]
    if empty:
        raise ToolError(f"no logs for runs: {empty}")
    try:
        return compare.compare(runs)
    except pools.Ambiguous as e:
        raise ToolError(str(e)) from e


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
        "seeds that only the aggregation is meant to read. REST serves the "
        "same report at GET /v1/experiment/{id}/report."
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
    from errors import Refusal
    from use_cases import experiment_report

    with Session() as session:
        try:
            return experiment_report.report_of(session, id, pair)
        except Refusal as e:
            raise ToolError(str(e)) from e


@mcp_ops.tool(
    name="engines",
    description=(
        "Which engine holds the GPU right now and which models it has there, whether each vLLM "
        "on the card is asleep, and whether every registered engine answers. Read from the "
        "servers, not from a table. Use before a run or a judging pass to see who owns the card."
    ),
    annotations={"readOnlyHint": True},
)
def engines_on_the_stand() -> dict:
    from use_cases import stand_health

    # one reader for `/health` and this tool, so the two can never tell different stories
    return stand_health.engines_section()


@mcp_ops.tool(
    name="broker_balances",
    description=(
        "What is left on each cloud engine's key, read from the broker's own service route: the "
        "balance in the broker's unit, which reader read it and when. A cloud with no reader named, "
        "or one that did not answer, says why instead of dropping out. Free to call: nothing is "
        "generated. Read it before and after a cloud run to see what the run cost."
    ),
    annotations={"readOnlyHint": True},
)
def broker_balances() -> list[dict]:
    from engines import balances

    return balances.summary()


@mcp_ops.tool(
    name="queue_stats",
    description=(
        "The queue in one screen: how many jobs of each type wait, what runs now and for how long, what finished "
        "in the window by type and status, each type's mean seconds over the last day, and the waiting jobs priced "
        "at those means as an estimate of when the queue ends; a type with no history is listed as unpriced."
    ),
    annotations={"readOnlyHint": True},
)
def queue_stats(
    window_minutes: Annotated[int, Field(description="The window for finished jobs, in minutes.", ge=1, le=1440)] = 60,
) -> dict:
    from use_cases import queue_stats as stats_of

    return stats_of.stats(window_minutes)


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
                # per role, per engine and model; null for a job from before the count
                "tokens": j.tokens,
                # per cloud, the broker's balance before and after; null for a job that called no cloud
                "balances": j.balances,
                "result": j.result,
                "error": j.error,
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


@mcp_ops.tool(
    name="preregister",
    description=(
        "Write what a run promises before it produces a row: the population it measures on, the "
        "control and the arm, the closing columns, the guards, the vetoes and everything declared in "
        "words (stop rules, price, the unflattering expectation). Every column name is checked against "
        "the registry, so a predicate named here has one reading and can be recomputed later; a name the "
        "registry does not know is refused with the known ones listed. A column is read from a run's "
        "question logs or from a recorded grader measurement, and the registry says which. A "
        "preregistration is written once and never edited afterwards. A run started with "
        "`purpose: closing` must name one."
    ),
)
def preregister(
    name: Annotated[str, Field(description="A name for this promise, unique, e.g. `mr4_sgr`.")],
    population: Annotated[
        dict,
        Field(
            description=(
                "{'sets': [question set names], 'question_ids': [ids]?, 'language': 'en' | 'ru'?}; named ids"
                " narrow the sets to a declared draw, a language to its half."
            )
        ),
    ],
    arms: Annotated[dict, Field(description="{'control': ..., 'arm': ...}; `arm` alone under a one-arm `bar`.")],
    closing: Annotated[
        dict,
        Field(
            description=(
                "{'columns': [names], 'arm_should': 'lower' | 'raise', 'floor_value': x?}; shares join into a"
                " union, a judge score closes alone; the bar is the floor run's upper edge or `floor_value`."
                " One arm against a level: {'bar': x, 'read_on': 'point' | 'edge'} instead of a floor."
            )
        ),
    ],
    guards: Annotated[
        list | None,
        Field(
            description=(
                "[{'column': name, 'must_not': 'rise' | 'fall', 'margin': share >= 0, 'sets': [names]?}];"
                " without `sets` a guard reads the closing population."
            )
        ),
    ] = None,
    declared: Annotated[dict | None, Field(description="Stop rules, price, expectations, in words.")] = None,
    vetoes: Annotated[
        list | None,
        Field(
            description=(
                "[{'column': name, 'above': name, 'margin': x >= 0, 'min_rows': n, 'on': 'arm' | 'control'}]: fires"
                " when the mean of `column` exceeds the mean of `above` by more than the margin, on one arm, by"
                " point, and only on at least `min_rows` rows; on fewer it is undecided."
            )
        ),
    ] = None,
) -> dict:
    try:
        return prereg.write(name, population, arms, closing, guards or [], declared or {}, vetoes or [])
    except prereg.Refused as e:
        raise ToolError(str(e)) from e


@mcp_ops.tool(
    name="preregistration",
    description=(
        "Read back a promise by name: population, arms, closing columns, guards and the declared "
        "words. This is the door a session reads after a compaction, because the promise lives in "
        "the base and not in anyone's context."
    ),
    annotations={"readOnlyHint": True},
)
def preregistration(
    name: Annotated[str, Field(description="The preregistration's name.")],
) -> dict:
    try:
        return prereg.read(name)
    except prereg.Refused as e:
        raise ToolError(str(e)) from e


@mcp_ops.tool(
    name="close_preregistration",
    description=(
        "Compute what the promise declared over the named runs and nothing else: shares per arm on "
        "the paired questions, the paired effect in the declared direction with its interval, the "
        "floor band when a second control pass is named as `floor`, and each guard as holds, broken, "
        "undecided or unreadable, read against its margin or, with a `floor` run, the floor's upper edge if higher, "
        "and each veto as fired, quiet or unreadable. Columns read from a measurement take their files "
        "from `measurements`; a veto can be read before the runs exist. "
        "`cleared` is true, false or null, always with `cleared_because`. "
        "A decided close is recorded as `closed_with`, and the promise then closes with those runs only."
    ),
)
def close_preregistration(
    name: Annotated[str, Field(description="The preregistration's name.")],
    runs: Annotated[
        dict | None, Field(description=("{'control': run_name, 'arm': run_name, 'floor': run_name?}."))
    ] = None,
    measurements: Annotated[
        dict | None,
        Field(
            description=(
                "{'control': file, 'arm': file, 'floor': file?}: measurement file names in datasets/measurements."
            )
        ),
    ] = None,
) -> dict:
    try:
        return prereg.close(name, runs, measurements)
    except prereg.Refused as e:
        raise ToolError(str(e)) from e


@mcp_ops.tool(
    name="add_source",
    description=(
        "Declare a source to add, as POST /v1/source takes it: a name, a licence, optionally a language (en or ru), "
        "and exactly one origin: urls to download, a folder placed by hand, a git repository or a family of them, or "
        "pages of a site with the site's settings (main, drop, generated, release, sitemap with include and exclude, "
        "release_page with release_pattern). Beside it what the index reads: categories, category_by_path, versions, "
        "reader, skip, drop_docs_containing, intake knobs, its own questions, reference_leaf. Never an engine: the "
        "route reads each file and picks it. The source starts `declared` and inactive; converting it is a separate "
        "step."
    ),
)
def add_source(
    declaration: Annotated[dict, Field(description=(
        "The declaration, field for field as `sources/<name>.yaml` writes it, e.g. {'name': 'nginx-org-en', "
        "'licence': 'BSD-2', 'pages': [...], 'site': {'main': 'div#content'}, 'categories': ['nginx']}."))],
) -> dict:
    from errors import Refusal
    from pydantic import ValidationError
    from sources.declaration import Declaration
    from use_cases import source_intake

    try:
        declaration = Declaration.model_validate(declaration)
    except ValidationError as e:
        raise ToolError(str(e)) from e
    with Session() as session:
        try:
            source = source_intake.declare(session, declaration)
        except Refusal as e:
            raise ToolError(str(e)) from e
        return source_intake.view(source, 0, 0)


@mcp_ops.tool(
    name="source",
    description="One source by name: its stage, declaration, chunk count and what its raw conversion said.",
    annotations={"readOnlyHint": True},
)
def source(name: Annotated[str, Field(description="The source's name.")]) -> dict:
    from use_cases import source_intake

    with Session() as session:
        found = _source_named(session, name)
        return source_intake.view(found, *source_intake.chunk_counts(session, found))


@mcp_ops.tool(
    name="sources",
    description=(
        "Sources by stage (declared, raw, accepted), a page at a time by name, the same rows as GET /v1/source: "
        "name, stage, language, whether active, raw verdict, chunks in every variant and in the one last indexed."
    ),
    annotations={"readOnlyHint": True},
)
def sources(
    stage: Annotated[Stage | None, Field(description="Filter by stage.")] = None,
    limit: Annotated[int, Field(ge=1, le=1000, description="Rows in the page.")] = 100,
    offset: Annotated[int, Field(ge=0, description="Rows to skip, by name.")] = 0,
) -> list[dict]:
    from use_cases import source_intake

    with Session() as session:
        return source_intake.listed(session, stage, limit, offset)


@mcp_ops.tool(
    name="intake_board",
    description=(
        "The intake at a glance for one corpus variant: sources counted by stage and verdict, the chunks the variant "
        "holds, and the sources that wait, grouped by who moves them next: a person (a dirty or bad run, with its "
        "breaching share and top reasons), a door (onboard, accept, index, turn on) or the queue (a job of theirs is "
        "queued or running). Read this first when tracking a corpus build."
    ),
    annotations={"readOnlyHint": True},
)
def intake_board(
    variant: Annotated[str | None, Field(description="The corpus variant; the served one when left out.")] = None,
) -> dict:
    from use_cases import intake_board as boards

    return boards.board(variant or config.settings.corpus.variant)


@mcp_ops.tool(
    name="source_trail",
    description=(
        "One source in a screen: stage, trust, origin, the run's verdict with its breaching share and reasons, what "
        "it skipped counted by suffix, chunks per variant, drift, its last jobs (onboard, index, report, questions) "
        "with their result in a line, and the step that waits next. The compact companion of `source`."
    ),
    annotations={"readOnlyHint": True},
)
def source_trail(name: Annotated[str, Field(description="The source's name.")]) -> dict:
    from use_cases import intake_board as boards

    found = boards.trail(name)
    if found is None:
        raise ToolError(f"no source named {name}")
    return found


@mcp_ops.tool(
    name="onboard_source",
    description=(
        "Convert a declared source to a raw one: the job routes every file to its engine (markdown as it is, a "
        "page without a text layer to MinerU, the rest to Docling), writes the markdown and a suitability report "
        "without a gold, and marks the source raw with its verdict and reasons. Nothing is indexed. `settings` "
        "names a settings file per engine, e.g. {'docling': 'docling/default'}; another set is another raw folder. "
        "`fresh` reads every piece by its tool again, no kept reading or piece, stamped in the record: for a measure "
        "of the tool itself."
    ),
)
def onboard_source(
    name: Annotated[str, Field(description="The declared source's name.")],
    settings: Annotated[dict | None, Field(description="{engine: 'tool/settings'} overriding intake.settings.")] = None,
    fresh: Annotated[bool, Field(description="Read every piece by its tool again, past every cache.")] = False,
) -> dict:
    from use_cases import source_intake

    with Session() as session:
        _transition(source_intake.check_onboard, _source_named(session, name))
    return {"job_id": job_queue.enqueue("onboard_source", source_intake.onboard_options(name, settings, fresh))}


def _transition(step, source, *args):
    from errors import Final

    try:
        return step(source, *args)
    except Final as e:
        raise ToolError(str(e)) from e


def _source_named(session, name: str):
    from models.corpus import DataSource

    found = session.scalar(select(DataSource).where(DataSource.name == name))
    if found is None:
        raise ToolError(f"no source named {name}")
    return found


@mcp_ops.tool(
    name="set_source_active",
    description="Put a source in search or take it out, by name; its chunks and files stay. `remove_source` deletes.",
    annotations={"idempotentHint": True},
)
def set_source_active(
    name: Annotated[str, Field(description="The source's name.")],
    active: Annotated[bool, Field(description="True puts it in search, false takes it out.")],
) -> dict:
    from use_cases import source_intake

    with Session() as session:
        _transition(source_intake.set_active, _source_named(session, name), active)
        session.commit()
        return {"source": name, "active": active}


@mcp_ops.tool(
    name="accept_source",
    description=(
        "Accept a raw source for indexing: the owner's word that its conversion is fit. A bad raw verdict needs "
        "a `reason`, kept on the row. The index then reads it once a source file names it; it stays out of search "
        "until set_source_active turns it on."
    ),
)
def accept_source(
    name: Annotated[str, Field(description="The raw source's name.")],
    reason: Annotated[str | None, Field(description="Why a bad verdict is accepted anyway.")] = None,
) -> dict:
    from use_cases import source_intake

    with Session() as session:
        replaced = _transition(source_intake.accept, _source_named(session, name), reason)
        session.commit()
        source_intake.drop_folder(replaced, name)
        return {"source": name, "stage": "accepted"}


@mcp_ops.tool(
    name="set_source_intake",
    description=(
        "Set a source's own intake knobs over the stand's defaults in config/intake.yaml, read by its next "
        "onboarding: `settings` per tool (as {\"docling\": \"docling/pypdfium2\"}), "
        + ", ".join(f"`{name}`" for name in config.SOURCE_KNOBS)
        + ". An empty object clears them. Refused while a job reads the source or a source file speaks for it."
    ),
)
def set_source_intake(
    name: Annotated[str, Field(description="The source's name.")],
    intake: Annotated[dict, Field(description="The knobs; a key left out keeps the stand's own.")],
) -> dict:
    from sources.declaration import IntakeOverride
    from use_cases import source_intake

    try:
        block = IntakeOverride(**intake).model_dump(exclude_none=True, exclude_defaults=True)
    except ValueError as e:
        raise ToolError(str(e)) from e
    with Session() as session:
        _transition(source_intake.set_intake, _source_named(session, name), block)
        session.commit()
        return {"source": name, "intake": block}


@mcp_ops.tool(
    name="set_source_fields",
    description=(
        "Set declared fields of a source on its row in place, read by its next onboarding or index, as the REST doors "
        "`PUT /v1/source/{id}/skip_paths|markup|section_roots` do: `skip_paths` (globs under the root), `markup` "
        "(hugo or mdn) with `markup_values` (the site parameters it prints), `section_root_by_path` (a book's heading "
        "root by file glob). An empty value clears a field. Refused while a job reads the source or a source file "
        "speaks for it."
    ),
)
def set_source_fields(
    name: Annotated[str, Field(description="The source's name.")],
    fields: Annotated[dict, Field(description="The fields to set, e.g. {'markup': 'hugo', 'markup_values': {...}}.")],
) -> dict:
    from errors import Refusal
    from use_cases import source_intake

    with Session() as session:
        found = _source_named(session, name)
        try:
            _transition(source_intake.set_fields, found, fields)
        except Refusal as e:
            raise ToolError(str(e)) from e
        session.commit()
        return {"source": name, "declaration": found.declaration}


@mcp_ops.tool(
    name="probe_intake",
    description=(
        "Try intake knobs on a few pages before setting them: a job reads the pages of the source's PDF with its own "
        "knobs and again with these over them, and scores both against the text layer; the job's `result` holds "
        "each side's defect counts, read with list_jobs. The source is not changed. Knobs as in set_source_intake."
    ),
)
def probe_intake(
    name: Annotated[str, Field(description="The source's name.")],
    pages: Annotated[list[int], Field(description="[first, last], counted from 1.", min_length=2, max_length=2)],
    knobs: Annotated[dict, Field(description="The knobs to try over the source's own.")],
    file: Annotated[str | None, Field(description="The PDF, for a source of several.")] = None,
) -> dict:
    from job_specs import Refused

    options = {"source": name, "pages": pages, "knobs": knobs, **({"file": file} if file else {})}
    try:
        return {"job_id": job_queue.enqueue("probe_intake", options)}
    except Refused as e:
        raise ToolError(str(e)) from e


@mcp_ops.tool(
    name="remove_source",
    description=(
        "Delete a source by name: its row, its chunks in every variant, and the stand's own folders of it (raw "
        "conversions, what its job fetched); a folder origin the owner placed stays, and its gate history goes with "
        "the row. Refused while a job reads it or questions have their gold in it."
    ),
    annotations={"destructiveHint": True},
)
def remove_source(name: Annotated[str, Field(description="The source's name.")]) -> dict:
    from errors import Final
    from paths import RAW
    from use_cases import source_intake

    with Session() as session:
        found = _source_named(session, name)
    try:
        return source_intake.remove_source(found, RAW)
    except Final as e:
        raise ToolError(str(e)) from e


@mcp_ops.tool(
    name="remove_variant",
    description="Delete a corpus variant's chunks in every source; the variant the stand searches is refused.",
    annotations={"destructiveHint": True},
)
def remove_variant(variant: Annotated[str, Field(description="The variant's name.")]) -> dict:
    from errors import Final
    from use_cases import source_intake

    try:
        return source_intake.remove_variant(variant, config.settings.corpus.variant)
    except Final as e:
        raise ToolError(str(e)) from e


@mcp_ops.tool(
    name="remove_question_set",
    description=(
        "Delete a question set by name with its questions. Refused while the verdict names it, a job reads it, answer "
        "logs hold its questions, or questions of another set are drawn from it."
    ),
    annotations={"destructiveHint": True},
)
def remove_question_set(
    set_name: Annotated[str, Field(description="The set's name.", max_length=limits.MAX_SET_NAME)],
) -> dict:
    from errors import Final
    from use_cases import question_set_removal

    try:
        return question_set_removal.remove(set_name)
    except Final as e:
        raise ToolError(str(e)) from e


@mcp_ops.tool(
    name="raw_rows",
    description=(
        "The rows of a source's raw report. `pieces`: one per page range or file, the engine and settings that made "
        "it, the conversion's signals (agreement with the file's own text layer, share of the layer, mixed-script "
        "words, seconds). `sections`: one per chapter of a file's whole markdown, the chunker's metrics and gates. "
        "Each row says what it breached; `breached_only` keeps the rows to look at first."
    ),
    annotations={"readOnlyHint": True},
)
def raw_rows(
    name: Annotated[str, Field(description="The source's name.")],
    breached_only: Annotated[bool, Field(description="Only rows that breached something.")] = True,
    kind: Annotated[
        Literal["pieces", "sections"],
        Field(description="pieces: the conversion a page range; sections: the chunker's gates a chapter."),
    ] = "pieces",
    limit: Annotated[int, Field(description="Max rows (1-500).", ge=1, le=500)] = 50,
) -> dict:
    from evals import measurements
    from models.corpus import DataSource
    from use_cases import source_intake

    with Session() as session:
        found = session.scalar(select(DataSource).where(DataSource.name == name))
        report = source_intake.run_under_review(found).get("report") if found is not None else None
        if not report:
            raise ToolError(f"{name} has no raw report yet")
    try:
        rows = measurements.rows_of(measurements.ROOT / report, "rows" if kind == "pieces" else kind)
    except FileNotFoundError as e:
        # a report written before chapters were read has no sections
        raise ToolError(str(e)) from e
    kept = [r for r in rows if r["breached"]] if breached_only else rows
    return {"report": report, "rows": len(rows), "shown": kept[:limit]}
