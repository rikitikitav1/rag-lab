import argparse
import json
import re
import subprocess
import urllib.parse
import urllib.request
from functools import lru_cache
from pathlib import Path

API = "http://localhost:8000"
ROOT = Path(__file__).resolve().parent.parent


def sh(*args: str) -> str:
    done = subprocess.run(args, capture_output=True, text=True)
    return done.stdout.strip() if done.returncode == 0 else ""


# asked of the worker rather than imported: this script runs on the host, where the
# application's dependencies are not installed, and importing app code to read one
# constant is what broke it. One owner still, just reached the way everything else is
@lru_cache(maxsize=1)
def vector_index_prefix() -> str:
    out = _in_worker("from use_cases.index import VECTOR_INDEX_PREFIX as p; print(p)")
    return out or "data_chunks_embedding_"


def get(path: str):
    with urllib.request.urlopen(f"{API}{path}", timeout=30) as response:
        return json.load(response)


# docker reports UTC with a Z and nanoseconds, git reports an offset
def _moment(stamp: str):
    from datetime import datetime

    parts = re.split(r"([+-]\d\d:\d\d|Z)$", stamp.strip(), maxsplit=1)
    body, offset = parts[0], parts[1] if len(parts) > 1 else ""
    if "." in body:
        whole, _, fraction = body.partition(".")
        body = f"{whole}.{fraction[:6]}"
    return datetime.fromisoformat(f"{body}{'+00:00' if offset in ('Z', '') else offset}")


def _newest_source() -> tuple[float, str]:
    files = [p for p in (ROOT / "app").rglob("*.py") if "__pycache__" not in p.parts]
    # the run reads thresholds and the window from the config, so it counts as source here
    files.append(ROOT / "config.yaml")
    newest = max(files, key=lambda p: p.stat().st_mtime)
    return newest.stat().st_mtime, str(newest.relative_to(ROOT))


# the worker mounts the working tree, so what matters is the file on disk, not the commit
def worker_newer_than_sources() -> tuple[bool, str]:
    started = sh("docker", "inspect", "-f", "{{.State.StartedAt}}", "rag-lab-worker-1")
    if not started:
        return False, "cannot read the worker start time"
    mtime, name = _newest_source()
    return _moment(started).timestamp() > mtime, f"worker started {started}, newest source {name}"


# code_version in the snapshot is the commit, so an edited tree makes it name code that never ran
def tree_is_clean() -> tuple[bool, str]:
    dirty = sh("git", "status", "--porcelain")
    files = [line.split(maxsplit=1)[-1] for line in dirty.splitlines() if line.strip()]
    return not files, f"working tree: {'clean' if not files else f'{len(files)} edited, {files[:3]}'}"


def worker_imports() -> tuple[bool, str]:
    out = sh(
        "docker", "compose", "exec", "-T", "worker", "python", "-c",
        "import sys; sys.path.insert(0, '/app/app');"
        " from orchestrators import graph, react;"
        " from use_cases import agent; print('ok')",
    )
    return out.endswith("ok"), f"imports inside the worker: {out or 'failed'}"


def window_matches_config() -> tuple[bool, str]:
    configured = sh(
        "docker", "compose", "exec", "-T", "rag-lab", "python", "-c",
        "import sys; sys.path.insert(0, '/app/app'); import config;"
        " print(config.settings.llm.context_length)",
    )
    # whichever generator is actually loaded, not only the configured one: a run with a
    # model override leaves the configured name unloaded, and the check then read
    # "the server says nothing" as though the window disagreed
    out = _in_worker(
        "import json, llm;"
        " print(json.dumps({'loaded': [e['model'] for e in llm.residency()],"
        " 'configured': llm.resolve_name('generation')}))"
    )
    if not out.startswith("{"):
        return False, f"context window: cannot read the residency ({out[:40] or 'no answer'})"
    state = json.loads(out)
    asked = state["configured"] if state["configured"] in state["loaded"] else next(
        (m for m in state["loaded"] if "embed" not in m and "bge" not in m), None
    )
    if asked is None:
        return True, (
            f"context window: config {configured or 'unknown'}, no generator loaded"
            " (descriptive: nothing to compare, ask one a question first)"
        )
    live = _in_worker(f"import llm; print(llm.server_context_length({asked!r}))")
    # the server is the authority: a stray env var in a running container beat the config once
    ok = bool(configured) and live == configured
    return ok, f"context window: config {configured or 'unknown'}, {asked} says {live or 'unknown'}"


# the scheduler keeps reporting free VRAM after the card is gone, so ask what is actually resident
def models_are_on_the_card() -> tuple[bool, str]:
    out = sh(
        "docker", "compose", "exec", "-T", "rag-lab", "python", "-c",
        "import sys, json; sys.path.insert(0, '/app/app'); import llm;"
        " roles = {r: llm.resolve_name(r) for r in"
        " ('generation', 'embedding', 'judging', 'paraphrasing')};"
        " print(json.dumps({'roles': roles, 'loaded': llm.residency()}))",
    )
    if not out.startswith("{"):
        return False, f"residency: cannot read ({out[:60] or 'no answer'})"
    state = json.loads(out)
    loaded = {e["model"]: e for e in state["loaded"]}
    if not loaded:
        return False, "no model is loaded: ask one to load before reading the window"
    # naming the role matters: a job runs one model, and the others being resident proves nothing
    lines = []
    for role, name in state["roles"].items():
        entry = loaded.get(name)
        where = f"{entry['vram_mb']}/{entry['size_mb']} MiB" if entry else "not resident"
        lines.append(f"{role}={name} {where}")
    off = [e["model"] for e in state["loaded"] if e["vram_mb"] == 0]
    return not off, "; ".join(lines) + (f"; ON CPU: {', '.join(off)}" if off else "")


def queue_is_idle() -> tuple[bool, str]:
    jobs = get("/v1/job?status=new&status=running&limit=1000")
    return not jobs, f"{len(jobs)} jobs still queued or running"



# the answer is the last line: anything that touches the application may log on the way,
# and a check that reads the first line calls a log line a broken answer
def _in_worker(code: str) -> str:
    out = sh(
        "docker", "compose", "exec", "-T", "worker", "python", "-c",
        "import sys; sys.path.insert(0, '/app/app');" + code,
    )
    lines = [line for line in out.splitlines() if line.strip()]
    return lines[-1] if lines else ""


def corpus_variant_is_usable() -> tuple[bool, str]:
    out = _in_worker(
        "import json, config, db;"
        " v = config.settings.corpus.variant;"
        " print(json.dumps({'variant': v, 'known': db.corpus_variants()}))"
    )
    if not out.startswith("{"):
        return False, f"corpus variant: cannot read ({out or 'no answer'})"
    state = json.loads(out)
    counts = {row["variant"]: row["chunks"] for row in state["known"]}
    chunks = counts.get(state["variant"], 0)
    listing = ", ".join(f"{k}={v}" for k, v in counts.items()) or "none"
    return chunks > 0, f"corpus variant {state['variant']}: {chunks} chunks (known: {listing})"


# a generic plan cannot use a partial index, and the fallback is silent
# every indexed variant, not only the served one: the depth at which the planner stops
# walking the index is priced against a read of the whole table, so indexing one variant
# changes where every other one stops. Checking only the served variant is how the
# crossover moved twice without anybody noticing
def every_variant_walks_its_index() -> tuple[bool, str]:
    out = _in_worker(
        "import json; from use_cases import search_depth;"
        " print(json.dumps(search_depth.audit()))"
    )
    if not out.startswith("["):
        return False, f"depth audit: cannot read ({out[-60:] or 'no answer'})"
    rows = json.loads(out)
    if not rows:
        return True, "depth: no variant holds rows yet"
    sorting = [r["variant"] for r in rows if not r["serving_uses_index"]]
    reading = ", ".join(
        f"{r['variant']}@{r['serving']}"
        + ("" if r["serving_uses_index"] else " SORTS")
        for r in rows
    )
    return not sorting, (
        f"depth ({rows[0]['declared']}, {rows[0]['rows_estimate']} rows estimated): {reading}"
    )


def table_is_vacuumed() -> tuple[bool, str]:
    out = _in_worker(
        "from orm.sync_db import engine; from sqlalchemy import text;"
        " print(engine.connect().execute(text(\"SELECT n_dead_tup FROM pg_stat_user_tables"
        " WHERE relname = 'data_chunks'\")).scalar())"
    )
    dead = int(out) if out.isdigit() else -1
    # a mass update leaves dead entries in the hnsw graph, and retrieval shrinks quietly
    return 0 <= dead <= 1000, f"dead tuples in data_chunks: {out or 'unknown'}"


def one_question_per_original() -> tuple[bool, str]:
    out = _in_worker(
        "from orm.sync_db import engine; from sqlalchemy import text;"
        " print(engine.connect().execute(text(\"SELECT coalesce(string_agg(set_name || '=' ||"
        " n, ', '), 'none') FROM (SELECT set_name, count(*) n FROM (SELECT set_name,"
        " source_question_id FROM questions WHERE source_question_id IS NOT NULL"
        " GROUP BY 1, 2 HAVING count(*) > 1) d GROUP BY set_name) s\")).scalar())"
    )
    # a job the worker requeues can paraphrase the same original twice, and the second
    # paraphrase is new text with a new hash, so nothing else catches it
    return out == "none", f"originals paraphrased more than once: {out or 'unknown'}"


PAIRED_SETS = """
SELECT coalesce(string_agg(base || '=' || cnt, ', '), 'none') FROM (
  SELECT base, count(*) cnt FROM (
    SELECT base, source_question_id FROM (
      SELECT CASE WHEN set_name LIKE '%%\\_ru'
                  THEN left(set_name, length(set_name) - 3) ELSE set_name END AS base,
             source_question_id
      FROM questions WHERE source_question_id IS NOT NULL
    ) q GROUP BY base, source_question_id HAVING count(*) <> 2
  ) d GROUP BY base
) s
"""


# the strongest check this branch has, and it used to look at baseline alone, the one
# variant frozen by definition. Every indexed variant is asked, and by the text of each
# chunk rather than by a count per source: fourteen of the sixteen sources that changed
# under the new parser kept their counts exactly. notes drifts by design, its directory
# is live and the owner writes in it
def every_variant_cuts_into_its_own_rows() -> tuple[bool, str]:
    out = _in_worker("import runpy; runpy.run_path('scripts/cut_digest.py', run_name='__main__')")
    line = next(
        (row for row in reversed(out.splitlines()) if row.startswith("[")), ""
    )
    try:
        report = json.loads(line)
    except ValueError:
        return False, "variants cut into their own rows: unknown"

    bad = []
    for entry in report:
        drift_only = entry["differing"] == ["notes"]
        if entry["sources_differing"] and not drift_only:
            bad.append(
                f"{entry['variant']}: {entry['sources_differing']} sources, "
                f"{entry['files_differing']} files, {entry['chunks_changed']} chunks changed"
            )
    listing = "; ".join(bad) or "all variants reproduce (notes drifts, by design)"
    return not bad, f"variants cut into their own rows: {listing}"


def halves_of_pairs_are_counted() -> tuple[bool, str]:
    out = _in_worker(
        "from orm.sync_db import engine; from sqlalchemy import text;"
        f' print(engine.connect().execute(text("""{PAIRED_SETS}""")).scalar())'
    )
    # descriptive on purpose, hence the name: an original makes a pair, a paraphrase and
    # its translation, and a run that died between them leaves a half. A half with only
    # the russian side is skipped by the generator now rather than paired with a fresh
    # paraphrase, so it stays until someone removes it, and that is not a failure
    return True, f"originals missing half of their pair: {out or 'unknown'}"


def schema_holds_no_variant_indexes() -> tuple[bool, str]:
    dump = ROOT / "db" / "schema.sql"
    if not dump.exists():
        return False, "db/schema.sql is missing"
    leaked = [
        line.replace(" IF NOT EXISTS", "").split()[2]
        for line in dump.read_text().splitlines()
        if vector_index_prefix() in line and line.startswith("CREATE INDEX")
    ]
    # a variant is a line in the config, so its index is built at runtime; one in the
    # dump means the dump has become a function of what happened to be indexed locally
    return not leaked, (
        f"schema.sql carries variant indexes: {leaked}"
        if leaked
        else "schema.sql holds no variant indexes"
    )


def keyword_switches_match_the_worker() -> tuple[bool, str]:
    logs = get("/v1/question-log?limit=1")
    if not logs:
        return True, "keyword switches: no logged run to compare against yet"
    config_row = (logs[0].get("metrics") or {}).get("config", {})
    logged = config_row.get("keyword")
    # an agent answer that never called the corpus tool searched at no depth and records
    # none. That is not a mismatch, so it abstains from this comparison instead of
    # failing it, and the worker is asked without the depth in that case
    searched = logged is not None and config_row.get("ef_search") is not None
    if searched:
        logged = {**logged, "ef_search": config_row.get("ef_search")}
    # the depth is resolved, not declared: a run records the number it searched at, and
    # comparing it against the word `auto` in the config makes this check fail for as
    # long as the config says `auto`. Resolved for the variant the row was taken on: the
    # crossover is a property of the table and the answer differs per variant
    variant = json.dumps(config_row.get("variant"))
    depth = (
        f" 'ef_search': search_depth.resolve({variant})," if searched else ""
    )
    out = _in_worker(
        "import json, config; from use_cases import search_depth;"
        " r = config.settings.retrieval;"
        " print(json.dumps({'query': r.keyword_query, 'rank': r.keyword_rank,"
        " 'norm': r.keyword_norm, 'query_lang': r.query_lang,"
        f"{depth}"
        "}))"
    )
    live = json.loads(out) if out.startswith("{") else None
    if logged is None:
        return True, f"keyword switches: worker {live}, last run predates the field"
    ok = logged == live
    return ok, f"keyword switches: worker {live}, last run {logged}"


# only these decide a verdict; elsewhere an unreachable label is a note, not a stop.
# Asked of the worker rather than written here twice: this list said `paraphrased_ru`
# for a day after every measurement had moved to v2, and nothing complained
@lru_cache(maxsize=1)
def criterion_sets() -> tuple[str, ...]:
    out = _in_worker(
        "import config; print(','.join(config.settings.retrieval.criterion_sets))"
    )
    return tuple(name for name in out.split(",") if name) or ("paraphrased_v2_ru",)


def marks_are_reachable() -> tuple[bool, str]:
    out = _in_worker(
        "import json, config;"
        " from orm.sync_db import engine; from sqlalchemy import text;"
        " sql = text(\"\"\"SELECT q.set_name, count(*) AS unreachable FROM questions q"
        " WHERE array_length(q.marked_sources, 1) > 0 AND NOT EXISTS ("
        "   SELECT 1 FROM data_chunks dc, unnest(q.marked_sources) m"
        "   WHERE dc.variant = :v AND dc.source LIKE '%' || m || '%')"
        " GROUP BY q.set_name ORDER BY 2 DESC\"\"\");"
        " rows = engine.connect().execute(sql, {'v': config.settings.corpus.variant}).all();"
        " print(json.dumps([[r[0], r[1]] for r in rows]))"
    )
    rows = json.loads(out) if out.startswith("[") else None
    if rows is None:
        return False, f"label reachability: cannot read ({out[:60] or 'no answer'})"
    if not rows:
        return True, "label reachability: every marked question can be hit"
    listing = ", ".join(f"{name}={n}" for name, n in rows)
    blocking = [name for name, _ in rows if name in criterion_sets()]
    verdict = "" if not blocking else f"; blocks the criterion sets {blocking}"
    return not blocking, f"questions no chunk can satisfy: {listing}{verdict}"


# a liveness check, not the gate: what the depth is judged by is max_mrr_loss. The floor
# and the question count live in config, next to the value they qualify
def index_is_alive() -> tuple[bool, str]:
    thresholds = _alive_thresholds()
    if thresholds is None:
        return False, "index recall: cannot read the thresholds from the worker"
    floor, asked = thresholds
    out = sh(
        "docker", "compose", "exec", "-T", "worker", "python",
        "/app/scripts/retrieval_report.py", "--set", criterion_sets()[0], "--recall",
        "--limit", str(asked),
    )
    score = out.rsplit(":", 1)[-1].strip()
    try:
        value = float(score)
    except ValueError:
        return False, f"index recall: cannot read ({out[-60:] or 'no answer'})"
    return value >= floor, (
        f"index alive: recall@20 against exact {value} on {asked} questions "
        f"(liveness, floor {floor}; the gate is max_mrr_loss)"
    )


def _alive_thresholds() -> tuple[float, int] | None:
    # sh() returns "" on any non-zero exit, which is what a downed worker looks like:
    # the condition this check exists to report, not one to crash on
    out = _in_worker(
        "import config;"
        " r = config.settings.retrieval;"
        " print(f'{r.index_alive_recall} {r.index_alive_questions}')"
    )
    try:
        floor, asked = out.split()
        return float(floor), int(asked)
    except ValueError:
        return None


CHECKS = (
    tree_is_clean, worker_newer_than_sources, worker_imports, window_matches_config,
    models_are_on_the_card, queue_is_idle, corpus_variant_is_usable,
    every_variant_walks_its_index,
    table_is_vacuumed, schema_holds_no_variant_indexes, one_question_per_original,
    every_variant_cuts_into_its_own_rows,
    keyword_switches_match_the_worker, marks_are_reachable, index_is_alive,
)

# these read something out and never refuse: standing among fifteen things that can fail
# made them look like gates with a permanently green light
NOTES = (halves_of_pairs_are_counted,)


def _rows(run_name: str) -> list:
    rows, offset = [], 0
    while True:
        query = urllib.parse.urlencode({"run_name": run_name, "limit": 1000, "offset": offset})
        page = get(f"/v1/question-log?{query}")
        rows += page
        if len(page) < 1000:
            return rows
        offset += len(page)


def verify_run(spec: str, expect: int | None, shared: set | None) -> int:
    run_name, _, wanted = spec.partition("=")
    logs = _rows(run_name)
    if not logs:
        print(f"{run_name}: no rows yet")
        return 1
    questions = {row["question_id"] for row in logs}
    # the orchestrator and the context window are the agent's to record; a single-shot
    # snapshot has neither, so requiring them made this check unable to pass on the
    # default pipeline
    agent_rows = [row for row in logs if row.get("pipeline") == "agent"]
    agent_snapshots = [(row.get("metrics") or {}).get("config") or {} for row in agent_rows]
    missing = [
        key
        for key in ("orchestrator", "context_length")
        if any(key not in snapshot for snapshot in agent_snapshots)
    ]
    windows = {snapshot.get("context_length") for snapshot in agent_snapshots}
    # an options payload without an orchestrator silently runs the hand-rolled loop
    orchestrators = {
        (snapshot.get("orchestrator") or {}).get("name") for snapshot in agent_snapshots
    }
    broken = [
        row
        for row in logs
        if (row.get("metrics") or {}).get("outcome") == "error"
        or (row.get("metrics") or {}).get("failed")
    ]
    problems = []
    if None in windows:
        problems.append(
            f"{sum(1 for s in agent_snapshots if not s.get('context_length'))} agent rows without"
            " a context window: the server could not be asked"
        )
    if len(broken) > len(logs) // 10:
        problems.append(f"{len(broken)} of {len(logs)} rows are errors")
    if len(logs) != len(questions):
        problems.append(f"{len(logs) - len(questions)} duplicate rows")
    if expect and len(questions) != expect:
        problems.append(f"expected {expect} questions, got {len(questions)}")
    if shared and questions != shared:
        problems.append(f"{len(shared - questions)} questions missing against the other runs")
    if missing:
        problems.append(f"snapshot keys missing: {missing}")
    if len(windows) > 1:
        problems.append(f"more than one context window: {sorted(w for w in windows if w)}")
    if len(orchestrators) > 1:
        problems.append(f"more than one orchestrator: {sorted(o for o in orchestrators if o)}")
    if wanted and orchestrators != {wanted}:
        problems.append(f"orchestrator is {sorted(orchestrators)}, expected {wanted}")
    print(
        f"{run_name}: {len(logs)} rows over {len(questions)} questions, "
        f"orchestrator {sorted(o for o in orchestrators if o)}, "
        f"windows {sorted(w for w in windows if w)}, errors {len(broken)}"
        + ("" if not problems else "\n     FAIL " + "; FAIL ".join(problems))
    )
    return 1 if problems else 0


# what has to be identical across arms, or the runs are not comparable at all
PINNED = (
    "corpus_fingerprint", "variant", "variant_policy", "keyword", "ef_search", "k", "max_hops",
    "fallback_policy", "gate", "topic", "context_length", "rerank", "distance_threshold",
    "mcp_configured", "code_version",
)


def _setting(key: str, snapshot: dict):
    value = snapshot.get(key)
    # the topic block carries this question's score next to the threshold that was configured
    if key == "topic" and isinstance(value, dict):
        return value.get("threshold")
    return value


# a setting changed halfway through a run is invisible if only the first row is read
def _pinned(spec: str) -> dict:
    logs = _rows(spec.partition("=")[0])
    if not logs:
        return {}
    values: dict = {}
    for row in logs:
        config = (row.get("metrics") or {}).get("config") or {}
        gateless = (config.get("orchestrator") or {}).get("name") == "langgraph_idiomatic"
        for key in PINNED:
            # the arm without a gate records none on purpose, that is not a settings mismatch
            if key == "gate" and gateless:
                continue
            # an agent answer that never called the corpus tool searched at no depth, so
            # it records none. Rows that did search must still agree with each other, and
            # a run of both kinds is a normal run, not a settings mismatch
            if key == "ef_search" and config.get("ef_search") is None:
                continue
            values.setdefault(key, set()).add(json.dumps(_setting(key, config), sort_keys=True))
        # which prompts a run used depends on what fired, so compare a version per name
        for group in ("models", "prompts"):
            for name, version in (row.get(group) or {}).items():
                values.setdefault(f"{group}.{name}", set()).add(json.dumps(version))
    return {
        key: (seen.pop() if len(seen) == 1 else f"MIXED {sorted(seen)}")
        for key, seen in values.items()
    }


def compare_runs(specs: list[str]) -> int:
    pinned = {spec.partition("=")[0]: _pinned(spec) for spec in specs}
    names = [name for name, values in pinned.items() if values]
    if len(names) < 2:
        return 0
    problems = 0
    keys = sorted({key for name in names for key in pinned[name]})
    for key in keys:
        # a key only some runs carry is not a mismatch: not every run fires every prompt
        values = {name: pinned[name][key] for name in names if key in pinned[name]}
        if len(set(values.values())) > 1:
            problems += 1
            print(f"     FAIL {key} differs between runs: " + ", ".join(
                f"{name}={value}" for name, value in values.items()
            ))
    print(f"pinned settings identical across {len(names)} runs: {'no' if problems else 'yes'}")
    return 1 if problems else 0


def _verify(runs: list[str], expect: int | None) -> int:
    # every arm answers the same questions, so a short run shows up as a difference between runs
    seen = [{row["question_id"] for row in _rows(spec.partition("=")[0])} for spec in runs]
    shared = set().union(*seen) if len(runs) > 1 else None
    failed = max(verify_run(spec, expect, shared) for spec in runs)
    return max(failed, compare_runs(runs))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verify", action="store_true", help="check runs that already happened, not the stand"
    )
    parser.add_argument("--expect", type=int, help="how many rows each run must have")
    parser.add_argument(
        "runs", nargs="*", metavar="RUN[=ORCHESTRATOR]", help="run names to verify"
    )
    args = parser.parse_args(argv)

    if args.verify:
        if not args.runs:
            parser.error("--verify needs at least one run name")
        return _verify(args.runs, args.expect)
    if args.runs or args.expect is not None:
        parser.error("run names and --expect only make sense with --verify")

    failed = 0
    for check in CHECKS:
        ok, message = check()
        print(f"{'ok  ' if ok else 'FAIL'} {message}")
        failed += not ok
    for note in NOTES:
        _, message = note()
        print(f"note {message}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
