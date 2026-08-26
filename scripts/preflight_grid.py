import argparse
import json
import re
import subprocess
import urllib.parse
import urllib.request
from pathlib import Path

API = "http://localhost:8000"
ROOT = Path(__file__).resolve().parent.parent


def sh(*args: str) -> str:
    done = subprocess.run(args, capture_output=True, text=True)
    return done.stdout.strip() if done.returncode == 0 else ""


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
        " from orchestrators import graph, middleware, react;"
        " from use_cases import agent; print('ok')",
    )
    return out.endswith("ok"), f"imports inside the worker: {out or 'failed'}"


def window_matches_config() -> tuple[bool, str]:
    configured = sh(
        "docker", "compose", "exec", "-T", "rag-lab", "python", "-c",
        "import sys; sys.path.insert(0, '/app/app'); import config;"
        " print(config.settings.llm.context_length)",
    )
    live = sh(
        "docker", "compose", "exec", "-T", "rag-lab", "python", "-c",
        "import sys; sys.path.insert(0, '/app/app'); import llm;"
        " print(llm.server_context_length(llm.resolve_name('generation')))",
    )
    # the server is the authority: a stray env var in a running container beat the config once
    ok = bool(configured) and live == configured
    hint = "" if live else " (the generator is not loaded, ask it one question first)"
    return ok, f"context window: config {configured or 'unknown'}, server {live or 'unknown'}{hint}"


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



def _in_worker(code: str) -> str:
    return sh(
        "docker", "compose", "exec", "-T", "worker", "python", "-c",
        "import sys; sys.path.insert(0, '/app/app');" + code,
    )


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
def variant_index_is_used() -> tuple[bool, str]:
    out = _in_worker(
        "import config;"
        " from orm.sync_db import engine; from sqlalchemy import text;"
        " v = config.settings.corpus.variant;"
        " q = 'SELECT id FROM data_chunks WHERE variant = :v AND embedding IS NOT NULL"
        " ORDER BY embedding <=> (SELECT embedding FROM data_chunks WHERE variant = :v LIMIT 1)"
        " LIMIT 20';"
        " c = engine.connect();"
        " [c.execute(text(q), {'v': v}).all() for _ in range(6)];"
        " plan = ' '.join(r[0] for r in c.execute(text('EXPLAIN (COSTS OFF) ' + q), {'v': v}));"
        " print(plan)"
    )
    wanted = "data_chunks_embedding_"
    ok = wanted in out and "Index Scan" in out
    return ok, f"vector index in the plan: {'partial, per variant' if ok else out[:90] or 'none'}"


def table_is_vacuumed() -> tuple[bool, str]:
    out = _in_worker(
        "from orm.sync_db import engine; from sqlalchemy import text;"
        " print(engine.connect().execute(text(\"SELECT n_dead_tup FROM pg_stat_user_tables"
        " WHERE relname = 'data_chunks'\")).scalar())"
    )
    dead = int(out) if out.isdigit() else -1
    # a mass update leaves dead entries in the hnsw graph, and retrieval shrinks quietly
    return 0 <= dead <= 1000, f"dead tuples in data_chunks: {out or 'unknown'}"


def keyword_switches_match_the_worker() -> tuple[bool, str]:
    logs = get("/v1/question-log?limit=1")
    if not logs:
        return True, "keyword switches: no logged run to compare against yet"
    config_row = (logs[0].get("metrics") or {}).get("config", {})
    logged = config_row.get("keyword")
    if logged is not None:
        logged = {**logged, "ef_search": config_row.get("ef_search")}
    out = _in_worker(
        "import json, config;"
        " r = config.settings.retrieval;"
        " print(json.dumps({'query': r.keyword_query, 'rank': r.keyword_rank,"
        " 'norm': r.keyword_norm, 'query_lang': r.query_lang, 'ef_search': r.ef_search}))"
    )
    live = json.loads(out) if out.startswith("{") else None
    if logged is None:
        return True, f"keyword switches: worker {live}, last run predates the field"
    ok = logged == live
    return ok, f"keyword switches: worker {live}, last run {logged}"


# only these decide a verdict; elsewhere an unreachable label is a note, not a stop
CRITERION_SETS = ("paraphrased_ru", "paraphrased")


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
    blocking = [name for name, _ in rows if name in CRITERION_SETS]
    verdict = "" if not blocking else f"; blocks the criterion sets {blocking}"
    return not blocking, f"questions no chunk can satisfy: {listing}{verdict}"


# the agent path goes through hnsw, so a variant whose index lost recall would read as bad chunking
def index_recall_is_intact() -> tuple[bool, str]:
    out = sh(
        "docker", "compose", "exec", "-T", "worker", "python",
        "/app/scripts/retrieval_report.py", "--set", "paraphrased_ru", "--recall",
        "--limit", "40",
    )
    score = out.rsplit(":", 1)[-1].strip()
    try:
        value = float(score)
    except ValueError:
        return False, f"index recall: cannot read ({out[-60:] or 'no answer'})"
    return value >= 0.98, f"hnsw recall@20 against exact search: {value}"


CHECKS = (
    tree_is_clean, worker_newer_than_sources, worker_imports, window_matches_config,
    models_are_on_the_card, queue_is_idle, corpus_variant_is_usable, variant_index_is_used,
    table_is_vacuumed, keyword_switches_match_the_worker, marks_are_reachable,
    index_recall_is_intact,
)


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
    snapshots = [(row.get("metrics") or {}).get("config") or {} for row in logs]
    missing = [
        key
        for key in ("orchestrator", "context_length")
        if any(key not in snapshot for snapshot in snapshots)
    ]
    windows = {snapshot.get("context_length") for snapshot in snapshots}
    # an options payload without an orchestrator silently runs the hand-rolled loop
    orchestrators = {(snapshot.get("orchestrator") or {}).get("name") for snapshot in snapshots}
    broken = [
        row
        for row in logs
        if (row.get("metrics") or {}).get("outcome") == "error"
        or (row.get("metrics") or {}).get("failed")
    ]
    problems = []
    if None in windows:
        problems.append(f"{sum(1 for s in snapshots if not s.get('context_length'))} rows without a"
                        " context window: the server could not be asked")
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
    if len(windows) != 1:
        problems.append(f"more than one context window: {sorted(w for w in windows if w)}")
    if len(orchestrators) != 1:
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
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
