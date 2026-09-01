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


# asked of the worker: this script runs on the host, without the app's dependencies
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
        "from orchestrators import graph, react;"
        " from use_cases import agent; print('ok')",
    )
    return out.endswith("ok"), f"imports inside the worker: {out or 'failed'}"


def window_matches_config() -> tuple[bool, str]:
    configured = sh(
        "docker", "compose", "exec", "-T", "rag-lab", "python", "-c",
        "import config;"
        " print(config.settings.llm.context_length)",
    )
    # whichever generator is loaded: an override left the configured name unloaded
    out = _in_worker(
        "import json, llm;"
        " print(json.dumps({'loaded': [e['model'] for e in llm.residency()],"
        " 'asked': llm.window_model()}))"
    )
    if not out.startswith("{"):
        return False, f"context window: cannot read the residency ({out[:40] or 'no answer'})"
    state = json.loads(out)
    # asked of the worker: the route and the record read the same rule from the same holder
    asked = state["asked"]
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
        "import json, llm;"
        " roles = {r: llm.resolve_name(r) for r in"
        " ('generation', 'embedding', 'judging', 'paraphrasing')};"
        " loaded = llm.residency();"
        " print(json.dumps({'roles': roles, 'loaded': loaded,"
        " 'off_the_card': [e['model'] for e in loaded if llm.off_the_card(e)]}))",
    )
    if not out.startswith("{"):
        return False, f"residency: cannot read ({out[:60] or 'no answer'})"
    state = json.loads(out)
    loaded = {e["model"]: e for e in state["loaded"]}
    if not loaded:
        # the card is what a run about to load four models will be given
        return False, (
            "no model is loaded: ask one to load before reading the window"
            f"; {_card()}"
        )
    # naming the role matters: a job runs one model, and the others being resident proves nothing
    lines = []
    for role, name in state["roles"].items():
        entry = loaded.get(name)
        where = f"{entry['vram_mb']}/{entry['size_mb']} MiB" if entry else "not resident"
        lines.append(f"{role}={name} {where}")
    off = state["off_the_card"]
    return not off, (
        "; ".join(lines) + (f"; ON CPU: {', '.join(off)}" if off else "") + f"; {_card()}"
    )


# the predicate of `stand_health.drifting_roles`, spelled out because this may not import
def role_drift(declared: dict, served: dict) -> list[str]:
    return [
        f"{role}: config says {name}, the stand serves {served.get(role, 'nothing')}"
        for role, name in sorted(declared.items())
        if served.get(role) != name
    ] + [
        f"{role}: the stand serves {name}, the config declares no such role"
        for role, name in sorted(served.items())
        if role not in declared
    ]


# the file declares a role's model and the database serves it, and the two drift in silence
def roles_match_the_config() -> tuple[bool, str]:
    out = _in_worker(
        "import json, config; print(json.dumps("
        "{r: c.model for r, c in config.settings.llm.roles.items()}))"
    )
    if not out.startswith("{"):
        return False, f"roles: cannot read the config ({out[:60] or 'no answer'})"
    declared = json.loads(out)
    models = {m["id"]: m["name"] for m in get("/v1/model?limit=200")}
    served = {r["role"]: models.get(r["model_id"], f"model {r['model_id']}") for r in get("/v1/role")}
    drift = role_drift(declared, served)
    if drift:
        return False, "; ".join(drift) + ". PUT /v1/role to change it, or edit the file to match"
    return True, "roles: " + ", ".join(f"{r}={n}" for r, n in sorted(served.items()))


# only what the driver says: a `docker compose exec` probe never loaded the reranker
def _card() -> str:
    out = sh(
        "docker", "compose", "exec", "-T", "rag-lab", "python", "-c",
        "import json, gpu;"
        " free, total = gpu.memory_mb() or (0, 0);"
        " print(json.dumps({'free': free, 'total': total}))",
    )
    line = [row for row in out.splitlines() if row.startswith("{")]
    if not line:
        return "card: cannot read"
    seen = json.loads(line[-1])
    if not seen["total"]:
        return "card: no cuda device visible"
    return (
        f"card free {seen['free']} of {seen['total']} MiB"
        " (the reranker's own place on it is checked by the run, not from here)"
    )


def queue_is_idle() -> tuple[bool, str]:
    jobs = get("/v1/job?status=new&status=running&limit=1000")
    return not jobs, f"{len(jobs)} jobs still queued or running"


# the answer is the last line: anything touching the application may log on the way
def _in_worker(code: str) -> str:
    out = sh("docker", "compose", "exec", "-T", "worker", "python", "-c", code)
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


# every indexed variant: indexing one moves where every other stops walking its index
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


# a comment claiming provenance cannot be checked; a `# tuned: file=` line can be
TUNED = re.compile(r"^\s*#\s*tuned:\s*file=(\S+)\s*$")


def _tuned_files() -> list[str]:
    text = (ROOT / "config.yaml").read_text().splitlines()
    return sorted({m.group(1) for line in text if (m := TUNED.match(line))})


@lru_cache(maxsize=1)
def _live_fingerprints() -> dict:
    out = _in_worker(
        "import json, db;"
        " print(json.dumps({v['variant']: db.fingerprint_or_none(variant=v['variant'])"
        " for v in db.corpus_variants()}))"
    )
    return json.loads(out) if out.startswith("{") else {}


def tuned_numbers_still_describe_the_corpus() -> tuple[bool, str]:
    missing, stale, unchecked, compared = [], [], [], 0
    live = _live_fingerprints()
    for name in _tuned_files():
        path = ROOT / name
        if not path.exists():
            missing.append(name)
            continue
        # a csv carries no fingerprint, and counting it as checked said more than was read
        if path.suffix != ".json":
            unchecked.append(Path(name).name)
            continue
        try:
            record = json.loads(path.read_text())
        except ValueError:
            missing.append(f"{name} (not readable)")
            continue
        taken = record.get("fingerprint") if isinstance(record, dict) else None
        variant = (taken or {}).get("variant") or (record or {}).get("variant")
        # a variant no longer indexed is a record of something deleted, not a stale reading
        if not taken or variant not in live:
            unchecked.append(Path(name).name)
            continue
        compared += 1
        if taken != live[variant]:
            stale.append(
                f"{Path(name).name} ({taken.get('chunks')} chunks against {live[variant].get('chunks')})"
            )
    parts = []
    if missing:
        parts.append(f"MISSING {missing}")
    if stale:
        parts.append(f"STALE {stale}")
    if unchecked:
        parts.append(f"{len(unchecked)} carry no comparable fingerprint")
    return not (missing or stale), (
        f"tuned numbers: {compared} of {len(_tuned_files())} files compared"
        + (f"; {'; '.join(parts)}" if parts else ", all still describe this corpus")
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
    # a requeued job paraphrases the same original twice, with a new hash nothing catches
    return out == "none", f"originals paraphrased more than once: {out or 'unknown'}"


# a set that makes one row per original, not a paraphrase and its translation
UNPAIRED_SETS = ("veto",)

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


# by the text of each chunk: fourteen of sixteen changed sources kept their counts
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
    # descriptive on purpose: a set with one row per original is not half of anything
    listed = [
        part for part in (out or "").split(", ")
        if part and not part.split("=")[0].startswith(UNPAIRED_SETS)
    ]
    return True, (
        "originals missing half of their pair: "
        + (", ".join(listed) if listed else "none")
        + f" (sets with one row per original are not counted: {', '.join(UNPAIRED_SETS)})"
    )


def schema_holds_no_variant_indexes() -> tuple[bool, str]:
    dump = ROOT / "db" / "schema.sql"
    if not dump.exists():
        return False, "db/schema.sql is missing"
    leaked = [
        line.replace(" IF NOT EXISTS", "").split()[2]
        for line in dump.read_text().splitlines()
        if vector_index_prefix() in line and line.startswith("CREATE INDEX")
    ]
    # one in the dump means the dump became a function of what was indexed locally
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
    # an agent answer that never searched records no depth, so it abstains here
    searched = logged is not None and config_row.get("ef_search") is not None
    if searched:
        logged = {**logged, "ef_search": config_row.get("ef_search")}
    # resolved, not declared, and per variant: the crossover is a property of the table
    variant = json.dumps(config_row.get("variant"))
    depth = (
        f" 'ef_search': search_depth.resolve({variant})," if searched else ""
    )
    out = _in_worker(
        "import json, config; from use_cases import search_depth;"
        " print(json.dumps({**config.keyword_switches(),"
        f"{depth}"
        "}))"
    )
    live = json.loads(out) if out.startswith("{") else None
    if logged is None:
        return True, f"keyword switches: worker {live}, last run predates the field"
    ok = logged == live
    return ok, f"keyword switches: worker {live}, last run {logged}"


# asked of the worker rather than written twice; `marks_are_reachable` blocks on these sets
@lru_cache(maxsize=1)
def criterion_sets() -> tuple[str, ...]:
    out = _in_worker(
        "import config; print(','.join(config.settings.retrieval.criterion_sets))"
    )
    return tuple(name for name in out.split(",") if name) or ("paraphrased_v2_ru",)


# a veto set can only veto, but an unreachable label in one still reads as a regression
@lru_cache(maxsize=1)
def veto_sets() -> tuple[str, ...]:
    out = _in_worker("import config; print(','.join(config.settings.retrieval.veto_sets))")
    return tuple(name for name in out.split(",") if name)


def marks_are_reachable() -> tuple[bool, str]:
    # `db.live_rows`: this asked only for the variant, so a deactivated file counted
    out = _in_worker(
        "import json, config, db;"
        " from orm.sync_db import engine; from sqlalchemy import text;"
        " sql = text(\"SELECT q.set_name, count(*) AS unreachable FROM questions q\""
        " \" WHERE array_length(q.marked_sources, 1) > 0 AND NOT EXISTS (\""
        " \"   SELECT 1 FROM data_chunks dc, unnest(q.marked_sources) m\""
        " f\"   WHERE {db.live_rows('dc')} AND dc.source LIKE '%' || m || '%')\""
        " \" GROUP BY q.set_name ORDER BY 2 DESC\");"
        " rows = engine.connect().execute("
        "   sql, {'variant': config.settings.corpus.variant}).all();"
        " print(json.dumps([[r[0], r[1]] for r in rows]))"
    )
    rows = json.loads(out) if out.startswith("[") else None
    if rows is None:
        return False, f"label reachability: cannot read ({out[:60] or 'no answer'})"
    if not rows:
        return True, "label reachability: every marked question can be hit"
    listing = ", ".join(f"{name}={n}" for name, n in rows)
    decisive = criterion_sets() + veto_sets()
    blocking = [name for name, _ in rows if name in decisive]
    verdict = "" if not blocking else f"; blocks the sets a verdict is read on {blocking}"
    return not blocking, f"questions no chunk can satisfy: {listing}{verdict}"


# a liveness check, not the gate: the depth is judged by max_mrr_loss
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
    # sh() returns "" on any non-zero exit, which is what a downed worker looks like
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
    models_are_on_the_card, roles_match_the_config, queue_is_idle, corpus_variant_is_usable,
    every_variant_walks_its_index, tuned_numbers_still_describe_the_corpus,
    table_is_vacuumed, schema_holds_no_variant_indexes, one_question_per_original,
    every_variant_cuts_into_its_own_rows,
    keyword_switches_match_the_worker, marks_are_reachable, index_is_alive,
)

# these read something out and never refuse, so they stood as permanently green gates
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


# checked on what a run asked for: ollama truncates and answers from what is left
def _prompts_over_the_window(logs: list) -> list:
    # the generation role, not the max: the judge never shares a window with these rows
    out = _in_worker(
        "import config; print(f'{config.settings.llm.context_length}"
        " {config.settings.llm.roles[\'generation\'].options.get(\'max_tokens\', 0)}')"
    )
    try:
        window, reserved = (int(part) for part in out.split())
    except ValueError:
        # a down worker and an empty roles map both land here, and "no rows over" is a pass
        return [f"cannot read the window from the worker ({out[:40] or 'no answer'})"]
    room = window - reserved
    return sorted(
        (row["prompt_tokens"] for row in logs if (row.get("prompt_tokens") or 0) > room),
        reverse=True,
    )


def verify_run(spec: str, expect: int | None, shared: set | None) -> int:
    run_name, _, wanted = spec.partition("=")
    logs = _rows(run_name)
    if not logs:
        print(f"{run_name}: no rows yet")
        return 1
    questions = {row["question_id"] for row in logs}
    # a single-shot snapshot has neither, so requiring them made this unable to pass
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
    over = _prompts_over_the_window(logs)
    if over and isinstance(over[0], str):
        problems.append(over[0])
    elif over:
        problems.append(
            f"{len(over)} rows asked for more prompt than the window leaves"
            f" (worst {max(over)} tokens): ollama truncates those without saying so"
        )
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


# a key a row does not record is not a value, so it joins no set of values
_ABSENT = object()


def _setting(key: str, snapshot: dict):
    # a row that never carried this key had its `None` join the set as a value
    if key not in snapshot:
        return _ABSENT
    value = snapshot.get(key)
    # what two arms have to share is the policy, not the number it produced
    if key == "topic" and isinstance(value, dict):
        # a row written before the policy was recorded is skipped like any absent key
        if "policy" not in value:
            return _ABSENT
        return json.dumps(value["policy"], sort_keys=True)
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
            # an agent answer that never searched records no depth: not a settings mismatch
            if key == "ef_search" and config.get("ef_search") is None:
                continue
            seen = _setting(key, config)
            if seen is _ABSENT:
                continue
            values.setdefault(key, set()).add(json.dumps(seen, sort_keys=True))
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
