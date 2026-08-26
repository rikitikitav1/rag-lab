import json
import re
import subprocess
import sys
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


def queue_is_idle() -> tuple[bool, str]:
    jobs = get("/v1/job?status=new&status=running&limit=1000")
    return not jobs, f"{len(jobs)} jobs still queued or running"


CHECKS = (
    tree_is_clean, worker_newer_than_sources, worker_imports, window_matches_config, queue_is_idle,
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
        f"windows {sorted(w for w in windows if w)}"
        + ("" if not problems else "\n     FAIL " + "; FAIL ".join(problems))
    )
    return 1 if problems else 0


# what has to be identical across arms, or the runs are not comparable at all
PINNED = (
    "corpus_fingerprint", "k", "max_hops", "fallback_policy", "gate", "topic",
    "context_length", "rerank", "distance_threshold", "mcp_configured", "code_version",
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


def _verify(argv: list[str]) -> int:
    expect = None
    if "--expect" in argv:
        at = argv.index("--expect")
        expect = int(argv[at + 1])
        argv = argv[:at] + argv[at + 2 :]
    if not argv:
        print("usage: preflight_grid.py --verify [--expect N] <run>[=orchestrator] ...")
        return 1
    # every arm answers the same questions, so a short run shows up as a difference between runs
    seen = [{row["question_id"] for row in _rows(spec.partition("=")[0])} for spec in argv]
    shared = set().union(*seen) if len(argv) > 1 else None
    failed = max(verify_run(spec, expect, shared) for spec in argv)
    return max(failed, compare_runs(argv))


def main(argv: list[str]) -> int:
    if argv and argv[0] == "--verify":
        return _verify(argv[1:])
    failed = 0
    for check in CHECKS:
        ok, message = check()
        print(f"{'ok  ' if ok else 'FAIL'} {message}")
        failed += not ok
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
