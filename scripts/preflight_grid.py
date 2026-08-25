import json
import re
import subprocess
import sys
import urllib.request

API = "http://localhost:8000"


def sh(*args: str) -> str:
    return subprocess.run(args, capture_output=True, text=True).stdout.strip()


def get(path: str):
    with urllib.request.urlopen(f"{API}{path}") as response:
        return json.load(response)


# docker reports UTC with a Z and nanoseconds, git reports an offset
def _moment(stamp: str):
    from datetime import datetime

    body, offset, _ = re.split(r"([+-]\d\d:\d\d|Z)$", stamp.strip(), maxsplit=1)[:3]
    if "." in body:
        whole, _, fraction = body.partition(".")
        body = f"{whole}.{fraction[:6]}"
    return datetime.fromisoformat(f"{body}{'+00:00' if offset in ('Z', '') else offset}")


def worker_newer_than_head() -> tuple[bool, str]:
    started = sh("docker", "inspect", "-f", "{{.State.StartedAt}}", "rag-lab-worker-1")
    head = sh("git", "log", "-1", "--format=%cI")
    if not started or not head:
        return False, "cannot read the worker start time or the commit time"
    ok = _moment(started) > _moment(head)
    return ok, f"worker started {started}, HEAD committed {head}"


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
    return live == configured, f"context window: config {configured}, server {live}"


def queue_is_idle() -> tuple[bool, str]:
    jobs = [j for j in get("/v1/job?limit=20") if j["status"] in ("new", "running")]
    return not jobs, f"{len(jobs)} jobs still queued or running"


CHECKS = (worker_newer_than_head, worker_imports, window_matches_config, queue_is_idle)


def verify_run(run_name: str) -> int:
    logs = get(f"/v1/question-log?run_name={run_name}&limit=1000")
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
    ok = len(logs) == len(questions) and not missing and len(windows) == 1
    print(
        f"{run_name}: {len(logs)} rows over {len(questions)} questions, "
        f"windows {sorted(w for w in windows if w)}, missing keys {missing or 'none'}"
    )
    return 0 if ok else 1


def main(argv: list[str]) -> int:
    if argv and argv[0] == "--verify":
        return max(verify_run(name) for name in argv[1:])
    failed = 0
    for check in CHECKS:
        ok, message = check()
        print(f"{'ok  ' if ok else 'FAIL'} {message}")
        failed += not ok
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
