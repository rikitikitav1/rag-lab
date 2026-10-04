"""Everything this stand offers in one screen: job types, MCP tools, routes; `uv run python scripts/surface.py`."""

import argparse
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def job_types() -> list[str]:
    from job_handlers.base import HANDLERS

    return sorted(HANDLERS)


# read from source, like the routes: no running server, and no fastmcp version to keep up with
def mcp_tools() -> dict[str, list[str]]:
    out = {}
    for label, module in (("rag-lab-ops", "mcp_ops"), ("rag-lab", "mcp_server")):
        source = (ROOT / "app" / f"{module}.py").read_text()
        out[label] = sorted(re.findall(r'\.tool\(\s*\n?\s*name="([^"]+)"', source))
    return out


def routes() -> list[tuple[str, str]]:
    found = []
    for path in sorted((ROOT / "app" / "api" / "v1").glob("*.py")):
        source = path.read_text()
        prefix = re.search(r'APIRouter\(prefix="([^"]*)"', source)
        prefix = prefix.group(1) if prefix else ""
        for method, route in re.findall(
            r'@router\.(get|post|put|delete|patch)\("([^"]*)"', source
        ):
            found.append((f"/v1{prefix}{route}", method.upper()))
    # the health doors sit outside v1: probes at the root and the stand read under /v1/health
    health = (ROOT / "app" / "api" / "health.py").read_text()
    for router, method, route in re.findall(r'@(router|v1)\.(get|post)\("([^"]*)"', health):
        found.append(((f"/v1/health{route}" if router == "v1" else route), method.upper()))
    return sorted(found)


def job_options() -> dict[str, list[str]]:
    from job_specs import SPECS

    return {name: sorted(spec.model_fields) for name, spec in sorted(SPECS.items())}


def orchestration() -> list[str]:
    from models.experiment import ExperimentKind

    return [
        "the queue runs jobs in id order, so a chain is enqueued at once, never waited on",
        f"`experiment` orchestrates arms and aggregates them: kinds {[k.value for k in ExperimentKind]}",
        "`try_aggregate_for_run` fires the next step when a run's judging finishes",
        "a report that needs no model is a script, not a job: it does not compete for the card",
        "wait for jobs with `scripts/wait_jobs.py <id...>` or `--line` (whole queue) in the worker:"
        " it listens, it does not poll",
        "restart the worker with `scripts/restart_worker.sh [--then-queue jobs.jsonl]`: it holds the line, waits out"
        " what runs, releases it and queues what you give it; never a hand-written waiter after it",
        "hold, release and cancel in bulk with POST /v1/job/pause|resume|cancel or MCP pause_jobs, resume_jobs,"
        " cancel_jobs; dry_run=true names the ids first",
        "a job queued by a handler names it in parent_id: list_jobs(parent_id=...) finds a job's children",
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--routes", action="store_true", help="list all routes, not the count")
    parser.add_argument("--options", action="store_true", help="list each job type's options")
    args = parser.parse_args()

    print("JOB TYPES")
    print("  " + ", ".join(job_types()))
    if args.options:
        for name, fields in job_options().items():
            print(f"  {name}: {', '.join(fields) or '-'}")
    print("\nMCP TOOLS")
    for server, tools in mcp_tools().items():
        print(f"  {server}: " + ", ".join(tools))
    print("\nORCHESTRATION, before you write a waiter")
    for line in orchestration():
        print(f"  - {line}")
    found = routes()
    print(f"\nROUTES ({len(found)})")
    if args.routes:
        for path, method in found:
            print(f"  {method:6} {path}")
    else:
        print("  " + ", ".join(sorted({p.split('/')[2] if p.startswith("/v1/") else p.strip("/") for p, _ in found})))
        print("  (--routes for all of them)")


if __name__ == "__main__":
    main()
