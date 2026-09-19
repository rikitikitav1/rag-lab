"""Everything this stand offers, in one screen: job types, MCP tools, routes."""

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
    return sorted(found)


def orchestration() -> list[str]:
    from models.experiment import ExperimentKind

    return [
        "the queue runs jobs in id order, so a chain is enqueued at once, never waited on",
        f"`experiment` orchestrates arms and aggregates them: kinds {[k.value for k in ExperimentKind]}",
        "`try_aggregate_for_run` fires the next step when a run's judging finishes",
        "a report that needs no model is a script, not a job: it does not compete for the card",
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--routes", action="store_true", help="list all routes, not the count")
    args = parser.parse_args()

    print("JOB TYPES")
    print("  " + ", ".join(job_types()))
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
        print("  " + ", ".join(sorted({p.split('/')[2] for p, _ in found})))
        print("  (--routes for all of them)")


if __name__ == "__main__":
    main()
