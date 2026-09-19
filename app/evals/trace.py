from collections import Counter

SCHEMA = 1

READS = (
    "hops counts the rows by the last hop their trace reached; nodes counts steps and the rows that"
    " reached each node; verdicts counts what the gate said per hop of retrieval; outcomes pairs the"
    " row's outcome with the hops it spent, so a run that answered on hop one is not read as one that"
    " spent three; rows without a trace are named, not counted as zero"
)


def report(rows: list) -> dict:
    hops, nodes, steps, verdicts, outcomes_by_hop = Counter(), Counter(), Counter(), Counter(), {}
    failed_hops, opened, dropped, announced, no_trace = 0, 0, 0, 0, []
    for row in rows:
        metrics = getattr(row, "metrics", None) or {}
        trace = metrics.get("trace")
        if not trace:
            no_trace.append(row.id)
            continue
        last = max((step.get("hop") or 0) for step in trace)
        hops[last] += 1
        outcome = str(metrics.get("outcome") or "unknown")
        outcomes_by_hop.setdefault(last, Counter())[outcome] += 1
        for node in {step.get("node") for step in trace}:
            nodes[node] += 1
        for step in trace:
            steps[step.get("node")] += 1
            if step.get("node") == "model" and step.get("failed"):
                failed_hops += 1
            if step.get("node") == "retrieve":
                verdicts[str(step.get("verdict"))] += 1
            if step.get("node") == "fallback":
                opened += bool(step.get("opened"))
                dropped += bool(step.get("dropped"))
                announced += bool(step.get("announced"))
    return {
        "schema": SCHEMA,
        "rows": len(rows),
        "rows_traced": len(rows) - len(no_trace),
        "rows_without_trace": no_trace[:20],
        "hops": dict(sorted(hops.items())),
        "nodes": {node: {"rows": nodes[node], "steps": steps[node]} for node in sorted(steps)},
        "verdicts": dict(verdicts.most_common()),
        "fallback": {"opened": opened, "dropped_context": dropped, "announced": announced},
        "failed_hops": failed_hops,
        "outcomes_by_hops": {
            hop: dict(counts.most_common()) for hop, counts in sorted(outcomes_by_hop.items())
        },
        "reads": READS,
    }


def of_run(run_name: str) -> dict:
    from evals.loaders import load_logs

    return {"run_name": run_name, **report(load_logs(run_name))}
