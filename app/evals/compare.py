import statistics
import sys

from evals.loaders import load_logs
from evals.pools import POOLS, has_remote_evidence, outcome, split
from evals.stats import delta_stats

OUTCOMES = ("answered", "refused", "unsupported_answer", "narrated_call", "exhausted", "error")
AXES = ("faithfulness", "relevance", "completeness")
GATE_REASONS = ("empty", "weak", "off_topic")


def _avg(values, digits=2):
    vals = [float(v) for v in values if v is not None]
    return round(statistics.fmean(vals), digits) if vals else None


def summarize(logs) -> dict:
    marks = [outcome(ql) for ql in logs]
    latency = [ql.elapsed for ql in logs if ql.elapsed is not None]
    remote = [ql for ql in logs if has_remote_evidence(ql)]
    home = [
        ql
        for ql, mark in zip(logs, marks, strict=True)
        if mark == "answered" and not has_remote_evidence(ql)
    ]
    # an open gate that answered from the corpus anyway is a failed handoff, not a shut gate
    opened = [ql for ql in home if (ql.metrics or {}).get("fallback_opened")]
    reasons = [(ql.metrics or {}).get("fallback_reason") for ql in logs]
    return {
        "n": len(logs),
        "judged": sum(1 for ql in logs if ql.faithfulness is not None),
        "faithfulness": _avg(ql.faithfulness for ql in logs),
        "relevance": _avg(ql.relevance for ql in logs),
        "completeness": _avg(ql.completeness for ql in logs),
        "answered_via_remote": len(remote),
        "answered_from_corpus": len(home),
        "answered_from_corpus_gate_shut": len(home) - len(opened),
        "answered_from_corpus_opened_no_evidence": len(opened),
        "answered_from_corpus_rate": round(len(home) / len(logs), 3) if logs else None,
        "gate_fired": sum(1 for r in reasons if r in GATE_REASONS),
        "latency_avg": _avg(latency, digits=1),
        "latency_p50": round(statistics.median(latency), 1) if latency else None,
        "outcomes": {o: marks.count(o) for o in OUTCOMES},
    }


def paired(left, right, axis) -> dict:
    by_question = {ql.question_id: ql for ql in left if ql.question_id is not None}
    pairs = []
    for ql in right:
        other = by_question.get(ql.question_id)
        if other is None:
            continue
        a, b = getattr(other, axis), getattr(ql, axis)
        if a is not None and b is not None:
            pairs.append((float(a), float(b)))

    result = {
        "n": len(pairs),
        "left": _avg(a for a, _ in pairs),
        "right": _avg(b for _, b in pairs),
        "better": sum(1 for a, b in pairs if b > a),
        "worse": sum(1 for a, b in pairs if b < a),
        "mean_delta": None,
        "ci95": None,
        "p_value": None,
    }
    if pairs:
        stats = delta_stats([b - a for a, b in pairs])
        result["mean_delta"] = stats["mean_delta"]
        result["ci95"] = stats["ci95"]
        result["p_value"] = stats["p"] if any(a != b for a, b in pairs) else None
    return result


def compare(runs: dict[str, list]) -> dict:
    by_pool = {name: split(logs) for name, logs in runs.items()}
    names = list(runs)

    pools = {}
    for pool in POOLS:
        if pool == "rejected" or not any(by_pool[name][pool] for name in names):
            continue
        pairs = [
            {
                "left": left,
                "right": right,
                **{
                    axis: paired(by_pool[left][pool], by_pool[right][pool], axis)
                    for axis in AXES
                },
            }
            for i, left in enumerate(names)
            for right in names[i + 1 :]
        ]
        pools[pool] = {
            "arms": {name: summarize(by_pool[name][pool]) for name in names},
            "pairs": pairs,
        }

    scored = {
        name: [ql for pool, logs in by_pool[name].items() if pool != "rejected" for ql in logs]
        for name in names
    }
    # pools differ in what they should do, so their blend ranks nothing: kept for latency only
    return {
        "runs": names,
        "pools": pools,
        "blended_do_not_rank": {name: summarize(logs) for name, logs in scored.items()},
    }


def compare_runs(run_names: list[str]) -> dict:
    return compare({name: load_logs(name) for name in run_names})


if __name__ == "__main__":
    print(compare_runs(sys.argv[1:]))
