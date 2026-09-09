import statistics
import sys

import limits
from evals.judge_correlation import JOINS_BOTH_JUDGES, joins_both_judges
from evals.loaders import load_logs
from evals.pools import (
    ALL_OUTCOMES,
    POOLS,
    has_remote_evidence,
    outcome,
    split,
)
from evals.stats import delta_stats, deltas_over, mean_of, tally
from use_cases import rejudge

OUTCOMES = ALL_OUTCOMES
AXES = rejudge.AXES
GATE_REASONS = ("empty", "weak", "off_topic")


# both doors onto `compare` walk every run named, so the caps belong here and not at one door
def named_runs(run_names: list[str]) -> list[str]:
    named = list(dict.fromkeys(run_names))
    if not named:
        raise ValueError("run_names must not be empty")
    if len(named) > limits.MAX_RUNS:
        raise ValueError(f"{len(named)} runs is over the cap of {limits.MAX_RUNS}")
    too_long = [n for n in named if len(n) > limits.MAX_RUN_NAME]
    if too_long:
        raise ValueError(f"run names over {limits.MAX_RUN_NAME} characters: {too_long[:3]}")
    return named


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
        "faithfulness": mean_of(ql.faithfulness for ql in logs),
        "relevance": mean_of(ql.relevance for ql in logs),
        "completeness": mean_of(ql.completeness for ql in logs),
        "answered_via_remote": len(remote),
        "answered_from_corpus": len(home),
        "answered_from_corpus_gate_shut": len(home) - len(opened),
        "answered_from_corpus_opened_no_evidence": len(opened),
        "answered_from_corpus_rate": round(len(home) / len(logs), 3) if logs else None,
        "gate_fired": sum(1 for r in reasons if r in GATE_REASONS),
        "latency_avg": mean_of(latency, digits=1),
        "latency_p50": round(statistics.median(latency), 1) if latency else None,
        "outcomes": {o: marks.count(o) for o in OUTCOMES},
    }


def _client(logs) -> str | None:
    for ql in logs:
        orchestrator = ((ql.metrics or {}).get("config") or {}).get("orchestrator") or {}
        if orchestrator.get("client"):
            return orchestrator["client"]
    return None


def paired(left, right, axis) -> dict:
    def scores(logs):
        return {
            ql.question_id: None if getattr(ql, axis) is None else float(getattr(ql, axis))
            for ql in logs
            if ql.question_id is not None
        }

    before, after = scores(left), scores(right)
    # in the order `right` arrives, which is the order the bootstrap was drawn over
    ids = [ql.question_id for ql in right if ql.question_id in before]
    kept = [i for i in ids if before[i] is not None and after[i] is not None]
    deltas = deltas_over(before, after, ids)

    counted = tally(deltas)
    result = {
        "n": len(deltas),
        "left": mean_of(before[i] for i in kept),
        "right": mean_of(after[i] for i in kept),
        "better": counted["better"],
        "worse": counted["worse"],
        "mean_delta": None,
        "ci95": None,
        "p_value": None,
    }
    if deltas:
        stats = delta_stats(deltas)
        result["mean_delta"] = stats["mean_delta"]
        result["ci95"] = stats["ci95"]
        result["p_value"] = stats["p"] if any(d != 0 for d in deltas) else None
    return result


# 1 pools and blend; 2 the residency block; 3 the engine and the correlation's own population
SCHEMA = 3


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
                # a pair that also swaps the model client measures two changes, not one
                "isolates_orchestrator": _client(by_pool[left][pool])
                == _client(by_pool[right][pool]),
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
        "schema": SCHEMA,
        "runs": names,
        "residency": residencies(runs),
        # the correlation's own predicate, called not restated: one label stood over two selections
        "correlation_population": {
            "predicate": JOINS_BOTH_JUDGES,
            "n": {name: sum(1 for ql in logs if joins_both_judges(ql))
                  for name, logs in runs.items()},
        },
        "pools": pools,
        "blended_do_not_rank": {name: summarize(logs) for name, logs in scored.items()},
    }


# an engine mismatch outranks a residency one: two backends are not one instrument at all
def _what_to_read_first(one_engine: bool | None, one: bool | None) -> str | None:
    if one_engine is False:
        return (
            "arms judged on different engines are not comparable: batching, kernels and "
            "quantisation all differ, and residency does not even mean the same thing on both"
        )
    if one is False:
        return (
            "arms judged across a reload are not comparable directly: the same judge moves 14% of "
            "its scores and 58% of its reason texts on byte-identical input"
        )
    if one is None:
        return "rows judged before this was recorded carry no residency, so nothing can be said"
    if one_engine is None:
        return (
            "the residency matches, but at least one arm recorded no engine, so whether both ran "
            "on one backend cannot be said, and that outranks any reading of the residency"
        )
    return (
        "one residency is necessary, not sufficient: two arms with identical rows, order and "
        "prompt still differed on 4 of 50 rows, so this contrast measures its own floor rather "
        "than inheriting a zero. A reading that rests on the reason text holds only here, and "
        "`seed` at temperature zero says the sampler took no part, not that a pass repeats"
    )


# None where any arm is silent: an empty set used to drop out and read as agreement
def _all_agree(by_run: dict) -> bool | None:
    if not by_run or any(not seen for seen in by_run.values()):
        return None
    return len({one for seen in by_run.values() for one in seen}) == 1


# two arms judged across a reload are two instruments: 14% of scores move on identical input
def residencies(runs: dict[str, list]) -> dict:
    from use_cases import rejudge

    seen, engines_seen = {}, {}
    for name, logs in runs.items():
        ids, engines = set(), set()
        for ql in logs:
            for axis in rejudge.AXES:
                stamp = ((ql.metrics or {}).get(axis) or {})
                if stamp.get("residency_id") is not None:
                    ids.add(stamp["residency_id"])
                if stamp.get("engine"):
                    engines.add(stamp["engine"])
        seen[name], engines_seen[name] = sorted(ids), sorted(engines)
    # an arm that recorded nothing cannot agree with one that did: silence is not a match
    one = _all_agree(seen)
    one_engine = _all_agree(engines_seen)
    return {
        "by_run": seen,
        "one_residency": one,
        "engines_by_run": engines_seen,
        "one_engine": one_engine,
        "read_this_first": _what_to_read_first(one_engine, one),
    }


def compare_runs(run_names: list[str]) -> dict:
    return compare({name: load_logs(name) for name in run_names})


if __name__ == "__main__":
    print(compare_runs(sys.argv[1:]))
