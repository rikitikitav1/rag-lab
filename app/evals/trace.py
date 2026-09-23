from collections import Counter

SCHEMA = 4

READS = (
    "hops counts the rows by the last hop their trace reached; nodes counts steps and the rows that"
    " reached each node; verdicts counts what the gate said per hop of retrieval; outcomes pairs the"
    " row's outcome with the hops it spent, so a run that answered on hop one is not read as one that"
    " spent three; rows without a trace are named, not counted as zero; a row whose every verdict came"
    " back unreadable is named too, because its arm says it graded and nothing was graded; the grader"
    " block counts both paths, the graph node through its steps and the direct path through the block"
    " it writes on the row, so a filtered arm of `single_shot` does not read as an arm that never graded;"
    " repeats counts a hop whose tool calls the row had already sent on an earlier hop, which is a hop"
    " spent to learn nothing, and it is read over the hops that called a tool, never over all hops;"
    " arguments are compared canonicalised, because two arms serialise one query by two conventions;"
    " blind_calls counts a call asked with no value in its arguments"
)


# two serialisations of one query are one query: the arms are written by two different hands
def _arguments(said) -> str:
    import json

    # a call with no arguments at all is the blindest call there is, not the string "None"
    if said is None:
        return ""
    if isinstance(said, dict):
        return json.dumps(said, sort_keys=True, ensure_ascii=False)
    text = said.strip() if isinstance(said, str) else str(said)
    try:
        return json.dumps(json.loads(text), sort_keys=True, ensure_ascii=False)
    except (ValueError, TypeError):
        return text


def _blind(said: str) -> bool:
    import json

    try:
        asked = json.loads(said)
    except (ValueError, TypeError):
        return not said
    return not any(str(v).strip() for v in asked.values()) if isinstance(asked, dict) else not asked


# every door that asks "is this the same call" reads the pair here, or the conventions drift apart
def calls(entry: dict) -> list[tuple[str, str]]:
    return [(c.get("name") or "", _arguments(c.get("arguments"))) for c in (entry.get("tool_calls") or ())]


# a hop whose set of (tool, arguments) the row already sent: the model asking itself the same thing again
def _repeats(transcript) -> tuple[int, int, int]:
    turns, seen, spent, blind = 0, set(), 0, 0
    for entry in transcript or ():
        if entry.get("role") != "assistant":
            continue
        made = calls(entry)
        if not made:
            continue
        turns += 1
        blind += sum(1 for _name, args in made if _blind(args))
        asked = frozenset(made)
        spent += asked in seen
        seen.add(asked)
    return turns, spent, blind


def report(rows: list) -> dict:
    hops, nodes, steps, verdicts, outcomes_by_hop = Counter(), Counter(), Counter(), Counter(), {}
    failed_hops, opened, dropped, announced, no_trace, unreadable = 0, 0, 0, 0, [], 0
    graded, graded_in_name_only, asked, chunks_kept, cut_verdicts = 0, [], 0, 0, 0
    calling_hops, repeat_hops, rows_repeating, blind_calls = 0, 0, [], 0
    for row in rows:
        turns, spent, blind = _repeats(getattr(row, "transcript", None))
        calling_hops += turns
        repeat_hops += spent
        blind_calls += blind
        if spent:
            rows_repeating.append(row.id)
        metrics = getattr(row, "metrics", None) or {}
        cut_verdicts += sum(
            1 for ask in (metrics.get("asks") or [])
            if ask.get("stage") == "grade" and ask.get("cut")
        )
        trace = metrics.get("trace")
        if not trace:
            no_trace.append(row.id)
            # the direct path writes no trace and grades all the same: its block is one dict on the row
            said = metrics.get("graded") or {}
            if said.get("asked"):
                graded += 1
                asked += said["asked"]
                chunks_kept += len(said.get("kept") or ())
                unreadable += said.get("unreadable") or 0
                if (said.get("unreadable") or 0) == said["asked"]:
                    graded_in_name_only.append(row.id)
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
        # a step is a hop, and a row is a row: a chunk seen again is a memo hit, not a new verdict
        steps_of_grade = [s for s in trace if s.get("node") == "grade"]
        row_asked = sum(s.get("asked") or 0 for s in steps_of_grade)
        row_unreadable = sum(s.get("unreadable") or 0 for s in steps_of_grade)
        if row_asked:
            graded += 1
            asked += row_asked
            chunks_kept += sum(s.get("kept") or 0 for s in steps_of_grade)
            # every verdict it asked for came back unreadable: the arm graded in name only
            if row_unreadable == row_asked:
                graded_in_name_only.append(row.id)
            unreadable += row_unreadable
    return {
        "schema": SCHEMA,
        "rows": len(rows),
        "rows_traced": len(rows) - len(no_trace),
        "rows_without_trace": no_trace[:20],
        "hops": dict(sorted(hops.items())),
        "nodes": {node: {"rows": nodes[node], "steps": steps[node]} for node in sorted(steps)},
        "verdicts": dict(verdicts.most_common()),
        "fallback": {"opened": opened, "dropped_context": dropped, "announced": announced},
        "grader": {
            "rows": graded, "verdicts": asked, "kept": chunks_kept,
            "unreadable": unreadable, "cut_verdicts": cut_verdicts,
            # the share the stop rule reads: over the verdicts asked, never over the pieces seen
            "unreadable_share": round(unreadable / asked, 4) if asked else None,
            "cut_share": round(cut_verdicts / asked, 4) if asked else None,
            "rows_graded_in_name_only": graded_in_name_only[:20],
        },
        "repeats": {
            "calling_hops": calling_hops,
            "repeat_hops": repeat_hops,
            "share": round(repeat_hops / calling_hops, 4) if calling_hops else None,
            "rows": len(rows_repeating),
            "rows_repeating": rows_repeating[:20],
            # a search asked with nothing in it: zero on every recorded arm, so any count is a finding
            "blind_calls": blind_calls,
        },
        "failed_hops": failed_hops,
        "outcomes_by_hops": {
            hop: dict(counts.most_common()) for hop, counts in sorted(outcomes_by_hop.items())
        },
        "reads": READS,
    }


def of_run(run_name: str) -> dict:
    from evals.loaders import load_logs

    return {"run_name": run_name, **report(load_logs(run_name))}
