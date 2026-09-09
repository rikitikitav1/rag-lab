"""Two judge prompt versions read by an external ruler, on the declared population only.

The first pass of this comparison ran from a shell over every judged row of the run, which is 138
rows where the declared population is 91. The population is the thing this arc keeps getting cut
by, so it is taken here from `judge_correlation.rows_of`, the same predicate that counts.
"""

import argparse
import json
import statistics
from pathlib import Path

SCHEMA = 2


def _by_question(run_name: str) -> dict:
    from evals.loaders import load_logs
    from evals.pools import by_question
    from evals.stats import to_unit

    scored = by_question(load_logs(run_name), lambda ql: to_unit(ql.faithfulness) is not None)
    return {
        question: {"ours": to_unit(ql.faithfulness), "answer": ql.answer or ""}
        for question, ql in scored.items()
    }


def _swaps(pairs: list) -> int:
    n = 0
    for i in range(len(pairs)):
        for j in range(i + 1, len(pairs)):
            a, b = pairs[i], pairs[j]
            if (a[0] - b[0]) * (a[1] - b[1]) < 0:
                n += 1
    return n


def measure(source_run: str, left_run: str, right_run: str, draws: int = 2000) -> dict:
    import random

    from evals.judge_correlation import rho, rows_of

    population, why = rows_of(source_run)
    left, right = _by_question(left_run), _by_question(right_run)
    rows = [
        {"question_id": r["question_id"], "guest": r["guest"],
         "left": left[r["question_id"]]["ours"], "right": right[r["question_id"]]["ours"],
         "same_answer": left[r["question_id"]]["answer"] == right[r["question_id"]]["answer"]}
        for r in population
        if r["question_id"] in left and r["question_id"] in right
    ]
    # two arms judging different answers measure the generator: this is read before the scores
    same_inputs = sum(1 for r in rows if r["same_answer"])
    if len(rows) < 3:
        return {"schema": SCHEMA, "n": len(rows), "unreadable": "fewer than three shared rows"}

    as_rows = [{"ours": r["left"], "guest": r["guest"]} for r in rows]
    other = [{"ours": r["right"], "guest": r["guest"]} for r in rows]
    rho_left, rho_right = rho(as_rows, "ours", "guest"), rho(other, "ours", "guest")

    rand = random.Random(0)
    drawn = []
    for _ in range(draws):
        pick = [rand.randrange(len(rows)) for _ in rows]
        a = rho([{"ours": rows[i]["left"], "guest": rows[i]["guest"]} for i in pick], "ours", "guest")
        b = rho([{"ours": rows[i]["right"], "guest": rows[i]["guest"]} for i in pick], "ours", "guest")
        if a is not None and b is not None:
            drawn.append(b - a)
    drawn.sort()
    band = [round(drawn[int(0.025 * len(drawn))], 4), round(drawn[int(0.975 * len(drawn))], 4)]

    return {
        "schema": SCHEMA,
        "source_run": source_run,
        "arms": [left_run, right_run],
        "population": "judge_correlation.rows_of, the declared one",
        "population_counts": why,
        "n": len(rows),
        "same_answers": f"{same_inputs}/{len(rows)}",
        "measures_the_judge": same_inputs == len(rows),
        "rank_swaps_between_arms": _swaps([(r["left"], r["right"]) for r in rows]),
        "means": {
            "left": round(statistics.fmean(r["left"] for r in rows) * 10, 4),
            "right": round(statistics.fmean(r["right"] for r in rows) * 10, 4),
            "shift": round(
                statistics.fmean(r["right"] - r["left"] for r in rows) * 10, 4
            ),
        },
        "rho_with_guest": {"left": rho_left, "right": rho_right},
        "rho_difference_right_minus_left": round((rho_right or 0) - (rho_left or 0), 4),
        "difference_ci95": band,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True)
    p.add_argument("--left", required=True)
    p.add_argument("--right", required=True)
    p.add_argument("--out")
    args = p.parse_args()
    out = measure(args.source, args.left, args.right)
    text = json.dumps(out, indent=2, ensure_ascii=False)
    print(text)
    if args.out:
        from evals.measurements import hand_back

        out = Path(args.out)
        out.write_text(text, encoding="utf-8")
        hand_back(out)
