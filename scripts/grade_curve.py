"""The table behind the closing number: every cut of one graded pass, and the cross-encoder beside it."""

import argparse

from evals.grade_curve import run

COLUMNS = ("retention", "lower edge", "gold chunks", "neighbours dropped", "strangers dropped")


def _cell(value) -> str:
    return f"{value:.4f}" if value is not None else "-"


def _table(steps: list, dial: str) -> str:
    lines = [f"| {dial} | " + " | ".join(COLUMNS) + " |", "|" + "---|" * (len(COLUMNS) + 1)]
    for step in steps:
        cells = [
            str(step["cut"]) if "cut" in step else str(step["threshold"]),
            _cell(step["gold_any"]["point"]),
            _cell(step["gold_any"]["low"]),
            _cell(step["gold_share"]["point"]),
            _cell(step["neighbours_dropped"]["point"]),
            _cell(step["strangers_dropped"]["point"]),
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("measurement")
    parser.add_argument("candidates")
    parser.add_argument("--arm", default="A", choices=("A", "B"))
    parser.add_argument("--name", default=None)
    parser.add_argument("--record", action="store_true")
    args = parser.parse_args()

    payload = run(args.measurement, args.candidates, args.arm, args.name, args.record)
    print(f"arm {payload['arm']}, form {payload['form']}, questions {payload['questions']}\n")
    print("### The grader\n")
    print(_table(payload["grader"], "cut"))
    print("\n### The cross-encoder\n")
    print(_table(payload["reranker"], "score"))
    print("\n### At matched retention\n")
    print("| cut | score | retention | strangers dropped | paired delta | ci95 |")
    print("|---|---|---|---|---|---|")
    for step in payload["matched_retention"]:
        delta = step["paired_delta"]
        print(f"| {step['cut']} | {step['threshold']} "
              f"| {step['retention'][0]:.4f} / {step['retention'][1]:.4f} "
              f"| {step['strangers_dropped'][0]:.4f} / {step['strangers_dropped'][1]:.4f} "
              f"| {delta.get('mean_delta')} | {delta.get('ci95')} |")
    print(f"\n{payload['where']}")


if __name__ == "__main__":
    main()
