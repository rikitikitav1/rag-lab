"""How often the second tool call differs when the first one did not, with and without a drop."""

import argparse
import json
from pathlib import Path

from evals import trace
from evals.loaders import load_logs


def calls_of(transcript) -> list:
    return [call for entry in (transcript or []) for call in trace.calls(entry)]


def by_question(run: str) -> dict:
    return {q.question_id: q for q in load_logs(run) if q.question_id}


def compare(left: dict, right: dict, dropped_in: dict) -> dict:
    out = {}
    for label, want in (("gate_dropped", True), ("gate_left_it", False)):
        qs = [
            q for q in set(left) & set(right)
            if q in dropped_in
            and bool(((dropped_in[q].metrics or {}).get("retrieval") or {}).get("dropped_sources"))
            == want
        ]
        first_same = [
            q for q in qs
            if calls_of(left[q].transcript)[:1] == calls_of(right[q].transcript)[:1]
            and calls_of(left[q].transcript)
        ]
        both_second = [
            q for q in first_same
            if len(calls_of(left[q].transcript)) > 1 and len(calls_of(right[q].transcript)) > 1
        ]
        differ = [
            q for q in both_second
            if calls_of(left[q].transcript)[1] != calls_of(right[q].transcript)[1]
        ]
        out[label] = {
            "questions": len(qs),
            "first_call_same": len(first_same),
            "both_reached_a_second": len(both_second),
            "second_call_differs": len(differ),
            "rate": round(len(differ) / len(both_second), 4) if both_second else None,
        }
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", nargs="+", required=True, help="left:right per pair")
    parser.add_argument("--dropped-in", required=True, help="run whose gate decides the split")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    marker = by_question(args.dropped_in)
    runs = {}
    report = {"schema": 1, "dropped_in": args.dropped_in, "pairs": {}}
    for pair in args.pairs:
        left, right = pair.split(":")
        for name in (left, right):
            runs.setdefault(name, by_question(name))
        report["pairs"][pair] = compare(runs[left], runs[right], marker)

    text = json.dumps(report, indent=2, ensure_ascii=False)
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n")


if __name__ == "__main__":
    main()
