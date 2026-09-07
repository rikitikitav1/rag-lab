"""The guest-axis probes, from the command line.

Every number it prints is `evals.guest_probes`; this file only chooses the arm, the rows and the file.
"""

import argparse
import json
import sys
from pathlib import Path

from evals.guest_probes import FENCE, arms_for, report, score
from evals.judge_correlation import overlap
from evals.loaders import load_logs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("paraphrase", "negated", "code"), required=True)
    parser.add_argument("--run-name", default="arc3_agent_baseline")
    parser.add_argument("--rows", type=int, default=10)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    from evals.guest_llm import OurClient
    from ragas.metrics._faithfulness import Faithfulness

    metric = Faithfulness(llm=OurClient())
    pool = [q for q in load_logs(args.run_name) if q.answered and q.contexts and q.answer]
    if args.arm == "code":
        pool = [q for q in pool if FENCE.search(q.answer or "")]
    done = []
    for ql in pool[: args.rows]:
        for name, answer in arms_for(args.arm, ql):
            value, n, error = score(metric, ql, answer)
            done.append({"row": ql.id, "arm": name, "score": value, "n_statements": n,
                         "error": error, "overlap": overlap(answer, ql.contexts)})
            print(f"{name} {ql.id}: {value if not error else error}", file=sys.stderr, flush=True)
    text = json.dumps(report(args.arm, done), indent=2, ensure_ascii=False)
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n")


if __name__ == "__main__":
    main()
