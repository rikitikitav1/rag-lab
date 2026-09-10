"""Our judge on the same claims in two languages, from the command line."""

import argparse
import json
import sys
from pathlib import Path

from evals.judge_language import measure
from job_handlers.judging import _residency, judge_width, stamp_of


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default="arc3_agent_baseline")
    parser.add_argument("--rows", type=int, default=9)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    # the same stamp the job writes: a control out of regime is unreadable without the instrument
    text = json.dumps(
        measure(
            args.run_name, args.rows,
            note=lambda line: print(line, file=sys.stderr, flush=True),
            # the residency too, or the control is out of regime and cannot be read at all
            stamp=stamp_of(judge_width(None), _residency(None)),
        ),
        indent=2, ensure_ascii=False,
    )
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n")


if __name__ == "__main__":
    main()
