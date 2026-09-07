"""Our judge on the same claims in two languages, from the command line."""

import argparse
import json
import sys
from pathlib import Path

from evals.judge_language import measure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default="arc3_agent_baseline")
    parser.add_argument("--rows", type=int, default=9)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    text = json.dumps(
        measure(args.run_name, args.rows, note=lambda line: print(line, file=sys.stderr, flush=True)),
        indent=2, ensure_ascii=False,
    )
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n")


if __name__ == "__main__":
    main()
