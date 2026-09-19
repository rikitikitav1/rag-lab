"""The judge-against-judge report, from the command line."""

import argparse
import json
from pathlib import Path

from evals.judge_correlation import report, rows_of


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    rows, counts = rows_of(args.run_name)
    if not rows:
        raise SystemExit("no row carries both our faithfulness and the guest's")
    text = json.dumps(report(rows, counts), indent=2, ensure_ascii=False)
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n")


if __name__ == "__main__":
    main()
