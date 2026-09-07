"""One recorded run driven through today's graph again, compared field by field."""

import argparse
import json
from pathlib import Path

from evals.loaders import load_logs
from evals.replay import report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default="arc3_refusals_with_context")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    text = json.dumps(
        report(args.run_name, load_logs(args.run_name)), indent=2, ensure_ascii=False
    )
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n")


if __name__ == "__main__":
    main()
