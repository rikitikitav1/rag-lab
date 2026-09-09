"""The command line of the anchor: the sheet, its key and the two readings live in `evals`.

A module that reads `sys.argv` cannot be called by a job or a test without a subprocess, and this
one is now called by both.
"""

import argparse
import json
from datetime import date

from evals import human_anchor
from evals.measurements import say_where

ACTIONS = ("build", "mark", "prune", "read", "repeats")
READINGS = {"read": "human_anchor", "repeats": "human_anchor_repeats"}

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=ACTIONS, nargs="?", default="build")
    p.add_argument("stamp", nargs="?", default=date.today().strftime("%Y%m%d"))
    p.add_argument("--record", action="store_true")
    args = p.parse_args()

    try:
        got = getattr(human_anchor, args.action)(args.stamp)
    except human_anchor.Refused as no:
        raise SystemExit(str(no)) from no

    print(json.dumps(got, indent=2, ensure_ascii=False))
    if args.action in READINGS:
        print(say_where(READINGS[args.action], f"pairs_{args.stamp}", got, args.record))
