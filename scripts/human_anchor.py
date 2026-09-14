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
    # source=copy, once per run the sheet names: our judge read off the copies a new judge rejudged
    p.add_argument("--copy", action="append", default=[], metavar="SOURCE=COPY")
    args = p.parse_args()
    copies = dict(pair.split("=", 1) for pair in args.copy)

    try:
        if copies and args.action != "read":
            raise human_anchor.Refused("--copy reads our judge off copies, and only `read` does that")
        got = human_anchor.read(args.stamp, copies) if copies else getattr(human_anchor, args.action)(args.stamp)
    except human_anchor.Refused as no:
        raise SystemExit(str(no)) from no

    print(json.dumps(got, indent=2, ensure_ascii=False))
    if args.action in READINGS:
        print(say_where(READINGS[args.action], f"pairs_{args.stamp}", got, args.record))
