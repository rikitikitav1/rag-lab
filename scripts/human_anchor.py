"""The command line of the anchor: the sheet, its key and the two readings live in `evals`.

A module that reads `sys.argv` cannot be called by a job or a test without a subprocess, and this
one is now called by both.
"""

import json
import sys
from datetime import date

from evals import human_anchor
from evals.measurements import record

ACTIONS = ("build", "mark", "prune", "read", "repeats")


def main(argv: list[str]) -> None:
    action = argv[1] if len(argv) > 1 else "build"
    if action not in ACTIONS:
        raise SystemExit(f"{action!r} is not one of {', '.join(ACTIONS)}")
    stamp = next((a for a in argv[2:] if not a.startswith("-")),
                 date.today().strftime("%Y%m%d"))
    wanted = "--record" in argv

    if action in ("build", "mark", "prune"):
        got = getattr(human_anchor, action)() if action == "build" else getattr(
            human_anchor, action
        )(stamp)
        print(json.dumps(got, indent=2, ensure_ascii=False))
        return

    got = human_anchor.read(stamp) if action == "read" else human_anchor.repeats(stamp)
    print(json.dumps(got, indent=2, ensure_ascii=False))
    # a dry read must not leave a number behind: `record` is asked for, never a side effect
    kind = "human_anchor" if action == "read" else "human_anchor_repeats"
    print(f"recorded: {record(kind, f'pairs_{stamp}', got)}" if wanted
          else "not recorded; add --record once the sheet is really filled")


if __name__ == "__main__":
    try:
        main(sys.argv)
    except human_anchor.Refused as no:
        raise SystemExit(str(no)) from no
