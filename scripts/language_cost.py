"""The command line of `evals.language_cost`: both cuts, the floor, and where the number goes."""

import argparse
import json

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--before", required=True)
    p.add_argument("--after", required=True)
    p.add_argument("--floor-against")
    p.add_argument("--record", action="store_true")
    args = p.parse_args()

    from evals import language_cost
    from evals.measurements import record

    got = language_cost.measure(args.before, args.after, args.floor_against)
    print(json.dumps(got, indent=2, ensure_ascii=False))
    # a dry read must not leave a number behind: recording is asked for, never a side effect
    print(f"recorded: {record('language_cost', args.after, got)}" if args.record
          else "not recorded; add --record to leave the number in datasets/measurements")
