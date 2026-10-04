"""Distance thresholds read off a corpus, run in the worker: `docker compose exec -T worker python scripts/<this>`."""

import argparse
import json
import random
import statistics

from evals import measurements
from models.eval import READ_BY_RUNS, Question
from orm.sync_db import Session
from sqlalchemy import select

import db

IN_CORPUS = ["paraphrased_v2", "paraphrased_v2_ru"]
OFF_DOMAIN = ["off_domain", "off_domain_extra"]
OUT_OF_CORPUS = ["out_of_corpus", "out_of_corpus_extra"]


def distances(sets: list[str], variant: str) -> dict[str, list[float]]:
    with Session() as session:
        rows = session.execute(
            select(Question.language, Question.embedding, Question.embedded_by).where(
                Question.set_name.in_(sets), READ_BY_RUNS, Question.embedding.is_not(None)
            )
        ).all()
    out: dict[str, list[float]] = {}
    for language, vector, embedded_by in rows:
        found = db.nearest_distance(vector, variant=variant, embedded_by=embedded_by)
        if found is not None:
            out.setdefault(language or "?", []).append(found)
    return out


# the threshold that refuses `budget` of the in-corpus questions: the in-corpus tail above it is that share
def at_budget(values: list[float], budget: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(len(ordered) * (1 - budget)))]


def share_at_or_above(values: list[float], threshold: float | None) -> float | None:
    return sum(v >= threshold for v in values) / len(values) if values and threshold is not None else None


def halves(values: list[float], seed: int) -> tuple[list[float], list[float]]:
    shuffled = values[:]
    random.Random(seed).shuffle(shuffled)
    return shuffled[::2], shuffled[1::2]


def _round(value: float | None) -> float | None:
    return None if value is None else round(value, 4)


def summary(values: list[float]) -> dict:
    if not values:
        return {"n": 0}
    return {"n": len(values), "min": min(values), "median": statistics.median(values), "max": max(values)}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--in-sets", nargs="+", default=IN_CORPUS)
    parser.add_argument("--off-sets", nargs="+", default=OFF_DOMAIN)
    parser.add_argument("--out-sets", nargs="+", default=OUT_OF_CORPUS)
    parser.add_argument("--budget", type=float, default=0.01, help="in-corpus share a topic refusal may cost")
    parser.add_argument("--weak-share", type=float, default=0.22, help="in-corpus share the weak gate flags")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--record", action="store_true")
    args = parser.parse_args()

    inside = distances(args.in_sets, args.variant)
    off = distances(args.off_sets, args.variant)
    out = distances(args.out_sets, args.variant)
    report = {"variant": args.variant, "sets": {"in": args.in_sets, "off": args.off_sets, "out": args.out_sets},
              "budget": args.budget, "weak_share": args.weak_share, "seed": args.seed, "by_language": {}}
    for language in sorted(inside):
        mine = inside[language]
        half_a, half_b = halves(mine, args.seed)
        topic = at_budget(half_a, args.budget)
        weak = at_budget(mine, args.weak_share)
        report["by_language"][language] = {
            "in_corpus": summary(mine),
            "off_domain": summary(off.get(language, [])),
            "out_of_corpus": summary(out.get(language, [])),
            # chosen on half A, read on half B: how much of the budget survives a sample it was not fitted on
            "topic_threshold": _round(topic),
            "topic_cost_on_a": _round(share_at_or_above(half_a, topic)),
            "topic_cost_on_b": _round(share_at_or_above(half_b, topic)),
            "topic_catch_off_domain": round(share_at_or_above(off.get(language, []), topic), 4),
            "weak_distance": _round(weak),
            "weak_catch_out_of_corpus": round(share_at_or_above(out.get(language, []), weak), 4),
            # `distance_threshold` cuts the pool: above this an in-corpus question's pool comes back empty
            "distance_threshold_floor": _round(max(mine)),
        }
    print(json.dumps(report["by_language"], indent=1))
    print(measurements.say_where("threshold_calibration", args.variant, report, args.record))


if __name__ == "__main__":
    main()
