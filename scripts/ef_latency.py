import argparse
import json
import statistics
import time
from pathlib import Path

import config
from sqlalchemy import text

import db

SAMPLE = """
SELECT original_text, embedding::text FROM questions
WHERE set_name = :set_name AND embedding IS NOT NULL
ORDER BY id LIMIT :limit
"""


def timings(rows, variant: str, ef: int) -> dict:
    # hybrid_search opens its own connection and sets the depth on it, per call
    took = []
    for question, embedding in rows:
        started = time.perf_counter()
        db.hybrid_search(question, embedding, None, limit=20, variant=variant, ef_search=ef)
        took.append((time.perf_counter() - started) * 1000)
    took.sort()
    return {
        "median_ms": round(statistics.median(took), 1),
        "p95_ms": round(took[int(len(took) * 0.95)], 1),
        "questions": len(took),
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description="What depth of the hnsw walk costs, per variant, through the index."
    )
    ap.add_argument("--set", dest="set_name", default="paraphrased_v2_ru")
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--variants", nargs="+", required=True)
    ap.add_argument("--ef", nargs="+", type=int, default=[100, 200])
    ap.add_argument("--out", default=None)
    ap.add_argument("--note", default="")
    args = ap.parse_args()

    with db.engine.connect() as conn:
        rows = conn.execute(
            text(SAMPLE), {"set_name": args.set_name, "limit": args.limit}
        ).all()

    report = {"set": args.set_name, "note": args.note, "by_variant": {}}
    for variant in args.variants:
        report["by_variant"][variant] = {}
        for ef in args.ef:
            got = timings(rows, variant, ef)
            report["by_variant"][variant][str(ef)] = got
            print(f"{variant:14} ef={ef:<4} median {got['median_ms']:6.1f} ms  p95 {got['p95_ms']:6.1f} ms")
    from use_cases import search_depth

    # resolved: an artifact recording the word `auto` cannot be compared against a number
    report["serving_ef_search"] = search_depth.resolve()
    report["declared_ef_search"] = config.settings.retrieval.ef_search
    with db.engine.connect() as conn:
        pages, rows = search_depth._shape(conn)
    report["table"] = {"relpages": pages, "reltuples": rows,
                       "variants": [v["variant"] for v in db.corpus_variants()]}
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2))
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
