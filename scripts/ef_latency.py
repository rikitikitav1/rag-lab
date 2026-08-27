import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))

import config  # noqa: E402
from sqlalchemy import text  # noqa: E402
from use_cases import retrieval_compare  # noqa: E402

import db  # noqa: E402


# the same trick lived here and in the measuring module, and the two had drifted: one
# disposed the pool and did not reset the index switch, the other the reverse
def apply_ef(ef: int) -> None:
    retrieval_compare.prepare(exact=False, ef=ef)
    db.engine.dispose()


SAMPLE = """
SELECT original_text, embedding::text FROM questions
WHERE set_name = :set_name AND embedding IS NOT NULL
ORDER BY id LIMIT :limit
"""


def timings(rows, variant: str, ef: int) -> dict:
    # on checkout, because hybrid_search opens its own connection: setting the guc on a
    # connection the query never uses measures the same depth in both arms, which is
    # exactly what the first version of this script did
    apply_ef(ef)
    took = []
    for question, embedding in rows:
        started = time.perf_counter()
        db.hybrid_search(question, embedding, None, limit=20, variant=variant)
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
    report["serving_ef_search"] = config.settings.retrieval.ef_search
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2))
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    import atexit

    atexit.register(retrieval_compare.release)
    raise SystemExit(main())
