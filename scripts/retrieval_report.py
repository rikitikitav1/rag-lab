"""Retrieval quality over stored question embeddings: no generator, no judge, no noise."""

import argparse
import hashlib
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "app"))

CANDIDATES = 100
DEPTH = 20
CUTOFFS = (1, 3, 5, 10)
NO_THRESHOLD = 2.0
BOOTSTRAP = 2000
EF_SEARCH = 200


def _clean(text):
    return re.sub(r"[_*`]", "", text or "").strip().lower()


def _heading_text(section):
    """section is a heading path ("h1 > 12. Question?"); only the leaf identifies the section."""
    leaf = (section or "").split(" > ")[-1]
    return _clean(re.sub(r"^\d+\.\s*", "", leaf))


def questions(conn, text, set_name, limit):
    from sqlalchemy import text as sql

    rows = conn.execute(
        sql("""
            SELECT q.id, q.original_text, q.marked_sources, q.embedding::text AS emb,
                   COALESCE(o.original_text, q.original_text) AS gold_heading
            FROM questions q
            LEFT JOIN questions o ON o.id = q.source_question_id
            WHERE q.set_name = :set_name
              AND q.embedding IS NOT NULL
              AND array_length(q.marked_sources, 1) > 0
            ORDER BY q.id
            LIMIT :limit
        """),
        {"set_name": set_name, "limit": limit},
    ).mappings().all()
    return [dict(r) for r in rows]


def ranked_lists(db, question, variant, depth=DEPTH, limit_keyword=CANDIDATES,
                 limit_vector=CANDIDATES, distance_threshold=NO_THRESHOLD):
    rows = db.hybrid_search(
        question["original_text"],
        question["emb"],
        None,
        limit_vector=limit_vector,
        limit_keyword=limit_keyword,
        limit=CANDIDATES,
        variant=variant,
        distance_threshold=distance_threshold,
    )
    files, sections = [], []
    for row in rows:
        source, section = row[1], row[8]
        if source not in files:
            files.append(source)
        key = (source, _heading_text(section))
        if key not in sections:
            sections.append(key)
    return files[:depth], sections[:depth], rows


def rank_of_file(files, marked):
    for i, source in enumerate(files, 1):
        if any(m in source for m in marked):
            return i
    return None


_APPLIED = None


def prepare(exact: bool, ef: int = EF_SEARCH) -> None:
    """Exact search measures the corpus; hnsw measures the index, and it is not reproducible.

    On checkout, not on connect: hybrid_search opens its own connection, and a pooled one created
    before this ran would come back without the setting and quietly search through the index.
    """
    global _APPLIED
    from orm.sync_db import engine
    from sqlalchemy import event

    if _APPLIED is not None:
        event.remove(engine, "checkout", _APPLIED)
        _APPLIED = None

    setting = "SET enable_indexscan = off" if exact else f"SET hnsw.ef_search = {int(ef)}"

    def apply(dbapi_conn, _record, _proxy):
        cursor = dbapi_conn.cursor()
        cursor.execute("RESET enable_indexscan")
        cursor.execute(setting)
        cursor.close()

    event.listen(engine, "checkout", apply)
    _APPLIED = apply


POOL_SHORTFALL = []


def assert_pool(rows, question_id, asked) -> None:
    """A pool smaller than asked means something capped the search, and the label is then a lie."""
    if len(rows) < asked:
        POOL_SHORTFALL.append((question_id, len(rows)))


def section_exists(conn, variant, marked, gold_heading):
    """A question is scorable at section level only if some chunk carries that section."""
    from sqlalchemy import text as sql

    gold = _clean(gold_heading)
    if not gold:
        return False
    rows = conn.execute(
        sql("""
            SELECT DISTINCT section FROM data_chunks
            WHERE variant = :variant AND section IS NOT NULL
              AND (""" + " OR ".join(f"position(:m{i} in source) > 0" for i in range(len(marked))) + """)
        """),
        {"variant": variant, **{f"m{i}": m for i, m in enumerate(marked)}},
    ).scalars().all()
    return any(_heading_text(s) == gold for s in rows)


def rank_of_section(sections, marked, gold_heading):
    gold = _clean(gold_heading)
    for i, (source, heading) in enumerate(sections, 1):
        if any(m in source for m in marked) and heading and heading == gold:
            return i
    return None


def measure(db, conn, set_name, variant, limit, exact, ef=EF_SEARCH, limit_keyword=CANDIDATES,
            limit_vector=CANDIDATES, distance_threshold=NO_THRESHOLD):
    qs = questions(conn, None, set_name, limit)
    prepare(exact, ef)
    out = []
    for q in qs:
        files, sections, rows = ranked_lists(
            db, q, variant, limit_keyword=limit_keyword, limit_vector=limit_vector,
            distance_threshold=distance_threshold,
        )
        assert_pool(rows, q["id"], min(limit_vector, CANDIDATES))
        scorable = section_exists(conn, variant, q["marked_sources"], q["gold_heading"])
        out.append({
            "id": q["id"],
            "file_rank": rank_of_file(files, q["marked_sources"]),
            "section_scorable": scorable,
            "section_rank": (
                rank_of_section(sections, q["marked_sources"], q["gold_heading"])
                if scorable else None
            ),
            "files": files,
            "sections": [list(s) for s in sections],
        })
    return out


def recall_against_exact(db, conn, set_name, variant, limit, ef=EF_SEARCH):
    """The agent and the interactive path go through hnsw, so its recall is a run-level fact."""
    exact = {r["id"]: r for r in measure(db, conn, set_name, variant, limit, True)}
    approx = measure(db, conn, set_name, variant, limit, False, ef)
    scores = []
    for row in approx:
        gold = {tuple(s) for s in exact[row["id"]]["sections"]}
        got = {tuple(s) for s in row["sections"]}
        if gold:
            scores.append(len(gold & got) / len(gold))
    return round(sum(scores) / len(scores), 4) if scores else None


def _rr(rank):
    return 1.0 / rank if rank else 0.0


def summarise(rows, level):
    key = f"{level}_rank"
    if level == "section":
        rows = [r for r in rows if r.get("section_scorable")]
    n = len(rows)
    if not n:
        return {}
    stats = {f"hit@{c}": sum(1 for r in rows if r[key] and r[key] <= c) / n for c in CUTOFFS}
    stats[f"MRR@{DEPTH}"] = sum(_rr(r[key]) for r in rows) / n
    stats["n"] = n
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in stats.items()}


def _bootstrap(deltas, seed=0):
    rng = random.Random(seed)
    means = []
    for _ in range(BOOTSTRAP):
        sample = [deltas[rng.randrange(len(deltas))] for _ in deltas]
        means.append(sum(sample) / len(sample))
    means.sort()
    return means[int(0.025 * BOOTSTRAP)], means[int(0.975 * BOOTSTRAP)]


# a delta between two different procedures is a number about nothing
COMPARABLE = (
    "set", "search", "candidates", "limit_vector", "limit_keyword",
    "distance_threshold", "keyword", "questions_hash",
)


def comparable(before, after) -> list[tuple]:
    return [
        (field, before.get(field), after.get(field))
        for field in COMPARABLE
        if before.get(field) != after.get(field)
    ]


def compare(before, after, level):
    key = f"{level}_rank"
    if level == "section":
        before = [r for r in before if r.get("section_scorable")]
        after = [r for r in after if r.get("section_scorable")]
    was = {r["id"]: r for r in before}
    paired = [(was[r["id"]], r) for r in after if r["id"] in was]
    if not paired:
        return {"error": "no shared questions"}
    deltas = [_rr(now[key]) - _rr(then[key]) for then, now in paired]
    better = sum(1 for d in deltas if d > 0)
    worse = sum(1 for d in deltas if d < 0)
    low, high = _bootstrap(deltas)
    return {
        "level": level,
        "questions": len(paired),
        "delta_MRR": round(sum(deltas) / len(deltas), 4),
        "ci95": [round(low, 4), round(high, 4)],
        "better": better,
        "worse": worse,
        "unchanged": len(deltas) - better - worse,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variant", default=None)
    ap.add_argument("--set", dest="set_name", default="paraphrased_ru")
    ap.add_argument("--limit", type=int, default=1000)
    ap.add_argument("--out", default=None)
    ap.add_argument("--hnsw", action="store_true", help="measure through the index, not exactly")
    ap.add_argument("--recall", action="store_true", help="recall@20 of hnsw against exact search")
    ap.add_argument("--ef", type=int, default=None, help="hnsw.ef_search for the index runs")
    ap.add_argument("--keyword-query", choices=("and", "or"))
    ap.add_argument("--limit-keyword", type=int, default=CANDIDATES)
    ap.add_argument("--limit-vector", type=int, default=CANDIDATES)
    ap.add_argument("--distance-threshold", type=float, default=NO_THRESHOLD)
    ap.add_argument("--production-limits", action="store_true",
                    help="what the agent actually runs: hnsw, 20/20, threshold on")
    ap.add_argument("--keyword-rank", choices=("ts_rank", "ts_rank_cd"))
    ap.add_argument("--keyword-norm", type=int)
    ap.add_argument("--query-lang", choices=("langdetect", "cyrillic_ratio", "function_words"))
    ap.add_argument("--compare", nargs=2, metavar=("BEFORE", "AFTER"))
    ap.add_argument("--force", action="store_true", help="compare runs of different procedures")
    args = ap.parse_args()

    if args.compare:
        before = json.loads(Path(args.compare[0]).read_text())
        after = json.loads(Path(args.compare[1]).read_text())
        differ = comparable(before, after)
        if differ and not args.force:
            print("refusing to compare, these runs are not the same procedure:")
            for field, was, now in differ:
                print(f"  {field}: {was} -> {now}")
            return 1
        print(f"{before['variant']} -> {after['variant']}  set={after['set']} "
              f"search={after.get('search')} pool={after.get('limit_vector')}")
        for level in ("file", "section"):
            print(json.dumps(compare(before["rows"], after["rows"], level)))
        return 0

    import config
    from orm.sync_db import engine
    from use_cases.index import check_variant

    import db

    # default to what the server runs, or a gate would pass while production loses recall
    if args.ef is None:
        args.ef = config.settings.retrieval.ef_search

    for name in ("keyword_query", "keyword_rank", "keyword_norm", "query_lang"):
        chosen = getattr(args, name)
        if chosen is not None:
            setattr(config.settings.retrieval, name, chosen)

    variant = check_variant(args.variant or config.settings.corpus.variant)
    if args.production_limits:
        args.hnsw = True
        args.ef = config.settings.retrieval.ef_search
        args.limit_keyword = config.settings.retrieval.limit_keywords
        args.limit_vector = config.settings.retrieval.limit_vector
        args.distance_threshold = config.settings.retrieval.distance_threshold
    exact = not args.hnsw
    with engine.connect() as conn:
        if args.recall:
            score = recall_against_exact(db, conn, args.set_name, variant, args.limit, args.ef)
            print(f"variant={variant} set={args.set_name} ef_search={args.ef} "
                  f"hnsw recall@{DEPTH} vs exact: {score}")
            if args.out:
                Path(args.out).write_text(json.dumps({
                    "variant": variant, "set": args.set_name, "ef_search": args.ef,
                    "recall_at_20_vs_exact": score,
                }))
                print(f"wrote {args.out}")
            return 0
        rows = measure(
            db, conn, args.set_name, variant, args.limit, exact, ef=args.ef,
            limit_keyword=args.limit_keyword, limit_vector=args.limit_vector,
            distance_threshold=args.distance_threshold,
        )
        fingerprint = db.corpus_fingerprint(variant=variant)

    ids = ",".join(str(r["id"]) for r in rows)
    shortfall = list(POOL_SHORTFALL)
    report = {
        "variant": variant,
        "set": args.set_name,
        "search": "exact" if exact else f"hnsw ef_search={args.ef}",
        "candidates": CANDIDATES,
        "limit_vector": args.limit_vector,
        "limit_keyword": args.limit_keyword,
        "distance_threshold": args.distance_threshold,
        "keyword": {
            "query": config.settings.retrieval.keyword_query,
            "rank": config.settings.retrieval.keyword_rank,
            "norm": config.settings.retrieval.keyword_norm,
            "query_lang": config.settings.retrieval.query_lang,
        },
        "questions": len(rows),
        "questions_hash": hashlib.md5(ids.encode()).hexdigest()[:12],
        "fingerprint": fingerprint,
        "policy": config.settings.corpus.policy(variant),
        "file": summarise(rows, "file"),
        "section": summarise(rows, "section"),
        "rows": rows,
    }
    print(f"variant={variant} set={args.set_name} n={len(rows)} "
          f"search={report['search']} candidates={CANDIDATES}")
    if shortfall:
        print(f"  WARNING vector leg shorter than the asked pool on {len(shortfall)} questions, "
              f"smallest {min(n for _, n in shortfall)}")
    print(f"  file   : {json.dumps(report['file'])}")
    scorable = sum(1 for r in rows if r.get("section_scorable"))
    print(f"  section: {json.dumps(report['section'])}"
          f"  (scorable {scorable}/{len(rows)})")
    if args.out:
        Path(args.out).write_text(json.dumps(report))
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
