"""Retrieval quality over stored question embeddings: no generator, no judge, no noise."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "app"))

import config  # noqa: E402  the path above is what makes it importable

# the measuring lives in the app so the comparison job and this script read a rank the
# same way; what stays here is the CLI, the artifacts and the comparison of two files
from use_cases import search_depth  # noqa: E402
from use_cases.retrieval_compare import (  # noqa: E402
    CANDIDATES,
    DEPTH,
    NO_THRESHOLD,
    POOL_SHORTFALL,
    arm_procedure,
    comparable,
    file_drift,
    half_of,
    measure,
    paired_delta,
    questions,
    summarise,
)


# a rung deep enough for the planner to prefer sorting the whole table is not a rung of
# the index at all: recall then compares exact search with exact search and reads 1.0 by
# construction. The crossover moves with the row count, so this is asked per variant and
# per depth rather than remembered
def vector_plan(conn, variant: str, ef: int) -> str:
    # one owner for the question: this used to ask its own EXPLAIN, with another LIMIT
    # and an inner select that could pick a row with no embedding, so the two answers
    # agreed only by luck
    return "index" if search_depth.uses_index(conn, variant, ef) else "sort"


def recall_against_exact(db, conn, set_name, variant, limit, ef=None):
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

# a delta between two different procedures is a number about nothing
# the ladder is fixed and small on purpose: a required_ef read off a continuous search
# would be a number chosen from the data it is meant to judge. It lives in the config
# with the value it produces, so the two cannot drift apart
def ef_ladder() -> tuple[int, ...]:
    # sorted, through the same accessor production uses: two readings of one ladder, one
    # of them sorted, is two different ladders whenever somebody writes it out of order
    return tuple(search_depth.ladder())


def recall_gate() -> float:
    return config.settings.retrieval.recall_gate


def mrr_loss_gate() -> float:
    return config.settings.retrieval.max_mrr_loss


def lost_questions_gate() -> int:
    return config.settings.retrieval.max_questions_lost


# recall@20 says how many of the exact neighbours the graph found. What we are paid for is
# where the right section lands, and a neighbour lost at rank 18 does not move that at all.
# So the depth is judged by what it costs the metric, on the same questions, paired
def index_cost(db, conn, set_name, variant, limit, ef):
    exact = measure(db, conn, set_name, variant, limit, True)
    approx = measure(db, conn, set_name, variant, limit, False, ef)
    out = {}
    for level in ("section", "file"):
        out[level] = paired_delta(exact, approx, level)
        # a summary saying "4 worse" is a number without a body: the questions that paid
        # are the only thing a person can look at
        rows = _worse_rows(exact, approx, level)
        out[level]["worse_rows"] = rows
        # the ones that did not merely slip: exact found the section, the index did not
        out[level]["lost"] = sum(1 for r in rows if r["index"] is None)
    return out


def _worse_rows(exact, approx, level):
    key = f"{level}_rank"
    was = {r["id"]: r for r in exact}
    rows = []
    for now in approx:
        then = was.get(now["id"])
        if then is None or (level == "section" and not now.get("section_scorable")):
            continue
        if _rr(now[key]) < _rr(then[key]):
            rows.append(
                {"id": now["id"], "exact": then[key], "index": now[key], "repo": now.get("repo")}
            )
    return rows



def repo_coverage(rows, ids) -> int:
    return len({row.get("repo") for row in rows if row["id"] in ids and row.get("repo")})


def ids_hash(ids) -> str:
    joined = ",".join(str(i) for i in sorted(ids))
    return hashlib.md5(joined.encode(), usedforsecurity=False).hexdigest()[:12]

# a grown set is not a different procedure: same questions plus new ones, and the old
# ids still sit inside. Anything else differing still refuses, --force is for that
def set_grew(before, after, differ) -> bool:
    if [field for field, _, _ in differ] != ["questions_hash"]:
        return False
    was = {row["id"] for row in before["rows"]}
    now = {row["id"] for row in after["rows"]}
    return bool(was) and was < now


# the delta is over questions, and a question can only move if both sides hold its file.
# What each side holds alone belongs next to the delta, not in a separate investigation
def print_file_drift(before: str, after: str) -> None:
    try:
        from orm.sync_db import engine

        with engine.connect() as conn:
            rows = file_drift(conn, before, after)
    except Exception as e:  # an archived pair may name variants this database no longer has
        print(f"corpus drift: not readable ({type(e).__name__})")
        return
    if not rows:
        print("corpus drift: none, both variants hold the same files")
        return
    for row in rows:
        print(
            f"corpus drift: {row['family']} only in {before}: {row['only_before']},"
            f" only in {after}: {row['only_after']}, in both: {row['shared']}"
        )


def compare_half(before, after, level, which):
    # over the ids both sides carry, not over everything the newer run measured: the hash
    # has to name the questions the numbers were taken on
    shared = {r["id"] for r in before} & {r["id"] for r in after}
    kept = {qid for qid in shared if half_of(qid) == which}
    result = paired_delta(
        [r for r in before if r["id"] in kept],
        [r for r in after if r["id"] in kept],
        level,
    )
    result["half"] = which
    result["repos"] = repo_coverage(after, kept)
    result["ids_hash"] = ids_hash(kept)
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variant", default=None)
    ap.add_argument(
        "--set", dest="set_name",
        default=config.settings.retrieval.criterion_sets[0],
        help="defaults to the first criterion set declared in config",
    )
    ap.add_argument("--limit", type=int, default=1000)
    ap.add_argument("--out", default=None)
    ap.add_argument("--hnsw", action="store_true", help="measure through the index, not exactly")
    ap.add_argument("--recall", action="store_true", help="recall@20 of hnsw against exact search")
    ap.add_argument(
        "--required-ef",
        action="store_true",
        help=f"recall through the index at each of {', '.join(map(str, ef_ladder()))}, "
        f"against exact search, with the plan of each rung. Reports rather than gates: the "
        f"gate on the depth is max_mrr_loss, measured by --index-cost",
    )
    ap.add_argument(
        "--index-cost",
        action="store_true",
        help="what each rung of the ladder costs the metric against exact search, paired, "
        "with a bootstrap interval; the gate is on the loss, not on recall",
    )
    ap.add_argument(
        "--rerank-top",
        type=int,
        default=0,
        help="reorder this many fused candidates with the cross-encoder before the ranks "
        "are read; 0 leaves the fusion alone",
    )
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
        grew = set_grew(before, after, differ)
        if grew:
            shared = len({r["id"] for r in before["rows"]})
            print(f"the set grew: comparing on the {shared} questions both sides carry")
            differ = []
        if differ and not args.force:
            print("refusing to compare, these runs are not the same procedure:")
            for field, was, now in differ:
                print(f"  {field}: {was} -> {now}")
            return 1
        print(f"{before['variant']} -> {after['variant']}  set={after['set']} "
              f"search={after.get('search')} pool={after.get('limit_vector')}")
        print_file_drift(before["variant"], after["variant"])
        # a report taken before rows carried a repository still splits by id parity, the
        # same rule as now; the repository is printed as coverage, never used to split
        for level in ("file", "section"):
            print(json.dumps(paired_delta(before["rows"], after["rows"], level)))
            for which in ("A", "B"):
                print(json.dumps(compare_half(before["rows"], after["rows"], level, which)))
        return 0

    from orm.sync_db import engine
    from use_cases.index import check_variant

    import db

    # default to what the server runs, or a gate would pass while production loses recall.
    # Resolved, never `auto`: what a report says it measured has to be a number
    variant_for_depth = args.variant or config.settings.corpus.variant
    if args.ef is None:
        args.ef = search_depth.resolve(variant_for_depth)

    for name in ("keyword_query", "keyword_rank", "keyword_norm", "query_lang"):
        chosen = getattr(args, name)
        if chosen is not None:
            setattr(config.settings.retrieval, name, chosen)

    variant = check_variant(args.variant or config.settings.corpus.variant)
    if args.production_limits:
        args.hnsw = True
        args.ef = search_depth.resolve(variant)
        args.limit_keyword = config.settings.retrieval.limit_keywords
        args.limit_vector = config.settings.retrieval.limit_vector
        args.distance_threshold = config.settings.retrieval.distance_threshold
    exact = not args.hnsw
    with engine.connect() as conn:
        if args.index_cost:
            costs, cheapest = {}, None
            for ef in ef_ladder():
                plan = vector_plan(conn, variant, ef)
                if plan != "index":
                    print(f"variant={variant} ef_search={ef} refused: no index in the plan")
                    continue
                cost = index_cost(db, conn, args.set_name, variant, args.limit, ef)
                costs[ef] = cost
                # `paired_delta` answers "no shared questions" with an error and no
                # interval; reading `ci95` off that shape blew up two lines below and
                # called a failure to measure a crash
                if "error" in cost["section"]:
                    print(
                        f"variant={variant} ef_search={ef} not measured: "
                        f"{cost['section']['error']}"
                    )
                    continue
                worst = -cost["section"]["ci95"][0]
                lost = cost["section"]["lost"]
                print(
                    f"variant={variant} ef_search={ef} section MRR against exact: "
                    f"{cost['section']['delta_MRR']:+.4f} ci95 {cost['section']['ci95']} "
                    f"n={cost['section']['questions']}"
                )
                # the interval, not the point: a loss we cannot tell from zero is a loss
                # we have no business paying depth for
                if lost > lost_questions_gate():
                    print(
                        f"variant={variant} ef_search={ef} refused: {lost} questions the exact "
                        f"arm answered are not in the index arm at all"
                    )
                    continue
                if cheapest is None and worst <= mrr_loss_gate():
                    cheapest = ef
            # not the depth production serves: `search_depth` takes the deepest rung
            # that still walks the index, this takes the shallowest rung inside the
            # gate. Two questions, two numbers, and only this one is a recommendation
            print(f"cheapest_ef={cheapest} (shallowest rung inside the gate: worst-case "
                  f"section MRR loss {mrr_loss_gate()}; the served depth is the deepest "
                  f"rung that still walks the index, see search_depth)")
            if args.out:
                path = Path(args.out)
                report = json.loads(path.read_text()) if path.exists() else {}
                report["index_cost"] = {str(k): v for k, v in costs.items()}
                report["set"] = args.set_name
                report["variant"] = variant
                # the corpus this depth was priced against: without it a file cannot say
                # whether it still describes the table, and two of them already did not
                report["fingerprint"] = db.fingerprint_or_none(variant=variant)
                report["cheapest_ef"] = cheapest
                report["mrr_loss_gate"] = mrr_loss_gate()
                report["max_questions_lost"] = lost_questions_gate()
                path.write_text(json.dumps(report))
                print(f"wrote {args.out}")
            return 0 if cheapest else 1

        if args.required_ef:
            by_ef = {}
            required = None
            plans = {}
            for ef in ef_ladder():
                plans[ef] = vector_plan(conn, variant, ef)
                if plans[ef] != "index":
                    by_ef[ef] = None
                    print(
                        f"variant={variant} ef_search={ef} refused: the planner sorts the "
                        f"table at this depth, so recall would compare exact with exact"
                    )
                    continue
                score = recall_against_exact(db, conn, args.set_name, variant, args.limit, ef)
                by_ef[ef] = score
                print(f"variant={variant} ef_search={ef} recall@{DEPTH} vs exact: {score}")
                if required is None and score is not None and score >= recall_gate():
                    required = ef
                    break
            print(f"required_ef={required}")
            if required is None:
                print(
                    "no rung of the ladder clears the recall gate through the index. That "
                    "is a reading, not a failure: recall is a diagnostic now and the gate "
                    "on the depth is max_mrr_loss, measured by --index-cost"
                )
            if args.out:
                path = Path(args.out)
                report = json.loads(path.read_text()) if path.exists() else {}
                report["required_ef"] = required
                report["recall_by_ef"] = by_ef
                report["plan_by_ef"] = plans
                report["set"] = args.set_name
                report["variant"] = variant
                report["fingerprint"] = db.fingerprint_or_none(variant=variant)
                report["questions"] = len(questions(conn, args.set_name, args.limit))
                report["recall_gate"] = recall_gate()
                path.write_text(json.dumps(report))
                print(f"wrote {args.out}")
            # the ladder reports; it stopped gating when recall did
            return 0

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
            distance_threshold=args.distance_threshold, rerank_top=args.rerank_top,
        )

    shortfall = list(POOL_SHORTFALL)
    # the same producer the comparison job writes each of its arms with: one run of this
    # script is one arm, and two artifacts of different provenance stay comparable only
    # while nobody writes the shape twice
    arm = {
        "variant": variant,
        "ef_search": None if exact else args.ef,
        "rerank_top": args.rerank_top,
        "limit_vector": args.limit_vector,
        "limit_keyword": args.limit_keyword,
        "distance_threshold": args.distance_threshold,
    }
    report = {
        **arm_procedure(arm, rows, args.set_name),
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
