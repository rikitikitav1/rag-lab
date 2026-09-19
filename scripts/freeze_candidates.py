"""The candidates of a question set, frozen once so every arm grades the same pool."""

import argparse
import json
import time
from datetime import date
from pathlib import Path

import config
import engines
import llm
import version
from models.eval import Question
from orm.sync_db import Session, engine
from sqlalchemy import select
from use_cases import chat, grading, search_depth

import db

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "datasets" / "candidates"


def _questions(set_name: str, limit: int | None) -> list[dict]:
    with Session() as session:
        rows = session.scalars(
            select(Question).where(Question.set_name == set_name).order_by(Question.id)
        ).all()
        # the population of every other door: a question with no marked source has no gold to keep
        rows = [q for q in rows if q.marked_sources]
        rows = rows[:limit] if limit else rows
        sources = {q.source_question_id for q in rows if q.source_question_id}
        headings = dict(
            session.execute(
                select(Question.id, Question.original_text).where(Question.id.in_(sources))
            ).all()
        ) if sources else {}
        return [
            {
                "id": q.id,
                "text": q.original_text,
                "language": q.language,
                "marked_sources": list(q.marked_sources or []),
                # the source question's own text, and its own when it is not a paraphrase
                "gold_heading": headings.get(q.source_question_id) or q.original_text,
            }
            for q in rows
        ]


# postgres hands back Decimal for a weighted score, and json refuses it
def _number(value):
    return None if value is None else round(float(value), 6)


# the stand's own, so the freeze and the grader cannot disagree about what a chunk is
_digest = grading.digest


def freeze(set_name: str, variant: str, pool: int, limit: int | None) -> dict:
    questions = _questions(set_name, limit)
    label, _ = llm.embed_with_label(questions[0]["text"])
    depth = search_depth.resolve(variant)
    started, out = time.perf_counter(), []
    for nth, question in enumerate(questions, 1):
        rows, _scores, _depth = chat._retrieve_rows(
            question["text"], None, pool, False, variant, None
        )
        served = {
            (hit.source, hit.chunk_index) for hit in rows
            if not chat._hidden_by_cut(hit.source, variant)
        }
        out.append({
            **question,
            "candidates": [
                {
                    "address": f"{hit.source}#{hit.chunk_index}",
                    "source": hit.source,
                    "section": hit.section,
                    "chunk_index": hit.chunk_index,
                    "rank": rank,
                    "vector_rank": hit.vector_rank,
                    "keyword_rank": hit.keyword_rank,
                    "distance": _number(hit.distance),
                    "score": _number(hit.score),
                    "text_sha": _digest(hit.content),
                    "chars": len(hit.content),
                    # the cut policy hides a chunk from the generator without hiding it from search
                    "served": (hit.source, hit.chunk_index) in served,
                }
                for rank, hit in enumerate(rows, 1)
            ],
        })
        if nth % 100 == 0:
            print(f"{nth}/{len(questions)} in {round(time.perf_counter() - started)}s", flush=True)
    return {
        "stamp": {
            "set_name": set_name,
            "variant": variant,
            "pool": pool,
            "questions": len(out),
            "embedder": label,
            "ef_search": depth,
            "limit_vector": config.settings.retrieval.limit_vector,
            "limit_keyword": config.settings.retrieval.limit_keywords,
            "corpus_fingerprint": db.fingerprint_or_none(variant=variant),
            "cut_policy": config.settings.corpus.policy_or_none(variant),
            "code_version": version.CODE_VERSION,
            "frozen_on": date.today().isoformat(),
            "seconds": round(time.perf_counter() - started, 1),
        },
        "rows": out,
    }


# the text is not kept in the file: it lives in the corpus, and the hash says it has not moved
def texts_of(addresses: set, variant: str) -> dict:
    from sqlalchemy import text as sql

    wanted = [(a.rsplit("#", 1)[0], int(a.rsplit("#", 1)[1])) for a in addresses]
    out = {}
    with engine.connect() as conn:
        for source, index in wanted:
            row = conn.execute(
                sql(f"SELECT content FROM data_chunks WHERE {db.live_rows()}"
                    " AND source = :source AND chunk_index = :index"),
                {"variant": variant, "source": source, "index": index},
            ).first()
            if row:
                out[f"{source}#{index}"] = row[0]
    return out


def score(path: Path) -> dict:
    frozen = json.loads(path.read_text())
    variant = frozen["stamp"]["variant"]
    addresses = {c["address"] for row in frozen["rows"] for c in row["candidates"]}
    texts = texts_of(addresses, variant)
    missing = [a for a in addresses if a not in texts]
    moved = [a for a in addresses if a in texts and _digest(texts[a]) != next(
        c["text_sha"] for row in frozen["rows"] for c in row["candidates"] if c["address"] == a
    )]
    if missing or moved:
        raise SystemExit(f"{len(missing)} chunks are gone and {len(moved)} moved since the freeze")
    picked = llm.resolve("reranking")
    served = engines.served_models(picked.engine)
    started = time.perf_counter()
    for nth, row in enumerate(frozen["rows"], 1):
        pairs = [(row["text"], texts[c["address"]]) for c in row["candidates"]]
        for candidate, said in zip(row["candidates"], llm.score_pairs(pairs), strict=True):
            candidate["rerank_score"] = round(float(said), 6)
        if nth % 100 == 0:
            print(f"{nth}/{len(frozen['rows'])} in {round(time.perf_counter() - started)}s", flush=True)
    frozen["stamp"] |= {
        "reranker": f"{picked.name}@{picked.engine.name}",
        "reranker_served": sorted(served or []),
        "scored_on": date.today().isoformat(),
        "scoring_seconds": round(time.perf_counter() - started, 1),
    }
    path.write_text(json.dumps(frozen, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    return frozen["stamp"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--set-name", default="paraphrased_v2")
    parser.add_argument("--variant", default=None)
    parser.add_argument("--pool", type=int, default=20)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--score", default=None, help="add cross-encoder scores to a frozen file")
    args = parser.parse_args()

    if args.score:
        print(json.dumps(score(Path(args.score)), ensure_ascii=False, indent=1))
        return

    variant = args.variant or config.settings.corpus.variant
    frozen = freeze(args.set_name, variant, args.pool, args.limit)
    OUT.mkdir(parents=True, exist_ok=True)
    path = Path(args.out) if args.out else OUT / (
        f"{args.set_name}_{variant}_pool{args.pool}_{frozen['stamp']['frozen_on'].replace('-', '')}.json"
    )
    # compact: the file is an artifact to be read by a program, and indentation doubled it
    path.write_text(json.dumps(frozen, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(json.dumps(frozen["stamp"], ensure_ascii=False, indent=1))
    print("wrote", path)


if __name__ == "__main__":
    main()
