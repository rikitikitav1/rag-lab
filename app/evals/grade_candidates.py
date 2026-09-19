"""The grader over frozen candidates: no generator, no judge, one verdict per question and chunk."""

import json
import random
import time
from pathlib import Path

import job_queue
import logging_setup
from errors import StandFault
from evals import gold_classes, measurements
from use_cases import grading

log = logging_setup.get_logger(__name__)

SCHEMA = 1
FORMS = ("per_chunk", "whole_text")

READS = (
    "verdicts are per question and chunk, taken once; retention is read on the rows whose gold"
    " section reached the population, the floor on strangers, and a neighbour section of the gold"
    " file is neither; `p` is the probability the model gave the word it said, so a cut of the"
    " share can be moved over the record without asking the model again"
)


def population(row: dict, top: int) -> list[dict]:
    by_fusion = sorted(row["candidates"], key=lambda c: c["rank"])[:top]
    by_score = sorted(row["candidates"], key=lambda c: -(c.get("rerank_score") or 0))[:top]
    seen = {}
    for candidate in by_fusion + by_score:
        seen.setdefault(candidate["address"], candidate)
    return list(seen.values())


def _texts(addresses: list[str], variant: str) -> dict:
    from orm.sync_db import engine
    from sqlalchemy import text as sql

    import db

    out = {}
    with engine.connect() as conn:
        for address in addresses:
            source, _, place = address.rpartition("#")
            row = conn.execute(
                sql(f"SELECT content FROM data_chunks WHERE {db.live_rows()}"
                    " AND source = :source AND chunk_index = :index"),
                {"variant": variant, "source": source, "index": int(place)},
            ).first()
            if row:
                out[address] = row[0]
    return out


def _one_question(row: dict, chunks: list[dict], texts: dict, system: str, ask, form: str) -> dict:
    pieces = [texts[c["address"]] for c in chunks]
    addresses = [c["address"] for c in chunks]
    if form == "whole_text":
        # the canon of LangGraph: one verdict over everything the search returned
        said = grading.piece_verdict(row["text"], "\n\n".join(pieces), system, ask, "whole")
        kept = list(range(len(chunks))) if said != "no" else []
        return {"kept": kept, "asked": 1, "unreadable": int(said is None), "order": addresses,
                "dropped": [] if kept else addresses}
    return grading.grade_pieces(
        row["text"], pieces, [{"source": c["source"], "chunk_index": c["chunk_index"]}
                              for c in chunks], system, ask
    )


# the same rows every pass and every form: a sample drawn per run compares two draws, not two arms
def drawn(rows: list, sample: int | None, seed: int, limit: int | None) -> list:
    if sample and sample < len(rows):
        picked = random.Random(seed).sample(range(len(rows)), sample)
        return [rows[i] for i in sorted(picked)]
    return rows[:limit] if limit else rows


def run(path: str, form: str = "per_chunk", top: int = 5, limit: int | None = None,
        sample: int | None = None, seed: int = 0, name: str | None = None,
        shuffle: int | None = None, job_id: int | None = None) -> dict:
    if form not in FORMS:
        raise StandFault(f"a call form is one of {FORMS}, got {form!r}")
    frozen = json.loads(Path(path).read_text())
    variant = frozen["stamp"]["variant"]
    rows = drawn(frozen["rows"], sample, seed, limit)
    if shuffle is not None:
        # the same rows in another order: a floor taken twice in one residency must move the prefix cache
        rows = list(rows)
        random.Random(shuffle).shuffle(rows)
    system = grading.system_prompt()
    started, out, asks_all, cancelled = time.perf_counter(), [], [], False
    for nth, row in enumerate(rows, 1):
        if job_id is not None and job_queue.is_cancelled(job_id):
            cancelled = True
            break
        chunks = population(row, top)
        texts = _texts([c["address"] for c in chunks], variant)
        missing = [c["address"] for c in chunks if c["address"] not in texts]
        moved = [c["address"] for c in chunks
                 if c["address"] in texts and grading.digest(texts[c["address"]]) != c["text_sha"]]
        if missing or moved:
            raise StandFault(
                f"the corpus moved under the frozen file: {len(missing)} chunks gone,"
                f" {len(moved)} rewritten (question {row['id']})"
            )
        asks: list = []
        graded = _one_question(row, chunks, texts, system, grading.ask_door(asks), form)
        kept = set(graded["kept"])
        out.append({
            "question_id": row["id"],
            "classes": [gold_classes.classify(c, row["marked_sources"], row["gold_heading"])
                        for c in chunks],
            "addresses": [c["address"] for c in chunks],
            "kept": sorted(kept),
            "asked": graded["asked"],
            "unreadable": graded["unreadable"],
            "verdicts": [{"key": a["key"], "text": a["text"], **({"p": a["p"]} if "p" in a else {})}
                         for a in asks],
        })
        asks_all += asks
        if nth % 50 == 0:
            log.info("grade_candidates.progress", done=nth, of=len(rows),
                     seconds=round(time.perf_counter() - started))
    payload = {
        "schema": SCHEMA,
        "form": form,
        "top": top,
        "candidates_file": Path(path).name,
        "sample": sample,
        "seed": seed,
        "shuffle": shuffle,
        "candidates_stamp": frozen["stamp"],
        "questions": len(out),
        "cancelled": cancelled,
        "asked": sum(r["asked"] for r in out),
        "unreadable": sum(r["unreadable"] for r in out),
        "seconds": round(time.perf_counter() - started, 1),
        "rows": out,
        "reads": READS,
    }
    payload["unreadable_share"] = (
        round(payload["unreadable"] / payload["asked"], 4) if payload["asked"] else None
    )
    payload["seconds_a_verdict"] = (
        round(payload["seconds"] / payload["asked"], 3) if payload["asked"] else None
    )
    payload["where"] = measurements.record(
        "grade_candidates", name or f"{Path(path).stem}_{form}", payload
    )
    return payload
