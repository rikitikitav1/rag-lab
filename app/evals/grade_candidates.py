"""The grader over frozen candidates: no generator, no judge, one verdict per question and chunk."""

import json
import random
import time
from pathlib import Path

import job_queue
import logging_setup
import prompt_repo
from errors import StandFault
from evals import gold_classes, grade_curve, measurements
from models.registry import Purpose
from use_cases import grading

log = logging_setup.get_logger(__name__)

SCHEMA = 2
FORMS = ("per_chunk", "whole_text")

READS = (
    "verdicts are per question and chunk, taken once; retention is read on the rows whose gold"
    " section reached the population, the floor on strangers, and a neighbour section of the gold"
    " file is neither; `p` is the probability the model gave the word it said, so a cut of the"
    " share can be moved over the record without asking the model again; the curve of both arms"
    " is computed here, so the closing number carries the id of the job that asked the questions"
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
        started = time.perf_counter()
        said = grading.piece_verdict(row["text"], "\n\n".join(pieces), system, ask, "whole")
        kept = list(range(len(chunks))) if said != "no" else []
        # the same keys `grade_pieces` returns, `seconds` included: the two forms are compared on price
        return {"kept": kept, "asked": 1, "unreadable": int(said is None), "order": addresses,
                "dropped": [] if kept else addresses,
                "seconds": round(time.perf_counter() - started, 3)}
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


# everything the ask recorded but the stage: a row that copies three names by hand loses the fourth
def verdict_row(ask: dict) -> dict:
    return {key: value for key, value in ask.items() if key != "stage"}


# both arms at once: a curve computed afterwards by hand carries no job behind it
def curves(payload: dict, frozen: dict) -> dict:
    return {arm: grade_curve.curve(payload, frozen, arm) for arm in grade_curve.ARMS}


def run(path: str, form: str = "per_chunk", top: int = 5, limit: int | None = None,
        sample: int | None = None, seed: int = 0, name: str | None = None,
        shuffle: int | None = None, prompt_version: int | None = None,
        job_id: int | None = None) -> dict:
    if form not in FORMS:
        raise StandFault(f"a call form is one of {FORMS}, got {form!r}")
    frozen = json.loads(Path(path).read_text())
    variant = frozen["stamp"]["variant"]
    rows = drawn(frozen["rows"], sample, seed, limit)
    if shuffle is not None:
        # the same rows in another order: a floor taken twice in one residency must move the prefix cache
        rows = list(rows)
        random.Random(shuffle).shuffle(rows)
    # a measured arm names its version: activating one to measure it makes the stand serve an unjudged prompt
    version = prompt_version or prompt_repo.active_version(Purpose.grade_chunk)
    system = grading.system_prompt(prompt_version)
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
            "verdicts": [verdict_row(a) for a in asks],
        })
        asks_all += asks
        # the dial is the probability: without it every cut keeps everything and the curve is a line
        if nth == 1 and asks and not any("p" in a for a in asks):
            raise StandFault(
                "the engine returned no logprobs for the first question, so no cut can be read;"
                " a curve over such a pass is a straight line"
            )
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
        "prompt": {"purpose": str(Purpose.grade_chunk), "version": version},
        "questions": len(out),
        "cancelled": cancelled,
        "asked": sum(r["asked"] for r in out),
        "unreadable": sum(r["unreadable"] for r in out),
        "seconds": round(time.perf_counter() - started, 1),
        "rows": out,
        "reads": READS,
    }
    payload["without_probability"] = sum(1 for a in asks_all if "p" not in a)
    payload["unreadable_share"] = (
        round(payload["unreadable"] / payload["asked"], 4) if payload["asked"] else None
    )
    payload["seconds_a_verdict"] = (
        round(payload["seconds"] / payload["asked"], 3) if payload["asked"] else None
    )
    payload["curves"] = curves(payload, frozen) if out else {}
    # the verdicts go beside the file, not into it: five thousand rows are read by a program, not a person
    payload["where"] = measurements.record(
        "grade_candidates", name or f"{Path(path).stem}_{form}", payload, bulk=("rows",)
    )
    return payload
