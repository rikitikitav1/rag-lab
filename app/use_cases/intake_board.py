from collections import Counter

import job_queue
from models.corpus import DataChunk, DataSource, Stage
from models.jobs import Job
from orm.sync_db import Session
from sources import files
from sqlalchemy import func, select
from use_cases import dedup, source_intake

# the jobs that move a source along its path: the removal guard's list, which names the question jobs too
_SOURCE_JOBS = source_intake.SOURCE_JOBS
_TRAIL = 12
A_PERSON, A_DOOR, THE_QUEUE = "a person", "a door", "the queue"


def _verdict(row: DataSource) -> dict:
    raw = source_intake.run_under_review(row)
    return {"verdict": raw.get("verdict"), "bad_share": raw.get("bad_share"), "reasons": raw.get("reasons") or {}}


def _top(reasons: dict, n: int = 3) -> dict:
    return dict(sorted(reasons.items(), key=lambda kv: -kv[1])[:n])


# who moves a source next and what they do, said in the words of the door; none when nothing waits
def next_step(row: DataSource, chunks: int, open_jobs: list[str], variant: str | None = None,
              said: dict | None = None) -> tuple[str | None, str]:
    if open_jobs:
        return THE_QUEUE, f"waits for its queued {', '.join(open_jobs)}"
    said = said or _verdict(row)
    if row.stage == Stage.declared:
        return A_DOOR, "onboard it (onboard_source)"
    if row.stage == Stage.raw:
        if said["verdict"] == "ok":
            return A_DOOR, "accept it (accept_source)"
        return A_PERSON, (f"a person decides on a {said['verdict']} run: accept with a reason, set a knob and onboard"
                          " again, or leave it")
    if (row.raw or {}).get("candidate"):
        return A_PERSON, f"a newer {said['verdict']} run waits beside the accepted one: accept it or leave it"
    if not chunks:
        if variant and ((row.raw or {}).get("copies_kept_by") or {}).get(variant):
            return None, "nothing waits: every chunk of it is a copy a more trusted source keeps"
        return A_DOOR, "index it into a variant (index_data)"
    if not row.active:
        return A_DOOR, "turn it on for search (set_source_active)"
    return None, "nothing waits"


def _open_jobs(session) -> dict[str, list[str]]:
    held: dict[str, list[str]] = {}
    rows = session.execute(
        select(Job.type, Job.options).where(Job.status.in_(job_queue.ACTIVE), Job.type.in_(_SOURCE_JOBS))
    )
    for kind, options in rows:
        name = (options or {}).get("source")
        held.setdefault(name if name and name != "all" else "*", []).append(kind)
    return held


# every source by stage and verdict, and the ones that wait for a person or for a door, one line each
def board(variant: str) -> dict:
    with Session() as session:
        rows = list(session.scalars(select(DataSource).order_by(DataSource.name)))
        chunks = dict(session.execute(
            select(DataChunk.source_id, func.count()).where(DataChunk.variant == variant).group_by(DataChunk.source_id)
        ).all())
        held = _open_jobs(session)
    # a whole-corpus index names no source and holds every one
    everyone = held.pop("*", [])
    by_stage: Counter = Counter()
    waiting: dict[str, list] = {A_PERSON: [], A_DOOR: [], THE_QUEUE: []}
    for r in rows:
        said = _verdict(r)
        by_stage[f"{r.stage}:{said['verdict']}"] += 1
        who, step = next_step(r, chunks.get(r.id, 0), held.get(r.name, []) + everyone, variant, said)
        if who is None:
            continue
        line = {"source": r.name, "stage": r.stage, "verdict": said["verdict"], "next": step}
        if who == A_PERSON:
            share = max((said["bad_share"] or {}).values(), default=None)
            line |= {"bad_share": share, "reasons": _top(said["reasons"])}
        waiting[who].append(line)
    return {
        "variant": variant,
        "sources": len(rows),
        "by_stage_and_verdict": dict(sorted(by_stage.items())),
        "chunks_in_variant": sum(chunks.values()),
        "waiting_for": waiting,
    }


def _job_line(job: Job) -> dict:
    result = job.result if isinstance(job.result, dict) else {}
    said = {k: result[k] for k in ("verdict", "sources", "refused", "lower_copies_dropped", "questions_written",
                                   "accepted_pairs", "unchanged") if k in result}
    error = (job.error or {}).get("message") if isinstance(job.error, dict) else job.error
    return {"id": job.id, "type": job.type, "status": job.status, "elapsed": job.elapsed,
            "at": job.updated_at.isoformat(timespec="minutes") if job.updated_at else None,
            **({"result": said} if said else {}), **({"error": str(error)[:200]} if error else {})}


# one source in a screen: where it stands, why, what its jobs said, and what waits next
def trail(name: str) -> dict | None:
    with Session() as session:
        row = session.scalar(select(DataSource).where(DataSource.name == name))
        if row is None:
            return None
        per_variant = dict(session.execute(
            select(DataChunk.variant, func.count()).where(DataChunk.source_id == row.id).group_by(DataChunk.variant)
        ).all())
        jobs = list(session.scalars(
            select(Job).where(Job.type.in_(_SOURCE_JOBS), Job.options["source"].astext == name)
            .order_by(Job.id.desc()).limit(_TRAIL)
        ))
        open_jobs = [j.type for j in jobs if j.status in job_queue.ACTIVE]
    said = _verdict(row)
    raw = source_intake.run_under_review(row)
    return {
        "name": row.name,
        "stage": row.stage,
        "active": row.active,
        "trust": dedup.source_trust(row.declaration),
        "origin": next((k for k in ("folder", "urls", "git", "git_family", "pages", "site") if (row.declaration or {})
                        .get(k)), None),
        "verdict": said["verdict"],
        "bad_share": said["bad_share"],
        "reasons": _top(said["reasons"], 6),
        "skipped_by_reason": dict(Counter(str(why) for why in (raw.get("skipped") or {}).values()).most_common(8)),
        "accepted_by": (row.raw or {}).get("accepted_by"),
        "accepted_despite": (row.raw or {}).get("accepted_despite"),
        "candidate_waits": bool((row.raw or {}).get("candidate")),
        "chunks_by_variant": per_variant,
        "drift": files.drift(row.name, row.indexed_with, row.declaration).get("moved"),
        "copies_kept_by": (row.raw or {}).get("copies_kept_by") or {},
        "jobs": [_job_line(j) for j in jobs],
        "next": next_step(row, sum(per_variant.values()), open_jobs, said=said)[1],
    }
