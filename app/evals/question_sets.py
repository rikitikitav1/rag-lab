"""What each question set carries, and therefore which axes a run over it can be scored on."""

import collections

from evals.pools import POOLS, kind_of_question
from models.eval import Question
from orm.sync_db import Session
from sqlalchemy import select
from sqlalchemy.orm import defer


def _of(questions: list) -> dict:
    pools = collections.Counter(kind_of_question(q) for q in questions)
    languages = collections.Counter(q.language or "unknown" for q in questions)
    return {
        "questions": len(questions),
        "pools": {pool: pools.get(pool, 0) for pool in POOLS if pools.get(pool)},
        "languages": dict(sorted(languages.items())),
        # the corpus pool, and what every retrieval axis ranks against
        "with_marked_sources": sum(1 for q in questions if q.marked_sources),
        # a question of the exact kind: a file, its section path and version
        "with_exact_gold": sum(1 for q in questions if getattr(q, "gold", None)),
        # what the guest context axes need; without it they abstain and only faithfulness scores
        "with_reference_answer": sum(1 for q in questions if q.reference_answer),
        "paraphrases": sum(1 for q in questions if q.source_question_id),
    }


def inventory(set_name: str | None = None) -> list[dict]:
    with Session() as session:
        stmt = select(Question)
        if set_name:
            stmt = stmt.where(Question.set_name == set_name)
        rows = list(session.scalars(stmt))
    by_set = collections.defaultdict(list)
    for q in rows:
        by_set[q.set_name or "unnamed"].append(q)
    return [
        {"set_name": name, **_of(questions)}
        for name, questions in sorted(by_set.items(), key=lambda kv: -len(kv[1]))
    ]


# the rows themselves, pooled by the rule `_of` counts with, so a list and its set's counts agree
def rows(set_name: str | None = None, language: str | None = None, pool: str | None = None,
         limit: int = 100, offset: int = 0) -> list[dict]:
    stmt = select(Question).options(defer(Question.embedding)).order_by(Question.id)
    if set_name:
        stmt = stmt.where(Question.set_name == set_name)
    if language:
        stmt = stmt.where(Question.language == language)
    if not pool:
        stmt = stmt.offset(offset).limit(limit)
    with Session() as session:
        found = list(session.scalars(stmt))
    if pool:
        found = [q for q in found if kind_of_question(q) == pool][offset:offset + limit]
    return [_row(q) for q in found]


def _row(q) -> dict:
    return {
        "id": q.id,
        "set_name": q.set_name,
        "language": q.language,
        "pool": kind_of_question(q),
        "text": q.original_text,
        "has_reference": bool(q.reference_answer),
        "marked_sources": len(q.marked_sources or []),
        "gold": getattr(q, "gold", None),
        "embedded_by": q.embedded_by,
        "paraphrase_of": q.source_question_id,
    }


# the jobs that read a set by name
SET_JOBS = ("eval_run", "paraphrase_questions", "build_veto_set")


def removal_refusal(set_name: str, holds: dict, named_in_config: bool, queued: int | None) -> str | None:
    if not holds["questions"]:
        return f"no question set named {set_name}"
    if named_in_config:
        return f"{set_name} is named in config/evals.yaml"
    if queued:
        return f"{set_name} has job {queued} queued or running"
    if holds["answered"]:
        return f"{holds['answered']} answer logs hold questions of {set_name}"
    if holds["drawn_from"]:
        return f"{holds['drawn_from']} questions of other sets are drawn from {set_name}"
    return None


def remove(set_name: str) -> dict:
    import config
    import job_queue
    from errors import Final

    import db

    verdict = config.settings.verdict
    named = set_name in {*verdict.criterion_sets, *verdict.veto_sets}
    # a job may name its set by default or read it through another key, so any job that reads sets holds the door
    queued = next((j for t in (*SET_JOBS, "embed_questions") if (j := job_queue.pending_of_type(t))), None)
    if refusal := removal_refusal(set_name, db.question_set_holds(set_name), named, queued):
        raise Final(refusal)
    return {"set_name": set_name, "questions": db.remove_question_set(set_name)}
