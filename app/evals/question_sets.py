"""What each question set carries, and therefore which axes a run over it can be scored on."""

import collections
import json
from types import SimpleNamespace

import config
from evals import columns, measurements, pair_judge
from evals.pools import POOLS, kind_of_question
from models.eval import ACCEPTED, REFUSED, Question
from orm.sync_db import Session
from sqlalchemy import select
from sqlalchemy.orm import defer


def _of(questions: list) -> dict:
    pools = collections.Counter(kind_of_question(q) for q in questions)
    languages = collections.Counter(q.language or "unknown" for q in questions)
    return {
        "questions": len(questions),
        # what a run over the set reads; the rest wait for acceptance or were settled out
        "read_by_runs": sum(1 for q in questions if q.status == ACCEPTED),
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


_JUDGE_SAID = {pair_judge.NOT_ANSWERED, pair_judge.SPLIT, pair_judge.UNREAD, pair_judge.NOT_FOUND}


# a generated set's way from the generator to acceptance: what was asked and lost, then who settled what, by language
def stages(set_name: str, questions: list) -> dict | None:
    paired = [q for q in questions if q.pair_id]
    if not paired:
        return None
    asked, kept, lost = 0, 0, collections.Counter()
    for path in measurements.recorded("question_set", set_name):
        report = json.loads(path.read_text())
        if report.get("set_name") != set_name:
            continue
        asked += report.get("pairs_asked") or 0
        kept += report.get("pairs_kept") or 0
        lost.update(report.get("pairs_lost_by_reason") or {})
    reparsed = 0
    for path in measurements.recorded("question_reparse", set_name):
        report = json.loads(path.read_text())
        reparsed += report.get("pairs_kept_now", 0) if report.get("set_name") == set_name else 0
    by_language = {}
    for lang in sorted({q.language for q in paired if q.language}):
        mine = [q for q in paired if q.language == lang]
        refused = [q for q in mine if q.status == REFUSED]
        # the columns a run's slices read, read here the same way
        anchored = [columns.read("anchored_by_identifier", SimpleNamespace(question=q)) for q in mine]
        by_language[lang] = {
            "status": dict(collections.Counter(str(q.status) for q in mine)),
            "refused_by_judge": sum(1 for q in refused if q.acceptance_why in _JUDGE_SAID),
            "refused_by_sieve": sum(1 for q in refused if q.acceptance_why not in _JUDGE_SAID),
            "anchored": sum(1 for v in anchored if v == 1.0),
            "anchors_unread": sum(1 for v in anchored if v is None),
            "shares_heading_word": sum(
                1 for q in mine if columns.read("shares_heading_word", SimpleNamespace(question=q)) == 1.0
            ),
        }
    accepted_pairs = len({q.pair_id for q in paired if q.status == ACCEPTED})
    return {
        "generated": {"pairs_asked": asked, "pairs_kept": kept, "pairs_lost_by_reason": dict(lost)},
        # replies read again by later checks: kept then, so the generated and the set's own count differ by these
        "reparsed_pairs": reparsed,
        "pairs_in_set": len({q.pair_id for q in paired}),
        "accepted_pairs": accepted_pairs,
        "by_language": by_language,
        "under_the_floor": accepted_pairs < config.settings.evals.question_set.min_pairs,
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
    out = []
    for name, questions in sorted(by_set.items(), key=lambda kv: -len(kv[1])):
        row = {"set_name": name, **_of(questions)}
        if (way := stages(name, questions)) is not None:
            row["stages"] = way
        out.append(row)
    return out


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
