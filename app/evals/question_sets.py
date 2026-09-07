"""What each question set carries, and therefore which axes a run over it can be scored on.

Written after an hour of card was spent scoring guests on two pools that hold no reference answer:
the two context axes could not have produced a number, and the set knew that before the run did.
"""

import collections

from evals.pools import POOLS, kind_of_question
from models.eval import Question
from orm.sync_db import Session
from sqlalchemy import select


def _of(questions: list) -> dict:
    pools = collections.Counter(kind_of_question(q) for q in questions)
    languages = collections.Counter(q.language or "unknown" for q in questions)
    return {
        "questions": len(questions),
        "pools": {pool: pools.get(pool, 0) for pool in POOLS if pools.get(pool)},
        "languages": dict(sorted(languages.items())),
        # the corpus pool, and what every retrieval axis ranks against
        "with_marked_sources": sum(1 for q in questions if q.marked_sources),
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
