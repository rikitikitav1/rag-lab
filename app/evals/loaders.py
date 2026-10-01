from corpus_keys import Gold


def load_logs(run_name=None, ids=None):
    from models.eval import QuestionLog
    from orm.sync_db import Session
    from sqlalchemy import select
    from sqlalchemy.orm import selectinload

    with Session() as session:
        stmt = select(QuestionLog).options(selectinload(QuestionLog.question))
        if run_name:
            stmt = stmt.where(QuestionLog.run_name == run_name)
        if ids is not None:
            stmt = stmt.where(QuestionLog.id.in_(ids))
        # ordered: a bootstrap drawn over rows in the order postgres felt like gave two intervals
        return list(session.scalars(stmt.order_by(QuestionLog.id)))


# the population every gold door shares: a question without a marked source has no gold to keep
def gold_questions(set_name: str, limit: int | None = None, ids=None) -> list[dict]:
    from models.eval import READ_BY_RUNS, Question
    from orm.sync_db import Session
    from sqlalchemy import select

    with Session() as session:
        query = select(Question).where(Question.set_name == set_name, READ_BY_RUNS).order_by(Question.id)
        if ids:
            query = query.where(Question.id.in_(list(ids)))
        rows = [q for q in session.scalars(query).all() if Gold.of_question(q)]
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
                "gold": q.gold,
                # the source question's own text, and its own when it is not a paraphrase
                "gold_heading": headings.get(q.source_question_id) or q.original_text,
            }
            for q in rows
        ]


# each source's own reference pattern by its row's name, the sources that declare none left out
def reference_leaves() -> dict[str, str]:
    from models.corpus import DataSource
    from orm.sync_db import Session
    from sqlalchemy import select

    with Session() as session:
        found = session.execute(select(DataSource.name, DataSource.declaration)).all()
    return {name: d["reference_leaf"] for name, d in found if (d or {}).get("reference_leaf")}
