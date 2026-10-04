import logging_setup
from corpus_keys import READ_BY_RUNS_SQL
from errors import Final
from models.eval import Question, text_hash
from orm.sync_db import Session
from sqlalchemy import func, select, text
from sqlalchemy.dialects.postgresql import insert as pg_insert

log = logging_setup.get_logger(__name__)


# a question reworded by hand asks what its accepted original asks, so it keeps that gold and needs no new acceptance
def load(set_name: str, rows: list[dict]) -> dict:
    with Session() as session:
        if session.scalar(select(func.count()).where(Question.set_name == set_name)):
            raise Final(f"set {set_name} exists; variants go into a new set, never onto one a run has read")
        ids = sorted({int(r["source_question_id"]) for r in rows})
        accepted = {q.id: q for q in session.scalars(
            select(Question).where(Question.id.in_(ids), text(READ_BY_RUNS_SQL.format(q="questions"))))}
        if missing := [i for i in ids if i not in accepted]:
            raise Final(f"{len(missing)} originals are not accepted questions of the stand: {missing[:20]}")
        written, collided = 0, 0
        for r in rows:
            original = accepted[int(r["source_question_id"])]
            inserted = session.scalar(pg_insert(Question).values(
                text_hash=text_hash(r["original_text"]), original_text=r["original_text"], set_name=set_name,
                language=original.language, kind=original.kind, marked_sources=original.marked_sources,
                gold=original.gold, reference_answer=original.reference_answer, anchors=original.anchors,
                source_question_id=original.id,
            ).on_conflict_do_nothing(index_elements=["text_hash"]).returning(Question.id))
            written += inserted is not None
            collided += inserted is None
        session.commit()
    log.info("question_variants.loaded", set_name=set_name, written=written, collided=collided)
    return {"set_name": set_name, "questions_written": written, "texts_already_held": collided}
