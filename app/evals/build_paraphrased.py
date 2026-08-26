import hashlib
import os
import sys

import llm
import logging_setup
import prompt_repo
from models.eval import Question
from models.registry import Purpose
from orm.sync_db import Session
from sqlalchemy import String as sa_text_type
from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert

log = logging_setup.get_logger(__name__)

SOURCE_SET = "interview"
PARAPHRASE_ROLE = "paraphrasing"


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _paraphrase(text: str) -> str:
    return llm.ask(
        system=prompt_repo.active_template(Purpose.paraphrase_question),
        user=text,
        role=PARAPHRASE_ROLE,
    ).text.strip()


def _translate_ru(text: str) -> str:
    return llm.ask(
        system=prompt_repo.active_template(Purpose.translate_question),
        user=text,
        role=PARAPHRASE_ROLE,
    ).text.strip()


def _insert(session, text, set_name, language, original) -> bool:
    row = {
        "text_hash": _text_hash(text),
        "original_text": text,
        "set_name": set_name,
        "language": language,
        "kind": original.kind,
        "marked_sources": original.marked_sources,
        "reference_answer": original.reference_answer,
        "source_question_id": original.id,
    }
    inserted = session.scalar(
        pg_insert(Question)
        .values(row)
        .on_conflict_do_nothing(index_elements=["text_hash"])
        .returning(Question.id)
    )
    return inserted is not None


# md5 over id and seed: reproducible, and unlike random() it survives a rebuild of the set
def _pick(session, limit: int | None, source: str | None, seed: str, per_source: int | None):
    stmt = select(Question).where(Question.set_name == SOURCE_SET)
    if source:
        stmt = stmt.where(
            func.array_to_string(Question.marked_sources, " ").ilike(f"%{source}%")
        )
    order = func.md5(func.concat(func.cast(Question.id, sa_text_type), seed))
    if per_source is None:
        return session.scalars(stmt.order_by(order).limit(limit)).all()
    # limit after stratification would leave part of the sources with fewer than asked

    ranked_where = [Question.set_name == SOURCE_SET]
    if source:
        ranked_where.append(
            func.array_to_string(Question.marked_sources, " ").ilike(f"%{source}%")
        )
    ranked = (
        select(
            Question,
            func.row_number()
            .over(partition_by=Question.marked_sources[1], order_by=order)
            .label("rank"),
        )
        .where(*ranked_where)
        .subquery()
    )
    rows = session.execute(
        select(Question)
        .join(ranked, Question.id == ranked.c.id)
        .where(ranked.c.rank <= per_source)
        .order_by(func.md5(func.concat(func.cast(Question.id, sa_text_type), seed)))
    ).scalars().all()
    return rows


def build(
    limit: int | None,
    source: str | None = None,
    set_name: str = "paraphrased",
    seed: str = "",
    per_source: int | None = None,
) -> dict:
    # an unseeded set cannot be rebuilt, and a criterion set has to be
    if not seed and not set_name.startswith("test"):
        raise ValueError(f"set '{set_name}' needs a seed to be reproducible")
    ru_set = f"{set_name}_ru"
    made = {set_name: 0, ru_set: 0}
    with Session() as session:
        originals = _pick(session, limit, source, seed, per_source)
        log.info("paraphrase.picked", n=len(originals), seed=seed, per_source=per_source)

        for i, original in enumerate(originals, 1):
            rephrased = _paraphrase(original.original_text)
            if not rephrased:
                continue
            if _insert(session, rephrased, set_name, "eng", original):
                made[set_name] += 1

            translated = _translate_ru(rephrased)
            if translated and _insert(session, translated, ru_set, "rus", original):
                made[ru_set] += 1

            if i % 20 == 0:
                session.commit()
                log.info("paraphrase.progress", done=i, made=made)
        session.commit()
    log.info("paraphrase.done", requested=limit, made=made)
    return made


if __name__ == "__main__":
    logging_setup.configure(os.getenv("LOG_LEVEL", "INFO"))
    limit = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1] != "-" else None
    source = sys.argv[2] if len(sys.argv) > 2 else None
    set_name = sys.argv[3] if len(sys.argv) > 3 else "paraphrased"
    seed = sys.argv[4] if len(sys.argv) > 4 else ""
    per_source = int(sys.argv[5]) if len(sys.argv) > 5 else None
    print("made:", build(limit, source, set_name, seed, per_source))
