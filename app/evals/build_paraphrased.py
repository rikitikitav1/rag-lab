import hashlib
import os
import sys

import llm
import logging_setup
import prompt_repo
from models.eval import Question
from models.registry import Purpose
from orm.sync_db import Session
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


def build(limit: int, source: str | None = None, set_name: str = "paraphrased") -> dict:
    ru_set = f"{set_name}_ru"
    made = {set_name: 0, ru_set: 0}
    with Session() as session:
        stmt = select(Question).where(Question.set_name == SOURCE_SET)
        if source:
            stmt = stmt.where(
                func.array_to_string(Question.marked_sources, " ").ilike(f"%{source}%")
            )
        originals = session.scalars(
            stmt.order_by(func.random()).limit(limit)
        ).all()

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
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    source = sys.argv[2] if len(sys.argv) > 2 else None
    set_name = sys.argv[3] if len(sys.argv) > 3 else "paraphrased"
    print("made:", build(limit, source, set_name))
