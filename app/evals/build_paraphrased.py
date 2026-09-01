import os
import sys

import llm
import logging_setup
import prompt_repo
from evals import sampling
from models.eval import Question, text_hash
from models.registry import Purpose
from orm.sync_db import Session
from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert

log = logging_setup.get_logger(__name__)

SOURCE_SET = "interview"
PARAPHRASE_ROLE = "paraphrasing"


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
        "text_hash": text_hash(text),
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
def _pick(
    session,
    limit: int | None,
    source: str | None,
    seed: str,
    per_source: int | None,
    grow_set: str | None = None,
):
    # an original counts as used only when both halves of the pair exist
    used = (
        select(Question.source_question_id)
        .where(Question.set_name == grow_set, Question.source_question_id.isnot(None))
        .intersect(
            select(Question.source_question_id).where(
                Question.set_name == f"{grow_set}_ru",
                Question.source_question_id.isnot(None),
            )
        )
        .scalar_subquery()
        if grow_set
        else None
    )
    # one list of conditions for both paths: only one of the two copies would be changed
    where = [Question.set_name == SOURCE_SET]
    if used is not None:
        where.append(Question.id.not_in(used))
    if source:
        where.append(
            func.array_to_string(Question.marked_sources, " ").ilike(f"%{source}%")
        )
    order = sampling.by_id_and_seed(Question.id, seed)
    if per_source is None:
        return session.scalars(
            select(Question).where(*where).order_by(order).limit(limit)
        ).all()
    # limit after stratification would leave part of the sources with fewer than asked
    ranked = (
        select(
            Question,
            func.row_number()
            .over(partition_by=Question.marked_sources[1], order_by=order)
            .label("rank"),
        )
        .where(*where)
        .subquery()
    )
    return session.scalars(
        select(Question)
        .join(ranked, Question.id == ranked.c.id)
        .where(ranked.c.rank <= per_source)
        .order_by(order)
    ).all()


# resolved once so it can be written down: a restarted job must not redecide its scope
def plan(
    limit: int | None,
    source: str | None = None,
    set_name: str = "paraphrased",
    seed: str = "",
    per_source: int | None = None,
    grow: bool = False,
) -> list[int]:
    with Session() as session:
        picked = _pick(
            session, limit, source, seed, per_source,
            grow_set=set_name if grow else None,
        )
        return [q.id for q in picked]


# per set: a run that died between the two rows left it done in one and absent in the other
def _already_done(session, set_name: str) -> dict[int, str]:
    return dict(
        session.execute(
            select(Question.source_question_id, Question.original_text).where(
                Question.set_name == set_name,
                Question.source_question_id.isnot(None),
            )
        ).all()
    )


def build(
    limit: int | None,
    source: str | None = None,
    set_name: str = "paraphrased",
    seed: str = "",
    per_source: int | None = None,
    grow: bool = False,
    originals: list[int] | None = None,
) -> dict:
    # an unseeded set cannot be rebuilt, and a criterion set has to be
    if not seed and not set_name.startswith("test"):
        raise ValueError(f"set '{set_name}' needs a seed to be reproducible")
    ru_set = f"{set_name}_ru"
    made = {set_name: 0, ru_set: 0}
    with Session() as session:
        if originals is not None:
            # the rows come back in the list's order, or a half-finished run is not a prefix of it
            query = select(Question).where(Question.id.in_(originals))
            if originals:
                query = query.order_by(func.array_position(originals, Question.id))
            picked = session.scalars(query).all()
            log.info("paraphrase.given", n=len(picked), of=len(originals))
        else:
            picked = _pick(
                session, limit, source, seed, per_source,
                grow_set=set_name if grow else None,
            )
            log.info("paraphrase.picked", n=len(picked), seed=seed, per_source=per_source)
        originals = picked
        # a fresh paraphrase is new text with a new hash, so nothing would stop the double
        done_en = _already_done(session, set_name)
        done_ru = _already_done(session, ru_set)
        originals = [q for q in originals if not (q.id in done_en and q.id in done_ru)]
        if done_en or done_ru:
            log.info(
                "paraphrase.resuming", left=len(originals),
                done_en=len(done_en), done_ru=len(done_ru),
            )

        for i, original in enumerate(originals, 1):
            # a half that exists is the pair: derived from it, never from a fresh paraphrase
            stored = done_en.get(original.id)
            if original.id in done_ru and stored is None:
                log.warning("paraphrase.orphan_ru", original=original.id)
                continue
            rephrased = stored or _paraphrase(original.original_text)
            if not rephrased:
                continue
            if original.id not in done_en and _insert(
                session, rephrased, set_name, "eng", original
            ):
                made[set_name] += 1

            if original.id in done_ru:
                continue
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
