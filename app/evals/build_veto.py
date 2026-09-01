import hashlib
import re

import llm
import logging_setup
import prompt_repo
from models.eval import Question, text_hash
from models.registry import Purpose
from orm.sync_db import Session
from sqlalchemy import select, text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from use_cases.retrieval_compare import clean_gold, heading_text

import db

log = logging_setup.get_logger(__name__)

ORIGINALS = "veto_headings"
ROLE = "paraphrasing"
# `redis-doc/commands` is out: a question made from its file stem is a label
FAMILIES = {
    "cheatsheets": "cheatsheets/",
    "redis-doc/docs": "redis-doc/docs/",
    "notes": "notes/",
    "system-design-primer": "system-design-primer/",
}
QUOTAS = {
    "cheatsheets": 80,
    "redis-doc/docs": 80,
    "notes": 80,
    "system-design-primer": 30,
}
# shorter than this and the heading is a label rather than a subject: `Usage`, `101`, `fs`
MIN_HEADING = 12


def _family_of(source: str) -> str | None:
    for name, prefix in FAMILIES.items():
        if source.startswith(prefix):
            return name
    return None


# only files every variant holds, or the question measures corpus drift, not the cut
def _shared_sources(session, variants: list[str]) -> set[str]:
    shared: set[str] | None = None
    for variant in variants:
        rows = set(
            session.scalars(
                text(f"SELECT DISTINCT source FROM data_chunks WHERE {db.live_rows()}"),
                {"variant": variant},
            )
        )
        shared = rows if shared is None else shared & rows
    return shared or set()


def _headings(session, variant: str) -> list[tuple[str, str, str]]:
    rows = session.execute(
        text(f"""
            SELECT DISTINCT source, section, language
            FROM data_chunks
            WHERE {db.live_rows()} AND section IS NOT NULL AND section <> ''
        """),
        {"variant": variant},
    ).all()
    return [(r[0], r[1], r[2]) for r in rows]


# `rank_of_section` strips a numeric prefix on one side, so the stripped form is stored
def _leaf(section: str) -> str:
    written = (section or "").split(" > ")[-1].strip()
    return re.sub(r"^\d+\.\s*", "", written).strip()


def candidates(variants: list[str], cut_from: str) -> list[dict]:
    with Session() as session:
        shared = _shared_sources(session, variants)
        rows = _headings(session, cut_from)
        # including this set's own originals made a second run a silent no-op
        taken = set(
            session.scalars(
                select(Question.text_hash).where(
                    (Question.set_name != ORIGINALS) | Question.set_name.is_(None)
                )
            )
        )

    seen: dict[str, list[dict]] = {}
    for source, section, language in rows:
        family = _family_of(source)
        if family is None or source not in shared:
            continue
        leaf = _leaf(section)
        if len(leaf) < MIN_HEADING:
            continue
        key = clean_gold(leaf)
        # the gold has to identify one file: a shared heading goes to whichever ranked first
        seen.setdefault(key, []).append(
            {"family": family, "source": source, "heading": leaf, "language": language}
        )
    picked = []
    for key, found in sorted(seen.items()):
        if len({row["source"] for row in found}) != 1:
            continue
        # `_headings` is a DISTINCT with no ORDER BY, so row order decided, not the seed
        row = min(found, key=lambda r: (r["source"], r["heading"]))
        # a heading already naming a question elsewhere is dropped by the unique index
        if text_hash(row["heading"]) in taken:
            continue
        # the same check the matcher will make, made now rather than after the card is spent
        if key != heading_text(row["heading"]):
            continue
        picked.append(row)
    return picked


def _order_key(row: dict, seed: str) -> str:
    return hashlib.md5(
        f"{row['source']}\x00{row['heading']}\x00{seed}".encode(), usedforsecurity=False
    ).hexdigest()


# resolved once and written down: a restarted job must not redecide its scope
def plan(seed: str, variants: list[str], cut_from: str, quotas: dict | None = None) -> list[dict]:
    quotas = quotas or QUOTAS
    rows = candidates(variants, cut_from)
    out = []
    for family, quota in sorted(quotas.items()):
        of_family = sorted(
            (r for r in rows if r["family"] == family), key=lambda r: _order_key(r, seed)
        )
        if len(of_family) < quota:
            log.warning(
                "veto.short_of_quota", family=family, asked=quota, have=len(of_family)
            )
        out += of_family[:quota]
    return out


def _ask(row: dict) -> str:
    language = "Russian" if row["language"] == "rus" else "English"
    return llm.ask(
        system=prompt_repo.active_template(Purpose.question_from_heading),
        user=f"File: {row['source']}\nHeading: {row['heading']}\nTarget language: {language}",
        role=ROLE,
    ).text.strip()


def _add_question(session, values: dict) -> int | None:
    return session.scalar(
        pg_insert(Question)
        .values(values)
        .on_conflict_do_nothing(index_elements=["text_hash"])
        .returning(Question.id)
    )


def _originals(session, rows: list[dict]) -> dict[str, int]:
    ids = {}
    for row in rows:
        digest = text_hash(row["heading"])
        made = _add_question(
            session,
            {
                "text_hash": digest,
                "original_text": row["heading"],
                "set_name": ORIGINALS,
                "language": row["language"],
                "kind": "heading",
                "marked_sources": [row["source"]],
            },
        )
        ids[digest] = made or session.scalar(
            select(Question.id).where(Question.text_hash == digest)
        )
    session.commit()
    return ids


def _already_asked(session, set_name: str) -> set[int]:
    return set(
        session.scalars(
            select(Question.source_question_id).where(
                Question.set_name == set_name, Question.source_question_id.isnot(None)
            )
        )
    )


def build(
    seed: str,
    set_name: str = "veto_v1",
    variants: list[str] | None = None,
    cut_from: str = "clean_1024",
    quotas: dict | None = None,
) -> dict:
    if not seed:
        raise ValueError(f"set '{set_name}' needs a seed to be reproducible")
    variants = variants or ["baseline", cut_from]
    rows = plan(seed, variants, cut_from, quotas)
    counted = {"planned": len(rows), "asked": 0, "written": 0}
    with Session() as session:
        ids = _originals(session, rows)
        done = _already_asked(session, set_name)
        for i, row in enumerate(rows, 1):
            original = ids[text_hash(row["heading"])]
            if original in done:
                continue
            asked = _ask(row)
            counted["asked"] += 1
            if not asked:
                continue
            if _add_question(
                session,
                {
                    "text_hash": text_hash(asked),
                    "original_text": asked,
                    "set_name": set_name,
                    "language": row["language"],
                    "kind": "veto",
                    "marked_sources": [row["source"]],
                    "source_question_id": original,
                },
            ):
                counted["written"] += 1
            if i % 20 == 0:
                session.commit()
                log.info("veto.progress", done=i, **counted)
        session.commit()
    log.info("veto.done", set_name=set_name, seed=seed, **counted)
    return counted
