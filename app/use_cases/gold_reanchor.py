from collections import Counter

from corpus_keys import READ_BY_RUNS_SQL, SECTION_SEP, leaf_of, spaceless_key
from models.eval import READ_BY_RUNS, Question
from orm.sync_db import Session, engine
from sqlalchemy import select, text

import db


# accepted golds of a set no searched chunk holds now, matched once each as the reachability check does
def _unheld(conn, set_name: str, variant: str) -> list[tuple[str, str, str | None]]:
    rows = conn.execute(text(f"""
        WITH golds AS MATERIALIZED (
          SELECT DISTINCT gold->>'file' AS file, gold->>'section' AS section, gold->>'version' AS version
          FROM questions q WHERE q.set_name = :set AND {READ_BY_RUNS_SQL.format(q='q')} AND q.gold IS NOT NULL)
        SELECT g.file, g.section, g.version FROM golds g
        WHERE NOT EXISTS (SELECT 1 FROM data_chunks dc WHERE {db.live_rows("dc")} AND {db.EXACT_GOLD_OF_G})"""),
        {"set": set_name, "variant": variant})
    return [tuple(r) for r in rows]


def _sections_of(conn, file: str, version: str | None, variant: str) -> dict[str, list[str]]:
    rows = conn.execute(text(f"""
        SELECT dc.section, dc.content FROM data_chunks dc
        WHERE {db.live_rows("dc")} AND dc.source = :file
          AND (CAST(:version AS text) IS NULL OR cardinality(dc.versions) = 0 OR :version = ANY(dc.versions))"""),
        {"file": file, "version": version, "variant": variant})
    out: dict[str, list[str]] = {}
    for section, content in rows:
        out.setdefault(section or "", []).append(spaceless_key(content or ""))
    return out


# the same heading under a new root: a cleanup that renames a page's root leaves its leaves where they were
def _candidates(old: str, sections: dict[str, list[str]]) -> list[str]:
    if SECTION_SEP not in old:
        return [s for s in sections if s and SECTION_SEP not in s]
    return [s for s in sections if SECTION_SEP in s and leaf_of(s) == leaf_of(old)]


# a candidate holds the question's evidence; where two share a leaf, it decides between them
def _by_evidence(candidates: list[str], sections: dict[str, list[str]], evidence: list[str]) -> list[str]:
    keys = [spaceless_key(e) for e in evidence if e]
    return [s for s in candidates if any(k and k in body for k in keys for body in sections[s])]


# golds whose section the index no longer names are moved to the section of the same file and leaf
def reanchor(set_name: str, variant: str, dry: bool = False) -> dict:
    counted, moved = Counter(), []
    with engine.connect() as conn, Session() as session:
        for file, section, version in _unheld(conn, set_name, variant):
            questions = [q for q in session.scalars(select(Question).where(
                Question.set_name == set_name, READ_BY_RUNS,
                Question.gold["file"].astext == file, Question.gold["section"].astext == section))
                if (q.gold or {}).get("version") == version]
            sections = _sections_of(conn, file, version, variant)
            if not sections:
                counted["file_gone"] += len(questions)
                continue
            # the evidence must sit there even for one candidate: a removed section shares its leaf with another
            found = _by_evidence(_candidates(section, sections), sections, [q.evidence for q in questions])
            if len(found) != 1:
                counted["leaf_gone" if not found else "ambiguous"] += len(questions)
                continue
            moved.append({"file": file, "from": section, "to": found[0], "questions": len(questions)})
            counted["moved"] += len(questions)
            if not dry:
                for q in questions:
                    q.gold = {**q.gold, "section": found[0]}
        if not dry:
            session.commit()
    return {"set_name": set_name, "variant": variant, "dry": dry, **counted, "sections": moved}
