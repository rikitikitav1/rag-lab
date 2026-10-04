from collections import defaultdict

import logging_setup
from corpus_keys import SHARED_BODY_SQL
from models.corpus import DataSource, Trust
from orm.sync_db import Session
from sqlalchemy import select, text

log = logging_setup.get_logger(__name__)

_RANK = {Trust.official: 0, Trust.book: 1, Trust.notes: 2}
_BOOK_SUFFIXES = (".pdf", ".epub")


# a declared trust wins, as the notes declare theirs; otherwise a book is a file from the book shelf
def source_trust(declaration: dict | None) -> Trust:
    declaration = declaration or {}
    if declaration.get("trust"):
        return Trust(declaration["trust"])
    folder = declaration.get("folder") or ""
    urls = declaration.get("urls") or []
    if folder.startswith("datasets/inbox/books/") or any(u.lower().endswith(_BOOK_SUFFIXES) for u in urls):
        return Trust.book
    return Trust.official


# the run a source was last read by: a newer one is the fresher copy of the same text
def _freshness(row: DataSource) -> str:
    raw = row.raw or {}
    return raw.get("recorded_at") or raw.get("finished_at") or ""


# the copy that stays: most trusted, then freshest, then the name, so a rerun keeps the same one
def keeper(rows: list[DataSource]) -> DataSource:
    return max(rows, key=lambda r: (-_RANK[source_trust(r.declaration)], _freshness(r), r.name))


# copies by the hash the index stored for each body, the same key the quality report counts duplicates by
_SHARED_WITH = f"""
    WITH bodies AS (
        SELECT c.id, c.source_id, c.content_hash AS h
        FROM data_chunks c
        WHERE c.variant = :variant AND {SHARED_BODY_SQL}
    ), touched AS (
        SELECT DISTINCT h FROM bodies WHERE source_id = ANY(:ids)
    ), shared AS (
        SELECT h FROM bodies WHERE h IN (SELECT h FROM touched) GROUP BY h HAVING count(DISTINCT source_id) > 1
    )
    SELECT b.id, b.source_id, b.h FROM bodies b JOIN shared USING (h)
"""


# a shared text stays once, with the copy its trust order keeps; a loser's row names the keeper, as nothing undoes it
def drop_lower_copies(variant: str, names: list[str]) -> dict[str, dict[str, int]]:
    with Session() as session:
        ids = list(session.scalars(select(DataSource.id).where(DataSource.name.in_(names))))
        if not ids:
            return {}
        asked = {"variant": variant, "ids": ids}
        found = session.execute(text(_SHARED_WITH), asked).all()
        if not found:
            return {}
        holders = {s for _, s, _ in found}
        sources = {r.id: r for r in session.scalars(select(DataSource).where(DataSource.id.in_(holders)))}
        by_hash: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for chunk_id, source_id, h in found:
            by_hash[h].append((chunk_id, source_id))
        doomed: list[int] = []
        dropped: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for held in by_hash.values():
            kept = keeper([sources[s] for s in {s for _, s in held}])
            for chunk_id, source_id in held:
                if source_id != kept.id:
                    doomed.append(chunk_id)
                    dropped[sources[source_id].name][kept.name] += 1
        session.execute(text("DELETE FROM data_chunks WHERE id = ANY(:ids)"), {"ids": doomed})
        for name, keepers in dropped.items():
            row = next(r for r in sources.values() if r.name == name)
            copies = {**((row.raw or {}).get("copies_kept_by") or {}), variant: dict(keepers)}
            row.raw = {**(row.raw or {}), "copies_kept_by": copies}
        session.commit()
    out = {name: dict(keepers) for name, keepers in dropped.items()}
    log.info("dedup.lower_copies_dropped", variant=variant, dropped=out)
    return out
