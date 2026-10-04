from collections import defaultdict

import logging_setup
from models.corpus import DataSource, Trust
from orm.sync_db import Session
from sqlalchemy import select, text

log = logging_setup.get_logger(__name__)

_RANK = {Trust.official: 0, Trust.book: 1, Trust.notes: 2}
_NOTES_READERS = frozenset({"interview", "cheatsheets", "system-design-primer"})
_BOOK_SUFFIXES = (".pdf", ".epub")


# a declared trust wins; otherwise a book is a file from the book shelf, notes are the hand-written banks
def source_trust(declaration: dict | None) -> Trust:
    declaration = declaration or {}
    if declaration.get("trust"):
        return Trust(declaration["trust"])
    if declaration.get("reader") in _NOTES_READERS or declaration.get("name") == "notes":
        return Trust.notes
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


# a body this short is a heading's echo, a "See also" or a bare fence: its meaning is the path above it, not a copy
MIN_SHARED_CHARS = 200

# the same normalisation as `corpus_keys.body_hash`: one text with other line breaks is the same text
_SHARED_WITH = """
    WITH bodies AS (
        SELECT c.id, c.source_id,
               md5(btrim(regexp_replace(substr(c.content, coalesce(c.prefix_len, 0) + 1), '\\s+', ' ', 'g'))) AS h
        FROM data_chunks c
        WHERE c.variant = :variant AND length(c.content) - coalesce(c.prefix_len, 0) >= :min_chars
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
        asked = {"variant": variant, "ids": ids, "min_chars": MIN_SHARED_CHARS}
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
