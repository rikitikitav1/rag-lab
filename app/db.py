
import logging_setup
from corpus_keys import (
    GOLD_SQL,
    READ_BY_RUNS_SQL,
    Gold,
    exact_gold_sql,
)
from gold_match import clean_gold, heading_text
from orm.sync_db import engine
from sqlalchemy import text

log = logging_setup.get_logger(__name__)

# this variant, under a source still active: seven places wrote it and one had drifted
def live_rows(alias: str = "") -> str:
    col = f"{alias}." if alias else ""
    return f"{col}variant = :variant AND {col}source_id IN (SELECT id FROM data_sources WHERE active)"


# marks no searched chunk holds, by the stand's gold predicate (a file or a folder prefix), as the preflight reads them
def unreachable_marks(marks: list[str], *, variant: str) -> list[str]:
    if not marks:
        return []
    gold = GOLD_SQL.format(mark="m", source="dc.source")
    query = f"""SELECT m FROM unnest(CAST(:marks AS text[])) m
                WHERE NOT EXISTS (SELECT 1 FROM data_chunks dc WHERE {live_rows("dc")} AND {gold})
                ORDER BY m"""
    with engine.connect() as conn:
        return list(conn.execute(text(query), {"marks": sorted(set(marks)), "variant": variant}).scalars())


# a distinct gold `g` against a chunk `dc`, inside one query, as the reachability check and the reanchor read it
EXACT_GOLD_OF_G = exact_gold_sql("g.file", "g.section", "g.version", "dc.source", "dc.section", "dc.versions")


# questions per set whose gold no searched chunk holds, by the stand's one gold rule: a run on such a set misreads
def unreachable_by_set(*, variant: str) -> list[tuple[str | None, int]]:
    accepted = READ_BY_RUNS_SQL.format(q="q")
    # distinct golds and marks are matched once each: an EXISTS per question ran past the timeout on 800k chunks
    query = f"""WITH live AS MATERIALIZED (SELECT DISTINCT dc.source FROM data_chunks dc WHERE {live_rows("dc")}),
                marks AS MATERIALIZED (SELECT DISTINCT m FROM questions q, unnest(q.marked_sources) m WHERE {accepted}),
                held_marks AS (SELECT m FROM marks WHERE EXISTS (
                  SELECT 1 FROM live WHERE {GOLD_SQL.format(mark="m", source="live.source")})),
                golds AS MATERIALIZED (
                  SELECT DISTINCT q.gold->>'file' AS file, q.gold->>'section' AS section, q.gold->>'version' AS version
                  FROM questions q WHERE {accepted} AND q.gold IS NOT NULL),
                held_golds AS (SELECT g.* FROM golds g WHERE EXISTS (
                  SELECT 1 FROM data_chunks dc WHERE {live_rows("dc")} AND {EXACT_GOLD_OF_G}))
                SELECT set_name, count(*) FROM (
                  SELECT q.set_name FROM questions q
                  WHERE {accepted} AND cardinality(q.marked_sources) > 0
                    AND NOT EXISTS (SELECT 1 FROM unnest(q.marked_sources) m WHERE m IN (SELECT m FROM held_marks))
                  UNION ALL
                  SELECT q.set_name FROM questions q WHERE {accepted} AND q.gold IS NOT NULL AND NOT EXISTS (
                    SELECT 1 FROM held_golds h WHERE h.file = q.gold->>'file' AND h.section = q.gold->>'section'
                      AND h.version IS NOT DISTINCT FROM q.gold->>'version')
                ) unreachable GROUP BY set_name ORDER BY 2 DESC, 1"""
    with engine.connect() as conn:
        return [(name, n) for name, n in conn.execute(text(query), {"variant": variant}).all()]


def is_empty(*, variant):
    with engine.connect() as conn:
        return conn.execute(
            text(f"""SELECT NOT EXISTS (
                      SELECT 1 FROM data_chunks WHERE {live_rows()})"""),
            {"variant": variant},
        ).scalar()


# losing the bookkeeping must not cost the work: three call sites wrote this try/except
def fingerprint_or_none(*, variant) -> dict | None:
    try:
        return corpus_fingerprint(variant=variant)
    except Exception as e:
        log.error("corpus.fingerprint_failed", variant=variant, error=str(e))
        return None


# thresholds are calibrated against a corpus, so a run has to record which one it saw
def corpus_fingerprint(*, variant) -> dict:
    query = f"""
        SELECT count(*) AS chunks,
               count(DISTINCT source_id) AS sources,
               max(id) AS last_chunk_id
        FROM data_chunks WHERE {live_rows()}
    """
    with engine.connect() as conn:
        row = conn.execute(text(query), {"variant": variant}).mappings().one()
    return {"variant": variant, **dict(row)}


# `is_empty` asks whether any row exists, and a half-built variant has plenty
def sources_missing_from(*, variant) -> list[str]:
    with engine.connect() as conn:
        return [
            row[0]
            for row in conn.execute(
                text("""SELECT ds.name FROM data_sources ds
                        WHERE ds.active
                          AND NOT EXISTS (
                            SELECT 1 FROM data_chunks dc
                            WHERE dc.source_id = ds.id AND dc.variant = :variant)
                        ORDER BY ds.name"""),
                {"variant": variant},
            )
        ]


# a run on the wrong variant otherwise looks like an ordinary run with different numbers
def corpus_variants() -> list[dict]:
    query = """
        SELECT variant, count(*) AS chunks, count(DISTINCT source_id) AS sources
        FROM data_chunks GROUP BY variant ORDER BY variant
    """
    with engine.connect() as conn:
        return [dict(r) for r in conn.execute(text(query)).mappings()]


# `position()` cannot use an index, and this is constant per (variant, marked)
SECTIONS_UNDER: dict = {}


def _sections_under(conn, variant, marked: tuple[str, ...]) -> set[str]:
    from sqlalchemy import text as sql

    key = (variant, marked)
    if key in SECTIONS_UNDER:
        return SECTIONS_UNDER[key]
    rows = (
        conn.execute(
            sql(
                f"SELECT DISTINCT section FROM data_chunks"
                f" WHERE {live_rows()} AND section IS NOT NULL AND ("
                + " OR ".join(f"position(:m{i} in source) > 0" for i in range(len(marked)))
                + ")"
            ),
            {"variant": variant, **{f"m{i}": m for i, m in enumerate(marked)}},
        )
        .scalars()
        .all()
    )
    found = {heading_text(r) for r in rows}
    SECTIONS_UNDER[key] = found
    return found


# the corpus holds the gold section: an exact gold asks by its whole path, an older one by its heading's text
def section_exists(conn, variant, gold, gold_heading=None) -> bool:
    from sqlalchemy import text as sql

    gold = Gold.coerce(gold)
    if gold.exact:
        clause, params = gold.section_sql("source", "section", "versions")
        query = f"SELECT EXISTS (SELECT 1 FROM data_chunks WHERE {live_rows()} AND {clause})"
        return bool(conn.execute(sql(query), {"variant": variant, **params}).scalar())
    heading = clean_gold(gold_heading)
    if not heading:
        return False
    return heading in _sections_under(conn, variant, gold.marks)
