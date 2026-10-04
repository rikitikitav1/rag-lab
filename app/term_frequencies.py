import logging_setup
from corpus_keys import TERM_SHARE_FLOOR as SHARE_FLOOR
from orm.sync_db import Session
from sqlalchemy import text

from db import live_rows

log = logging_setup.get_logger(__name__)


# the rows search reads, active sources only: a share over inactive chunks is a share of another corpus
def _live(session, variant: str) -> tuple[int, int | None]:
    return tuple(session.execute(text(f"SELECT count(*), max(id) FROM data_chunks WHERE {live_rows()}"),
                                 {"variant": variant}).one())


# Postgres counts each lexeme of the variant's text index in seconds; the table is the variant's whole answer
def refresh(variant: str) -> dict:
    with Session() as session:
        total, newest = _live(session, variant)
        if not total:
            return {"variant": variant, "chunks": 0, "lexemes": 0}
        session.execute(text("DELETE FROM term_frequencies WHERE variant = :v"), {"v": variant})
        live = live_rows().replace(":variant", _literal(session, variant))
        stat = f"SELECT content_tsv FROM data_chunks WHERE {live}"
        kept = session.execute(text(
            "INSERT INTO term_frequencies (variant, lexeme, chunks, share, newest_chunk) "
            "SELECT :v, word, ndoc, ndoc::real / :total, :newest FROM ts_stat(:stat) WHERE ndoc >= :least"),
            {"v": variant, "total": total, "newest": newest, "stat": stat,
             "least": max(1, int(total * SHARE_FLOOR))}).rowcount
        session.execute(text(
            "INSERT INTO term_frequency_counts (variant, chunks, newest_chunk) VALUES (:v, :total, :newest) "
            "ON CONFLICT (variant) DO UPDATE SET chunks = EXCLUDED.chunks, newest_chunk = EXCLUDED.newest_chunk, "
            "computed_at = now()"), {"v": variant, "total": total, "newest": newest})
        session.commit()
    log.info("term_frequencies.refreshed", variant=variant, chunks=total, lexemes=kept)
    return {"variant": variant, "chunks": total, "lexemes": kept}


# ts_stat takes its query as text, so the variant goes in as a quoted literal rather than a bind
def _literal(session, value: str) -> str:
    return session.execute(text("SELECT quote_literal(:v)"), {"v": value}).scalar()


# what the shares were read over, named in a run's record: two counts under one switch are two settings
def counted(variant: str) -> dict | None:
    with Session() as session:
        row = session.execute(text("SELECT chunks, newest_chunk FROM term_frequency_counts WHERE variant = :v"),
                              {"v": variant}).one_or_none()
    return {"chunks": row[0], "newest_chunk": row[1]} if row else None


# a reindex, a removed source or one turned off since the count: a rare cut would read stale shares
def stale(variant: str) -> str | None:
    was = counted(variant)
    if was is None:
        return f"no term frequencies for {variant}"
    with Session() as session:
        total, newest = _live(session, variant)
    if (total, newest) != (was["chunks"], was["newest_chunk"]):
        return (f"term frequencies of {variant} read {was['chunks']} live chunks up to {was['newest_chunk']}, "
                f"the variant now serves {total} up to {newest}")
    return None
