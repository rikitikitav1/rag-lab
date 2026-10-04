import logging_setup
from corpus_keys import TERM_SHARE_FLOOR as SHARE_FLOOR
from orm.sync_db import Session
from sqlalchemy import text

log = logging_setup.get_logger(__name__)



# Postgres counts each lexeme of the variant's text index in seconds; the table is the variant's whole answer
def refresh(variant: str) -> dict:
    with Session() as session:
        total, newest = session.execute(text("SELECT count(*), max(id) FROM data_chunks WHERE variant = :v"),
                                        {"v": variant}).one()
        if not total:
            return {"variant": variant, "chunks": 0, "lexemes": 0}
        session.execute(text("DELETE FROM term_frequencies WHERE variant = :v"), {"v": variant})
        stat = "SELECT content_tsv FROM data_chunks WHERE variant = " + _literal(session, variant)
        kept = session.execute(text(
            "INSERT INTO term_frequencies (variant, lexeme, chunks, share, newest_chunk) "
            "SELECT :v, word, ndoc, ndoc::real / :total, :newest FROM ts_stat(:stat) WHERE ndoc >= :least"),
            {"v": variant, "total": total, "newest": newest, "stat": stat,
             "least": max(1, int(total * SHARE_FLOOR))}).rowcount
        session.commit()
    log.info("term_frequencies.refreshed", variant=variant, chunks=total, lexemes=kept)
    return {"variant": variant, "chunks": total, "lexemes": kept}


# ts_stat takes its query as text, so the variant goes in as a quoted literal rather than a bind
def _literal(session, value: str) -> str:
    return session.execute(text("SELECT quote_literal(:v)"), {"v": value}).scalar()


# the variant gained or rewrote a chunk since it was counted, or was never counted: a rare cut would read stale shares
def stale(variant: str) -> str | None:
    with Session() as session:
        counted = session.execute(text("SELECT max(newest_chunk) FROM term_frequencies WHERE variant = :v"),
                                  {"v": variant}).scalar()
        newest = session.execute(text("SELECT max(id) FROM data_chunks WHERE variant = :v"), {"v": variant}).scalar()
    if counted is None:
        return f"no term frequencies for {variant}"
    if newest is not None and newest > counted:
        return f"term frequencies of {variant} counted up to chunk {counted}, the variant holds chunk {newest}"
    return None
