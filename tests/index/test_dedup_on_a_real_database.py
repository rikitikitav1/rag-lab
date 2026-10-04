import json

from real_db import pytestmark  # noqa: F401
from sqlalchemy import text

SHARED = "VACUUM reclaims the space dead tuples hold. " * 6


# a book, the official docs and a notes bank hold one paragraph: the docs keep it, the others lose their copy
def test_a_text_held_by_several_sources_stays_with_the_most_trusted_copy(db, monkeypatch):
    from corpus_keys import body_hash
    from sqlalchemy.orm import sessionmaker
    from use_cases import dedup

    rows = [
        (1, "pg-docs", {"name": "pg-docs", "folder": "datasets/inbox/git/postgresql"}),
        (2, "pg-book", {"name": "pg-book", "folder": "datasets/inbox/books/postgresql-internals"}),
        (3, "sql-interview-questions", {"name": "interview", "reader": "interview"}),
    ]
    with db.connect() as c:
        c.execute(text("TRUNCATE data_sources CASCADE"))
        for sid, name, declaration in rows:
            c.execute(text("INSERT INTO data_sources (id, name, kind, stage, declaration) VALUES (:i, :n, 'local',"
                           " 'accepted', CAST(:d AS jsonb))"), {"i": sid, "n": name, "d": json.dumps(declaration)})
        for sid, prefix in ((1, "# Docs\n"), (2, "# Book\n"), (3, "# Bank\n")):
            # the bank's copy breaks its lines elsewhere, and is the same text all the same
            shared = SHARED.replace(". ", ".\n", 2) if sid == 3 else SHARED
            for i, body in enumerate((shared, "See also.", f"only in {sid}")):
                c.execute(text("INSERT INTO data_chunks (source_id, source, content, prefix_len, chunk_index,"
                               " language, variant, content_hash) VALUES (:s, 's', :c, :p, :i, 'en', 'v', :h)"),
                          {"s": sid, "c": prefix + body, "p": len(prefix), "i": i, "h": body_hash(body)})
    monkeypatch.setattr(dedup, "Session", sessionmaker(bind=db))

    dropped = dedup.drop_lower_copies("v", ["pg-book"])

    assert dropped == {"pg-book": {"pg-docs": 1}, "sql-interview-questions": {"pg-docs": 1}}
    with db.connect() as c:
        left = c.execute(text("SELECT source_id, content FROM data_chunks ORDER BY source_id, chunk_index")).all()
    assert [s for s, content in left if "VACUUM" in content] == [1]
    assert len(left) == 7, "each source keeps the text only it holds, and a body too short to be a copy"
    with db.connect() as c:
        kept_by = c.execute(text("SELECT raw->'copies_kept_by' FROM data_sources WHERE name = 'pg-book'")).scalar()
    assert kept_by == {"v": {"pg-docs": 1}}


# a source declares its own trust over the one its origin suggests
def test_a_declared_trust_wins_over_the_origin():
    from models.corpus import Trust
    from use_cases.dedup import source_trust

    assert source_trust({"folder": "datasets/inbox/books/x"}) == Trust.book
    assert source_trust({"urls": ["https://x/book.PDF"]}) == Trust.book
    assert source_trust({"reader": "cheatsheets"}) == Trust.notes
    assert source_trust({"folder": "datasets/inbox/git/kafka-docs"}) == Trust.official
    assert source_trust({"folder": "datasets/inbox/books/x", "trust": "official"}) == Trust.official
