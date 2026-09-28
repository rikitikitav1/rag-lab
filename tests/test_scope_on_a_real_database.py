from real_db import pytestmark  # noqa: F401
from sqlalchemy import text


# the scope is a sql predicate, so its versions and sources are read against rows, not against its text
def _count(conn, scope) -> int:
    import db as stand

    clause, params = stand._scope_filter(scope)
    return conn.execute(text(f"SELECT count(*) FROM data_chunks WHERE variant = 'v' {clause}"), params).scalar()


def test_no_version_reads_the_newest_and_a_version_reads_only_its_own(db):
    import db as stand

    with db.connect() as c:
        c.execute(text("TRUNCATE data_sources CASCADE"))
        c.execute(text("INSERT INTO data_sources (id, name, kind) VALUES (1, 'pg', 'git'), (2, 'book', 'local')"))
        for i, (source, versions) in enumerate(((1, ["18", "17"]), (1, ["17"]), (2, []))):
            c.execute(
                text(
                    "INSERT INTO data_chunks (source_id, source, content, chunk_index, category, language, variant,"
                    " versions) VALUES (:s, 's', 'c', :i, 'postgresql', 'en', 'v', :vs)"
                ),
                {"s": source, "i": i, "vs": versions},
            )
    with db.connect() as c:
        assert _count(c, stand.Scope()) == 2, "18 with 17 and the rolling book; the 17-only row waits for a version"
        assert _count(c, stand.Scope(label="postgresql", version="18")) == 2, "18 and the rolling book"
        assert _count(c, stand.Scope(label="postgresql", version="17")) == 3
        assert _count(c, stand.Scope(sources=("book",))) == 1


def test_the_index_cuts_no_row_that_is_not_accepted_and_empties_no_source_that_yields_nothing(db, monkeypatch):
    from types import SimpleNamespace

    from sources.declaration import SourceFile
    from sqlalchemy.orm import sessionmaker
    from use_cases import index

    with db.connect() as c:
        c.execute(text("TRUNCATE data_sources CASCADE"))
        c.execute(text("INSERT INTO data_sources (id, name, kind, stage) VALUES (1, 'raw-book', 'local', 'raw')"))
        c.execute(text("INSERT INTO data_sources (id, name, kind, stage) VALUES (2, 'gone', 'local', 'accepted')"))
        c.execute(
            text(
                "INSERT INTO data_chunks (source_id, source, content, chunk_index, language, variant)"
                " VALUES (2, 'gone/a.md', 'kept', 0, 'en', 'clean_1024')"
            )
        )

    def source(name, docs):
        settings = SourceFile(name=name, language="en", licence="x", folder=f"datasets/{name}")
        return SimpleNamespace(name=name, settings=settings, root=f"datasets/{name}", documents=lambda policy: docs)

    import llm

    monkeypatch.setattr(index, "Session", sessionmaker(bind=db))
    monkeypatch.setattr(llm, "resolve_name", lambda role: "bge-m3")
    result = index.collect_data([source("raw-book", None), source("gone", [])], variant="clean_1024", build_index=False)

    assert list(result.refused) == ["raw-book", "gone"]
    assert "not accepted" in result.refused["raw-book"] and "no documents" in result.refused["gone"]
    with db.connect() as c:
        assert c.execute(text("SELECT count(*) FROM data_chunks WHERE source_id = 2")).scalar() == 1


def test_a_mark_no_searched_chunk_holds_is_named_and_a_folder_mark_is_held(db, monkeypatch):
    import db as stand

    with db.connect() as c:
        c.execute(text("TRUNCATE data_sources CASCADE"))
        c.execute(text("INSERT INTO data_sources (id, name, kind, active) VALUES (1, 'pg', 'git', true)"))
        c.execute(
            text(
                "INSERT INTO data_chunks (source_id, source, content, chunk_index, language, variant)"
                " VALUES (1, 'pg/docs/locks.md', 'c', 0, 'en', 'v')"
            )
        )
    monkeypatch.setattr(stand, "engine", db)

    assert stand.unreachable_marks(["pg/docs", "pg/docs/locks.md", "pg/nope.md"], variant="v") == ["pg/nope.md"]
