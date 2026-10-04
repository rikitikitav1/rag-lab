import corpus_search
import search_scope
import text_language
from real_db import pytestmark  # noqa: F401
from sqlalchemy import text


# the scope is a sql predicate, so its versions and sources are read against rows, not against its text
def _count(conn, scope) -> int:

    clause, params = corpus_search._scope_filter(scope)
    return conn.execute(text(f"SELECT count(*) FROM data_chunks WHERE variant = 'v' {clause}"), params).scalar()


def test_no_version_reads_the_newest_and_a_version_reads_only_its_own(db):

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
        said = "18 with 17 and the rolling book; the 17-only row waits for a version"
        assert _count(c, search_scope.Scope()) == 2, said
        assert _count(c, search_scope.Scope(label="postgresql", version="18")) == 2, "18 and the rolling book"
        assert _count(c, search_scope.Scope(label="postgresql", version="17")) == 3
        assert _count(c, search_scope.Scope(sources=("book",))) == 1


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

    # a cancel is read between sources: what is left is named, and nothing of it is touched
    asked = iter([False, True])
    result = index.collect_data([source("raw-book", None), source("gone", [])], variant="clean_1024",
                                build_index=False, stop=lambda: next(asked))
    assert list(result.refused) == ["raw-book"] and result.left == ["gone"] and result.sources == 0


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
    engine = db
    monkeypatch.setattr(stand, "engine", engine)
    monkeypatch.setattr(corpus_search, "engine", engine)

    assert stand.unreachable_marks(["pg/docs", "pg/docs/locks.md", "pg/nope.md"], variant="v") == ["pg/nope.md"]


# the preflight's newest check reads the rows search reads: a version only an inactive source holds is not held
def test_the_versions_held_are_the_ones_an_active_source_holds(db, monkeypatch):
    import db as stand

    with db.connect() as c:
        c.execute(text("TRUNCATE data_sources CASCADE"))
        c.execute(
            text(
                "INSERT INTO data_sources (id, name, kind, active)"
                " VALUES (1, 'pg18', 'git', false), (2, 'pg17', 'git', true)"
            )
        )
        for source, versions in ((1, ["18"]), (2, ["17"])):
            c.execute(
                text(
                    "INSERT INTO data_chunks (source_id, source, content, chunk_index, category, language, variant,"
                    " versions) VALUES (:s, 's', 'c', 0, 'postgresql', 'en', 'v', :vs)"
                ),
                {"s": source, "vs": versions},
            )
    engine = db
    monkeypatch.setattr(stand, "engine", engine)
    monkeypatch.setattr(corpus_search, "engine", engine)

    assert corpus_search.versions_held("v") == {"postgresql": ["17"]}


# a scan cut at ef_search sees only the big source's neighbours; a search narrowed to the small one still finds its own
def test_a_search_narrowed_to_one_source_finds_its_vectors_past_the_scans_depth(db, monkeypatch):
    import random

    import db as stand

    rng = random.Random(7)

    def vector(lean: float) -> str:
        v = [1.0, lean, *(rng.uniform(-0.01, 0.01) for _ in range(1022))]
        return str(v)

    with db.connect() as c:
        c.execute(text("TRUNCATE data_sources CASCADE"))
        c.execute(
            text(
                "INSERT INTO data_sources (id, name, kind, stage, active)"
                " VALUES (1, 'big', 'git', 'accepted', true), (2, 'small', 'git', 'accepted', true)"
            )
        )
        rows = [(1, 0.0)] * 300 + [(2, 0.6)] * 5
        for i, (source, lean) in enumerate(rows):
            c.execute(
                text(
                    "INSERT INTO data_chunks (source_id, source, content, chunk_index, language, variant, embedding,"
                    " embedded_by) VALUES (:s, 's', 'c', :i, 'en', 'v', CAST(:e AS vector), 'm')"
                ),
                {"s": source, "i": i, "e": vector(lean)},
            )
        c.execute(text("CREATE INDEX ON data_chunks USING hnsw (embedding vector_cosine_ops) WHERE variant = 'v'"))
        c.execute(text("ANALYZE data_chunks"))
        # a table this small is sorted whole by any planner; the stand's is walked through the index, as here
        for setting in ("enable_seqscan", "enable_sort"):
            c.execute(text(f'ALTER DATABASE "{db.url.database}" SET {setting} = off'))
    db.dispose()
    # the stand's search sets its scan with SET LOCAL, which an autocommit connection would drop at once
    engine = db.execution_options(isolation_level="READ COMMITTED")
    monkeypatch.setattr(stand, "engine", engine)
    monkeypatch.setattr(corpus_search, "engine", engine)
    monkeypatch.setattr(text_language, "ts_config", lambda *a, **kw: "english")
    query = [1.0, 0.0, *([0.0] * 1022)]

    try:
        hits = corpus_search.hybrid_search(
            "zzzq", str(query), search_scope.Scope(sources=("small",)), variant="v", ef_search=10, embedded_by="m",
            distance_threshold=0.5, limit_vector=5, limit=5,
        )
    finally:
        with db.connect() as c:
            c.execute(text(f'ALTER DATABASE "{db.url.database}" RESET ALL'))
        db.dispose()

    assert len(hits) == 5 and all(h.vector_rank for h in hits)


# a tag the old category path carried in capitals is found by the label the filter asks, lowercased
def test_a_carried_over_tag_in_capitals_is_found_by_its_label_after_the_migration(db):
    from pathlib import Path


    migration = next((Path(__file__).resolve().parents[2] / "db" / "migrations").glob("*_a_tag_is_lowercase_*.sql"))
    up = migration.read_text().split("-- migrate:down")[0]
    with db.connect() as c:
        c.execute(text("TRUNCATE data_sources CASCADE"))
        c.execute(text("INSERT INTO data_sources (id, name, kind) VALUES (1, 'sheets', 'git')"))
        c.execute(
            text(
                "INSERT INTO data_chunks (source_id, source, content, chunk_index, language, variant, tags)"
                " VALUES (1, 's', 'c', 0, 'en', 'v', ARRAY['cheatsheets', 'README'])"
            )
        )
        assert _count(c, search_scope.Scope(label="README")) == 0
        c.execute(text(up))
        assert _count(c, search_scope.Scope(label="README")) == 1


# a version no searched source holds is refused, and the newest clause asks for a strict scan once an older one is in
def test_an_unheld_version_refuses_and_an_older_one_in_search_turns_the_scan_strict(db, monkeypatch):
    import pytest
    from search_scope import ScopeRefused

    import db as stand

    with db.connect() as c:
        c.execute(text("TRUNCATE data_sources CASCADE"))
        c.execute(text("INSERT INTO data_sources (id, name, kind, active) VALUES (1, 'pg', 'git', true)"))
        c.execute(
            text(
                "INSERT INTO data_chunks (source_id, source, content, chunk_index, category, language, variant,"
                " versions) VALUES (1, 's', 'c', 0, 'postgresql', 'en', 'v', ARRAY['18'])"
            )
        )
    engine = db
    monkeypatch.setattr(stand, "engine", engine)
    monkeypatch.setattr(corpus_search, "engine", engine)

    corpus_search.refuse_unheld_version(search_scope.Scope(label="postgresql", version="18"), "v")
    with pytest.raises(ScopeRefused, match="no source in search holds postgresql 17"):
        corpus_search.refuse_unheld_version(search_scope.Scope(label="postgresql", version="17"), "v")
    assert corpus_search.older_versions_held("v") is False

    with db.connect() as c:
        c.execute(
            text(
                "INSERT INTO data_chunks (source_id, source, content, chunk_index, category, language, variant,"
                " versions) VALUES (1, 's', 'c', 1, 'postgresql', 'en', 'v', ARRAY['17'])"
            )
        )
    assert corpus_search.older_versions_held("v") is True
    assert corpus_search.filtered_scan(search_scope.Scope(), "off", older_held=True) == "strict_order"
    assert corpus_search.filtered_scan(search_scope.Scope(), "off") == "off"


# a long body another source holds word for word is counted past each prefix; a short shared one is an echo, not a copy
def test_a_body_another_source_holds_is_counted_across_sources(db, monkeypatch):
    from corpus_keys import body_hash
    from sqlalchemy.orm import sessionmaker
    from use_cases import ingest_quality

    with db.connect() as c:
        c.execute(text("TRUNCATE data_sources CASCADE"))
        for sid, name in ((1, "a"), (2, "b")):
            c.execute(text("INSERT INTO data_sources (id, name, kind, stage) VALUES (:i, :n, 'local', 'accepted')"),
                      {"i": sid, "n": name})
        shared = "a shared body two sources hold word for word. " * 5
        rows = [(1, "a/x.md", "A > X\n" + shared, 6, 0), (1, "a/y.md", "own body", 0, 1),
                (2, "b/z.md", "B > Z\n" + shared, 6, 0), (2, "b/w.md", "own body", 0, 1)]
        for sid, src, content, prefix, idx in rows:
            c.execute(text("INSERT INTO data_chunks (source_id, source, content, prefix_len, chunk_index, language,"
                           " variant, content_hash) VALUES (:s, :src, :c, :p, :i, 'en', 'clean_1024', :h)"),
                      {"s": sid, "src": src, "c": content, "p": prefix, "i": idx, "h": body_hash(content[prefix:])})
        c.commit()
    monkeypatch.setattr(ingest_quality, "Session", sessionmaker(bind=db))

    assert ingest_quality.dup_across_sources("a", "clean_1024") == {
        "chunks": 2, "shared": 1, "share": 0.5, "with": {"b": 1}}


# a text one source holds under several files takes one place when the knob is on, and every place when it is off
def test_copies_of_a_text_in_one_source_collapse_only_when_asked(db, monkeypatch):
    import config

    import db as stand

    def vector(lean: float) -> str:
        return str([1.0, lean, *([0.0] * 1022)])

    with db.connect() as c:
        c.execute(text("TRUNCATE data_sources CASCADE"))
        c.execute(
            text(
                "INSERT INTO data_sources (id, name, kind, stage, active)"
                " VALUES (1, 'man', 'git', 'accepted', true), (2, 'other', 'git', 'accepted', true)"
            )
        )
        rows = [
            (1, "man/ls.1", "h1", 0.0), (1, "man/dir.1", "h1", 0.01), (1, "man/vdir.1", "h1", 0.02),
            (2, "other/ls.md", "h1", 0.03), (1, "man/cp.1", "h2", 0.04),
        ]
        for i, (source_id, source, content_hash, lean) in enumerate(rows):
            c.execute(
                text(
                    "INSERT INTO data_chunks (source_id, source, content, content_hash, chunk_index, language,"
                    " variant, embedding, embedded_by)"
                    " VALUES (:s, :src, :c, :h, :i, 'en', 'v', CAST(:e AS vector), 'm')"
                ),
                {"s": source_id, "src": source, "c": "x" * 300, "h": content_hash, "i": i, "e": vector(lean)},
            )
        c.commit()
    engine = db.execution_options(isolation_level="READ COMMITTED")
    monkeypatch.setattr(stand, "engine", engine)
    monkeypatch.setattr(corpus_search, "engine", engine)
    monkeypatch.setattr(text_language, "ts_config", lambda *a, **kw: "english")
    query = str([1.0, 0.0, *([0.0] * 1022)])

    def sources() -> list[str]:
        hits = corpus_search.hybrid_search(
            "zzzq", query, None, variant="v", exact=True, embedded_by="m", distance_threshold=1.0, limit=4,
        )
        return [h.source for h in hits]

    assert sources() == ["man/ls.1", "man/dir.1", "man/vdir.1", "other/ls.md"]
    monkeypatch.setattr(config.settings.retrieval, "collapse_copies_in_source", True)
    assert sources() == ["man/ls.1", "other/ls.md", "man/cp.1"]
    # a body under the shared-text length is no copy: "See also" in two files stays twice
    with db.connect() as c:
        c.execute(text("UPDATE data_chunks SET content = 'short'"))
        c.commit()
    assert sources() == ["man/ls.1", "man/dir.1", "man/vdir.1", "other/ls.md"]
