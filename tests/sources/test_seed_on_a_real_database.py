from real_db import pytestmark  # noqa: F401
from sqlalchemy import text


# the seed writes each file's whole declaration, again and again to the same rows, and keeps a row's stage
def test_the_seed_is_idempotent_the_file_wins_and_a_new_row_starts_declared(db, monkeypatch):
    import seed
    from sources import files
    from sources.declaration import SourceFile
    from sqlalchemy.orm import sessionmaker

    sheets = SourceFile(name="sheets", language="en", licence="CC BY", folder="/sheets", categories=["python"])
    book = SourceFile(name="book", language="en", licence="CC BY", folder="inbox/book", reader="converted")
    found = {"sheets": sheets, "book": book}
    monkeypatch.setattr(seed, "Session", sessionmaker(bind=db))
    monkeypatch.setattr(files, "source_files", lambda: found)
    monkeypatch.setattr(files, "veto_families", lambda found: {})
    import job_queue

    queued = []
    monkeypatch.setattr(job_queue, "enqueue", lambda kind, options: queued.append((kind, options["source"])))
    monkeypatch.setattr(job_queue, "pending_of_type", lambda kind, source=None: 1 if (kind, source) in queued else None)

    def rows():
        with db.connect() as c:
            return {
                r.name: (r.stage, r.declaration, r.seeded)
                for r in c.execute(text("SELECT name, stage, declaration, seeded FROM data_sources ORDER BY name"))
            }

    seed.seed_sources()
    first = rows()
    # each inserted row walks the line
    assert sorted(queued) == [("onboard_source", "book"), ("onboard_source", "sheets")]
    seed.seed_sources()
    assert rows() == first
    assert len(queued) == 2, "a second seed queues no second onboarding"
    # a row left declared with no job waiting, as after a seed that died before queueing, is queued on the next seed
    queued.remove(("onboard_source", "book"))
    seed.seed_sources()
    assert sorted(queued) == [("onboard_source", "book"), ("onboard_source", "sheets")]
    assert first["sheets"][2] is True, "a seeded row is marked so the knobs door refuses it"
    assert first["book"][0] == "declared" and first["sheets"][0] == "declared", "every new row walks the line"
    assert first["book"][1]["reader"] == "converted" and first["sheets"][1]["categories"] == ["python"]

    with db.connect() as c:
        c.execute(text("UPDATE data_sources SET stage = 'accepted' WHERE name = 'book'"))
    found["sheets"] = sheets.model_copy(update={"categories": ["redis"]})
    seed.seed_sources()
    after = rows()
    assert after["book"][0] == "accepted", "a row's stage is its own, the seed does not set it back"
    assert after["sheets"][1]["categories"] == ["redis"], "the file wins over the row's declaration"


# a file that leaves the language unsaid keeps the one onboarding found, as the index keeps it
def test_the_seed_keeps_a_language_its_file_leaves_unsaid(db, monkeypatch):
    import job_queue
    import seed
    from sources import files
    from sources.declaration import SourceFile
    from sqlalchemy.orm import sessionmaker

    book = SourceFile(name="book", licence="CC BY", folder="inbox/book")
    monkeypatch.setattr(seed, "Session", sessionmaker(bind=db))
    monkeypatch.setattr(files, "source_files", lambda: {"book": book})
    monkeypatch.setattr(files, "veto_families", lambda found: {})
    monkeypatch.setattr(job_queue, "enqueue", lambda kind, options: None)
    monkeypatch.setattr(job_queue, "pending_of_type", lambda kind, source=None: 1)
    with db.connect() as c:
        c.execute(text("TRUNCATE data_sources CASCADE"))
        c.execute(text("INSERT INTO data_sources (name, kind, language, stage) VALUES ('book', 'local', 'ru', 'raw')"))

    seed.seed_sources()

    with db.connect() as c:
        row = c.execute(text("SELECT language, path FROM data_sources WHERE name = 'book'")).one()
    assert tuple(row) == ("ru", "inbox/book")

