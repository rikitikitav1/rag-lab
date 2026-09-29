import json

from real_db import pytestmark  # noqa: F401
from sqlalchemy import text


def _row(c, name, stage, declaration, raw=None):
    c.execute(
        text(
            "INSERT INTO data_sources (name, kind, stage, declaration, raw)"
            " VALUES (:n, 'local', :s, CAST(:d AS jsonb), CAST(:r AS jsonb))"
        ),
        {"n": name, "s": stage, "d": json.dumps(declaration) if declaration else None, "r": json.dumps(raw or {})},
    )


# the index reads accepted rows only, by name, a page at a time; a family shares one declaration over its rows
def test_the_index_reads_accepted_rows_by_name_a_page_at_a_time(db, monkeypatch):
    from sources import factory
    from sqlalchemy.orm import sessionmaker

    family = {"name": "repos", "git_family": {"base_url": "https://github.com/x", "repos": ["a-repo", "b-repo"]}}
    with db.connect() as c:
        c.execute(text("TRUNCATE data_sources CASCADE"))
        _row(c, "a-repo", "accepted", family)
        _row(c, "b-repo", "accepted", family)
        _row(c, "book", "accepted", {"name": "book", "folder": "inbox/book"}, {"folder": "datasets/raw_sources/book_1"})
        _row(c, "draft", "declared", {"name": "draft", "folder": "inbox/draft"})
        _row(c, "unseeded", "accepted", None)
    import orm.sync_db

    monkeypatch.setattr(orm.sync_db, "Session", sessionmaker(bind=db))

    found, rows = factory._accepted()
    assert sorted(rows) == ["a-repo", "b-repo", "book"], "a declared row and one with no declaration are not read"
    assert sorted(found) == ["book", "repos"] and rows["a-repo"]["source"] == "repos"
    assert rows["book"]["raw"]["folder"] == "datasets/raw_sources/book_1"
    assert sorted(factory._accepted(["book"])[1]) == ["book"]
    first, second = factory._accepted(limit=2)[1], factory._accepted(limit=2, offset=2)[1]
    assert len(first) == 2 and len(second) == 1 and not set(first) & set(second)
    assert sorted({*first, *second}) == ["a-repo", "b-repo", "book"], "the pages together are every accepted row"
