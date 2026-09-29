import seed
from conftest import FakeSession


def test_the_exported_batch_is_inserted_before_its_links_are_resolved(monkeypatch):
    # a set's originals can live in the same batch, and linking first found nothing
    order = []
    monkeypatch.setattr(seed, "_question_rows", lambda: [])
    monkeypatch.setattr(
        seed, "_exported_rows",
        lambda: [{"text_hash": "h", "original_text": "q", "_source_text": "heading"}],
    )
    monkeypatch.setattr(seed, "Session", FakeSession)
    monkeypatch.setattr(seed, "_insert_questions", lambda s, rows: order.append("insert"))
    monkeypatch.setattr(seed, "_link_originals", lambda s, rows: order.append("link"))

    seed.seed_questions()

    assert order == ["insert", "link"]


def test_the_lookup_key_never_reaches_the_insert(monkeypatch):
    # `_source_text` resolves the link and is not a column
    inserted = []
    monkeypatch.setattr(seed, "_question_rows", lambda: [])
    monkeypatch.setattr(
        seed, "_exported_rows",
        lambda: [{"text_hash": "h", "original_text": "q", "_source_text": "heading"}],
    )
    monkeypatch.setattr(seed, "Session", FakeSession)
    monkeypatch.setattr(seed, "_insert_questions", lambda s, rows: inserted.extend(rows))
    monkeypatch.setattr(seed, "_link_originals", lambda s, rows: None)

    seed.seed_questions()

    assert all("_source_text" not in row for row in inserted)
