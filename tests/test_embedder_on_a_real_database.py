# which vectors are foreign is a sql predicate, so it is checked against the real schema
import pytest
from real_db import pytestmark  # noqa: F401
from sqlalchemy import text
from sqlalchemy.orm import sessionmaker


def test_questions_embedded_by_another_embedder_are_embedded_again(db, monkeypatch):
    import llm
    from job_handlers import indexing

    with db.connect() as c:
        c.execute(text("TRUNCATE questions CASCADE"))
        for i, (vector, by) in enumerate(((None, None), ("[1]", "bge-m3@ollama"),
                                          ("[1]", "bge-m3@vllm"))):
            c.execute(text(
                "INSERT INTO questions (id, text_hash, original_text, embedding, embedded_by)"
                " VALUES (:i, :h, :t, CAST(:v AS vector(1024)), :by)"
            ), {"i": i + 1, "h": f"h{i}", "t": f"q{i}",
                "v": None if vector is None else str([1.0] * 1024), "by": by})
    asked = []
    monkeypatch.setattr(indexing, "Session", sessionmaker(bind=db))
    monkeypatch.setattr(indexing, "require_embedder_ready", lambda: None)
    monkeypatch.setattr(indexing, "clear_the_engine_for", lambda role: None)
    monkeypatch.setattr(llm, "embedder_label", lambda role="embedding": "bge-m3@vllm")
    monkeypatch.setattr(llm, "request_embeddings_batch",
                        lambda texts: asked.extend(texts) or [[0.5] * 1024] * len(texts))
    indexing.embed_questions({})

    assert sorted(asked) == ["q0", "q1"], "the one already embedded by vllm is left alone"
    with db.connect() as c:
        labels = c.execute(text("SELECT DISTINCT embedded_by FROM questions")).scalars().all()
    assert labels == ["bge-m3@vllm"]


def test_the_guard_reads_the_variant_it_searches_and_nothing_else(db, monkeypatch):
    import llm

    import db as stand

    with db.connect() as c:
        c.execute(text("TRUNCATE data_sources CASCADE"))
        c.execute(text("INSERT INTO data_sources (id, name, kind) VALUES (1, 's', 'git')"))
        for i, (variant, by) in enumerate((("baseline", "bge-m3@ollama"),
                                           ("clean", "bge-m3@vllm"), ("clean", None))):
            c.execute(text(
                "INSERT INTO data_chunks (source_id, source, content, chunk_index, category,"
                " language, variant, embedded_by) VALUES (1, 's', 'c', :i, 'a', 'en', :v, :by)"
            ), {"i": i, "v": variant, "by": by})
    monkeypatch.setattr(llm, "embedder_label", lambda role="embedding": "bge-m3@vllm")
    with db.connect() as c:
        stand.refuse_foreign_vectors(c, "clean")
        with pytest.raises(stand.ForeignVectors, match="bge-m3@ollama"):
            stand.refuse_foreign_vectors(c, "baseline")


def test_the_bootstrap_queues_the_questions_again_when_the_embedder_moved(db, monkeypatch):
    import bootstrap
    import llm

    with db.connect() as c:
        c.execute(text("TRUNCATE questions CASCADE"))
        c.execute(text(
            "INSERT INTO questions (id, text_hash, original_text, embedding, embedded_by)"
            " VALUES (1, 'h', 'q', CAST(:v AS vector(1024)), 'bge-m3@ollama')"
        ), {"v": str([1.0] * 1024)})
    queued = []
    monkeypatch.setattr(bootstrap, "Session", sessionmaker(bind=db))
    monkeypatch.setattr(bootstrap.job_queue, "pending_of_type", lambda t, **o: False)
    monkeypatch.setattr(bootstrap.job_queue, "enqueue", lambda t, o, **kw: queued.append(t))
    monkeypatch.setattr(llm, "embedder_label", lambda role="embedding": "bge-m3@ollama")
    bootstrap._ensure_question_embeddings()
    assert queued == [], "every question already carries a vector of this embedder"
    monkeypatch.setattr(llm, "embedder_label", lambda role="embedding": "bge-m3@vllm")
    bootstrap._ensure_question_embeddings()
    assert queued == ["embed_questions"]
