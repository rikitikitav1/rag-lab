# which vectors are foreign is a sql predicate, so it is checked against the real schema
import corpus_search
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
    monkeypatch.setattr(llm, "embed_labelled",
                        lambda texts: ("bge-m3@vllm", asked.extend(texts) or [[0.5] * 1024] * len(texts)))
    indexing.embed_questions({})

    assert sorted(asked) == ["q0", "q1"], "the one already embedded by vllm is left alone"
    with db.connect() as c:
        labels = c.execute(text("SELECT DISTINCT embedded_by FROM questions")).scalars().all()
    assert labels == ["bge-m3@vllm"]


def test_the_guard_reads_the_variant_it_searches_and_nothing_else(db):

    with db.connect() as c:
        c.execute(text("TRUNCATE data_sources CASCADE"))
        c.execute(text("INSERT INTO data_sources (id, name, kind) VALUES (1, 's', 'git')"))
        # a chunk with no vector is never searched, and one with a vector and no mark is refused
        for i, (variant, by, vector) in enumerate((
            ("baseline", "bge-m3@ollama", True), ("clean", "bge-m3@vllm", True),
            ("clean", None, False), ("unmarked", None, True),
        )):
            c.execute(text(
                "INSERT INTO data_chunks (source_id, source, content, chunk_index, category,"
                " language, variant, embedded_by, embedding) VALUES (1, 's', 'c', :i, 'a', 'en',"
                " :v, :by, CASE WHEN :vec THEN array_fill(0.1::real, ARRAY[1024])::vector END)"
            ), {"i": i, "v": variant, "by": by, "vec": vector})
    with db.connect() as c:
        corpus_search.refuse_foreign_vectors(c, "clean", "bge-m3@vllm")
        with pytest.raises(corpus_search.ForeignVectors, match="bge-m3@ollama"):
            corpus_search.refuse_foreign_vectors(c, "baseline", "bge-m3@vllm")
        with pytest.raises(corpus_search.ForeignVectors, match="no recorded embedder"):
            corpus_search.refuse_foreign_vectors(c, "unmarked", "bge-m3@vllm")


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


def test_a_reindex_embeds_only_the_text_this_embedder_has_not_seen(db, monkeypatch):
    import llm
    from models.corpus import DataChunk
    from use_cases import index

    with db.connect() as c:
        c.execute(text("TRUNCATE data_sources CASCADE"))
        c.execute(text("INSERT INTO data_sources (id, name, kind) VALUES (1, 's', 'git')"))
        for i, (content, by) in enumerate((("kept", "bge-m3@vllm"), ("foreign", "bge-m3@ollama"))):
            c.execute(text(
                "INSERT INTO data_chunks (source_id, source, content, chunk_index, category, language, variant,"
                " embedded_by, embedding) VALUES (1, 's', :t, :i, 'a', 'en', 'clean', :by,"
                " array_fill(0.25::real, ARRAY[1024])::vector)"
            ), {"t": content, "i": i, "by": by})
    asked = []
    monkeypatch.setattr(llm, "embedder_label", lambda role="embedding": "bge-m3@vllm")
    monkeypatch.setattr(llm, "embed_labelled",
                        lambda texts: ("bge-m3@vllm", asked.extend(texts) or [[0.5] * 1024] * len(texts)))
    chunks = [DataChunk(source_id=1, source="s", content=t, chunk_index=i, category="a", language="en",
                        variant="clean") for i, t in enumerate(("kept", "foreign", "new"))]
    with sessionmaker(bind=db)() as session:
        counted = index._replace_chunks(session, 1, "clean", chunks, embed_size=8)

    assert sorted(asked) == ["foreign", "new"], "another embedder's vector is never reused"
    assert (counted["chunks"], counted["reused"], counted["embedded"]) == (3, 1, 2)
    with db.connect() as c:
        rows = dict(c.execute(text("SELECT content, (embedding::real[])[1] FROM data_chunks")).all())
    assert rows == {"kept": 0.25, "foreign": 0.5, "new": 0.5}
