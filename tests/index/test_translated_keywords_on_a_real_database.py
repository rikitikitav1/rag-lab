# the third ranking is sql over the real text-search configs, so it is checked on a real base
import config
import corpus_search
import query_translation
from real_db import pytestmark  # noqa: F401
from sqlalchemy import text


def _search(db, monkeypatch, enabled: bool):
    monkeypatch.setattr(config.settings.retrieval.keyword.translation, "enabled", enabled)
    monkeypatch.setattr(query_translation, "_translate", lambda words, model_dir: "connection pool size limit")
    engine = db.execution_options(isolation_level="READ COMMITTED")
    monkeypatch.setattr(corpus_search, "engine", engine)
    far = str([0.0, 1.0, *([0.0] * 1022)])
    return corpus_search.hybrid_search("Какой лимит размера пула соединений?", far, variant="v", embedded_by="m",
                                       exact=True, distance_threshold=0.01, limit=5)


def test_a_russian_question_reaches_an_english_page_only_through_its_translation(db, monkeypatch):
    with db.connect() as c:
        c.execute(text("TRUNCATE data_sources CASCADE"))
        c.execute(text("INSERT INTO data_sources (id, name, kind) VALUES (1, 'docs', 'git')"))
        for i, (content, lang) in enumerate((("The connection pool size limit is 20 by default.", "en"),
                                             ("Лимит размера пула соединений по умолчанию 20.", "ru"))):
            c.execute(text(
                "INSERT INTO data_chunks (source_id, source, content, chunk_index, category, language, variant,"
                " embedded_by, embedding) VALUES (1, 'docs/a.md', :c, :i, 'a', :l, 'v', 'm',"
                " CAST(:e AS vector(1024)))"), {"c": content, "i": i, "l": lang, "e": str([1.0, *([0.0] * 1023)])})

    off = _search(db, monkeypatch, enabled=False)
    on = _search(db, monkeypatch, enabled=True)
    assert [h.content[:3] for h in off] == ["Лим"], "the original finds the Russian page and nothing else"
    assert {h.content[:3] for h in on} == {"Лим", "The"}, "the translation adds the English page, the Russian stays"
    assert next(h for h in on if h.content.startswith("The")).translated_rank == 1


def _chunks(db, contents):
    with db.connect() as c:
        c.execute(text("TRUNCATE data_sources CASCADE"))
        c.execute(text("DELETE FROM term_frequencies"))
        c.execute(text("INSERT INTO data_sources (id, name, kind) VALUES (1, 'docs', 'git')"))
        for i, (content, lang) in enumerate(contents):
            c.execute(text(
                "INSERT INTO data_chunks (source_id, source, content, chunk_index, category, language, variant,"
                " embedded_by, embedding) VALUES (1, :s, :c, :i, 'a', :l, 'v', 'm', CAST(:e AS vector(1024)))"),
                {"s": f"docs/{i}.md", "c": content, "i": i, "l": lang, "e": str([1.0, *([0.0] * 1023)])})
        c.commit()


def test_the_translation_stands_in_for_the_russian_words_when_it_replaces_them(db, monkeypatch):
    _chunks(db, (("The connection pool size limit is 20 by default.", "en"),
                 ("Лимит размера пула соединений по умолчанию 20.", "ru")))
    monkeypatch.setattr(config.settings.retrieval.keyword, "query", "or")
    monkeypatch.setattr(config.settings.retrieval.keyword.translation, "replaces", True)
    hits = _search(db, monkeypatch, enabled=True)
    assert [h.content[:3] for h in hits] == ["The"], "the keyword search reads the English words alone"
    assert hits[0].keyword_rank == 1 and hits[0].translated_rank is None, "no third list beside it"


def test_the_rare_cut_keeps_only_chunks_holding_a_rare_word_and_the_full_query_ranks_them(db, monkeypatch):
    import term_frequencies
    from sqlalchemy.orm import sessionmaker

    _chunks(db, [(f"connection timeout in section {i}", "en") for i in range(10)]
            + [("connection bashpid holds the shell's own process id", "en")])
    monkeypatch.setattr(term_frequencies, "Session", sessionmaker(bind=db))
    assert term_frequencies.refresh("v")["chunks"] == 11
    monkeypatch.setattr(corpus_search, "engine", db.execution_options(isolation_level="READ COMMITTED"))
    monkeypatch.setattr(config.settings.retrieval.keyword, "query", "or")
    far = str([0.0, 1.0, *([0.0] * 1022)])

    def keyword_hits(share):
        monkeypatch.setattr(config.settings.retrieval.keyword, "max_term_share", share)
        rows = corpus_search.hybrid_search("connection bashpid", far, variant="v", embedded_by="m", exact=True,
                                           distance_threshold=0.01, limit=20)
        return [h.content[:16] for h in rows if h.keyword_rank]

    assert len(keyword_hits(0.0)) == 11, "off: every chunk holding either word is a candidate"
    assert keyword_hits(0.5) == ["connection bashp"], "on: only the chunk holding the rare word"
    assert term_frequencies.stale("v") is None
