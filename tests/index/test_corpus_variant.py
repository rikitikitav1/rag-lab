import inspect
from types import SimpleNamespace

import corpus_search
import pytest
import text_language
from conftest import stub_engines
from evals import runner
from use_cases import agent, chat

import db


def _reads_of_data_chunks():
    return [corpus_search.hybrid_search, corpus_search.nearest_distance, db.corpus_fingerprint,
            db.is_empty, corpus_search.list_categories]


def test_no_reader_of_the_corpus_can_forget_which_variant_it_reads():
    for fn in _reads_of_data_chunks():
        param = inspect.signature(fn).parameters["variant"]
        assert param.kind is inspect.Parameter.KEYWORD_ONLY, fn.__name__
        assert param.default is inspect.Parameter.empty, fn.__name__


def test_a_run_against_an_empty_variant_stops_instead_of_answering_from_nothing(monkeypatch):
    monkeypatch.setattr(runner.db, "is_empty", lambda *, variant: True)
    monkeypatch.setattr(
        runner.db, "corpus_variants", lambda: [{"variant": "clean_big_1024", "chunks": 1}]
    )
    with pytest.raises(RuntimeError, match="typo|empty"):
        runner.run("run", set_name="curated", variant="clean_big_1024")


def test_a_run_against_a_variant_with_no_declared_policy_stops_before_the_first_question():
    with pytest.raises(ValueError, match="no declared policy"):
        runner.run("run", set_name="curated", variant="nowhere_declared")


def test_the_single_shot_snapshot_names_the_variant_it_read(monkeypatch):
    from use_cases import run_snapshot

    monkeypatch.setattr(run_snapshot.db, "corpus_fingerprint", lambda *, variant: {"chunks": 7})
    monkeypatch.setattr("engines.ollama.context_length", lambda model, spec=None: None)
    stub_engines(monkeypatch, run_snapshot)
    snapshot = chat._config_snapshot(False, 5, True, 0.55, None, "clean_big_1024")
    assert snapshot["variant"] == "clean_big_1024"
    assert snapshot["corpus_fingerprint"] == {"chunks": 7}


def test_neither_snapshot_can_be_built_without_being_told_the_variant():
    for fn in (chat._config_snapshot, chat._log_answer, agent._log_answer):
        param = inspect.signature(fn).parameters["variant"]
        assert param.default is inspect.Parameter.empty, fn.__qualname__


def test_an_empty_named_variant_is_not_a_reason_to_index(monkeypatch):
    import bootstrap

    enqueued = []
    monkeypatch.setattr(bootstrap.job_queue, "enqueue", lambda *a, **kw: enqueued.append(a))
    monkeypatch.setattr(
        "db.corpus_variants", lambda: [{"variant": "clean_big_1024", "chunks": 13068}]
    )
    monkeypatch.setattr("db.is_empty", lambda *, variant: True)
    bootstrap._ensure_index()
    assert enqueued == []


def test_an_empty_database_still_indexes_itself(monkeypatch):
    import bootstrap

    enqueued = []
    monkeypatch.setattr(bootstrap.job_queue, "enqueue", lambda *a, **kw: enqueued.append(a))
    monkeypatch.setattr("db.corpus_variants", list)
    bootstrap._ensure_index()
    assert enqueued and enqueued[0][0] == "index_data"


class _SessionSayingWorkIsPending:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def scalar(self, _):
        return True


def test_a_second_embedding_job_is_not_queued_on_top_of_a_running_one(monkeypatch):
    import bootstrap

    enqueued = []
    monkeypatch.setattr(bootstrap, "Session", _SessionSayingWorkIsPending)
    monkeypatch.setattr(bootstrap.job_queue, "enqueue", lambda *a, **kw: enqueued.append(a))
    monkeypatch.setattr(bootstrap.job_queue, "pending_of_type", lambda type: True)
    bootstrap._ensure_question_embeddings()
    assert enqueued == []

    monkeypatch.setattr(bootstrap.job_queue, "pending_of_type", lambda type: False)
    bootstrap._ensure_question_embeddings()
    assert [a[0] for a in enqueued] == ["embed_questions"]


def test_exact_search_sets_its_mode_on_the_connection_the_query_uses(monkeypatch):
    # a listener on the shared engine is a claim about every later checkout in the process
    import db

    seen = []

    class _Conn:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def execute(self, statement, *args):
            seen.append(str(statement))
            return SimpleNamespace(
                mappings=lambda: SimpleNamespace(all=list), all=list, scalar=lambda: None
            )

    # the language of the question is answered elsewhere and would open its own connection
    monkeypatch.setattr(text_language, "ts_config", lambda *a, **kw: "english")
    monkeypatch.setattr(db.engine, "connect", lambda: _Conn())
    monkeypatch.setattr(corpus_search, "refuse_foreign_vectors", lambda conn, variant, embedded_by=None: None)
    corpus_search.hybrid_search("q", [0.0], None, variant="clean_1024", exact=True, embedded_by="bge-m3@ollama")

    assert seen[0] == "SET LOCAL enable_indexscan = off"
    assert not any("hnsw.ef_search" in s for s in seen), "exact search names no depth"


def test_a_log_row_carries_what_it_was_asked():
    # a verdict read through today's registry is a verdict about another instrument
    from models.eval import QuestionLog

    carried = {c.name for c in QuestionLog.__table__.columns}
    assert {"question_text", "reference_answer"} <= carried

    from use_cases import agent, chat

    for writer in (chat._log_answer, agent._log_answer):
        source = inspect.getsource(writer)
        assert "question_text=question.original_text" in source, writer.__qualname__
        assert "reference_answer=question.reference_answer" in source, writer.__qualname__


def test_a_search_row_is_read_by_name_so_a_moved_column_cannot_change_its_meaning(monkeypatch):
    # nine readers indexed this row by position, `row[6]` being the distance thresholds use
    import db

    scrambled = {
        "section": "Redis", "score": 0.5, "distance": 0.42, "keyword_rank": 2,
        "vector_rank": 1, "chunk_index": 7, "category": "databases.redis",
        "source": "a.md", "content": "body", "versions": ["18"],
    }

    class _Conn:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def execute(self, statement, *args):
            return SimpleNamespace(mappings=lambda: SimpleNamespace(all=lambda: [scrambled]))

    monkeypatch.setattr(text_language, "ts_config", lambda *a, **kw: "english")
    monkeypatch.setattr(db.engine, "connect", lambda: _Conn())
    monkeypatch.setattr(corpus_search, "refuse_foreign_vectors", lambda conn, variant, embedded_by=None: None)
    hit, = corpus_search.hybrid_search("q", [0.0], None, variant="clean_1024", exact=True,
                            embedded_by="bge-m3@ollama")

    assert (hit.content, hit.source, hit.distance, hit.section, hit.versions) == (
        "body", "a.md", 0.42, "Redis", ("18",)
    )


def test_the_queries_that_claim_to_read_what_retrieval_reads_filter_the_same_rows():
    # the probe claims the shape `hybrid_search` gives the planner and filtered variant alone
    import search_depth
    from evals import build_veto

    import db

    assert db.live_rows() in search_depth._probe()

    issued = []

    class _Session:
        def execute(self, statement, params=None):
            issued.append(str(statement))
            return SimpleNamespace(all=list)

    build_veto._headings(_Session(), "clean_1024")

    assert db.live_rows() in issued[-1]


def test_a_variant_name_with_a_trailing_newline_is_refused():
    # `re.match` passed "clean_1024\n" into `ensure_vector_index`, which writes it into DDL
    from corpus_keys import check_variant

    assert check_variant("clean_1024") == "clean_1024"
    for bad in ("clean_1024\n", "clean 1024", "Clean_1024", "", "x" * 37):
        with pytest.raises(ValueError, match="must match"):
            check_variant(bad)



class _Seen:
    def __init__(self, labels):
        self.labels = labels
        self.asked = []

    def execute(self, statement, params=None):
        self.asked.append(params)
        labels = self.labels

        class _R:
            def scalars(self_inner):
                return self_inner

            def all(self_inner):
                return labels

        return _R()


def test_a_search_refuses_vectors_another_embedder_wrote(monkeypatch):
    # bge-m3 on two engines reordered the top-20 of 172 questions in 200, under one name

    with pytest.raises(corpus_search.ForeignVectors, match="bge-m3@ollama.*embeds with bge-m3@vllm"):
        corpus_search.refuse_foreign_vectors(_Seen(["bge-m3@ollama"]), "clean_big_1024", "bge-m3@vllm")
    # a variant half reindexed holds both, and is refused as well
    with pytest.raises(corpus_search.ForeignVectors):
        corpus_search.refuse_foreign_vectors(_Seen(["bge-m3@ollama", "bge-m3@vllm"]), "clean_big_1024", "bge-m3@vllm")
    seen = _Seen(["bge-m3@vllm"])
    corpus_search.refuse_foreign_vectors(seen, "clean_big_1024", "bge-m3@vllm")
    assert seen.asked == [{"variant": "clean_big_1024"}]
    # a question embedded earlier carries its own embedder, and that one decides
    corpus_search.refuse_foreign_vectors(_Seen(["bge-m3@ollama"]), "clean_big_1024", "bge-m3@ollama")
    # a vector nobody marked is a ruler nobody named, refused rather than passed
    with pytest.raises(corpus_search.ForeignVectors, match="no recorded embedder"):
        corpus_search.refuse_foreign_vectors(_Seen(["bge-m3@ollama", None]), "clean_big_1024", "bge-m3@ollama")


def test_no_search_asks_the_role_registry_on_its_own_connection():
    # the guard resolved the embedder through a second pooled connection per search
    import inspect


    for fn in (corpus_search.refuse_foreign_vectors, corpus_search.hybrid_search, corpus_search.nearest_distance):
        assert "llm." not in inspect.getsource(fn), fn.__name__
        assert inspect.signature(fn).parameters["embedded_by"].default is inspect.Parameter.empty


def test_the_index_and_the_questions_write_which_embedder_made_their_vectors(monkeypatch):
    import llm
    from use_cases import index

    monkeypatch.setattr(llm, "embedder_label", lambda role="embedding": "bge-m3@ollama")
    monkeypatch.setattr(llm, "request_embeddings_batch", lambda texts: [[0.0]] * len(texts))
    monkeypatch.setattr(llm, "embed_labelled",
                        lambda texts: ("bge-m3@ollama", [[0.0]] * len(texts)))

    class _Session:
        def execute(self, _stmt):
            return SimpleNamespace(all=lambda: [])

        def add_all(self, rows):
            self.rows = rows

        def commit(self):
            pass

    chunks = [SimpleNamespace(content="a", embedding=None), SimpleNamespace(content="b", embedding=None)]
    index._replace_chunks(_Session(), 1, "clean_big_1024", chunks, embed_size=1)
    assert [c.embedded_by for c in chunks] == ["bge-m3@ollama"] * 2


def test_every_vector_search_passes_the_guard_first(monkeypatch):
    import db

    class _Refused(Exception):
        pass

    guarded = []

    def refuse(conn, variant, embedded_by=None):
        guarded.append((variant, embedded_by))
        raise _Refused

    monkeypatch.setattr(corpus_search, "refuse_foreign_vectors", refuse)
    monkeypatch.setattr(text_language, "ts_config", lambda *a, **kw: "english")

    class _Conn(_Seen):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(db.engine, "connect", lambda: _Conn([]))
    with pytest.raises(_Refused):
        corpus_search.hybrid_search("q", "[0]", None, variant="clean_big_1024", exact=True, embedded_by="x@y")
    with pytest.raises(_Refused):
        corpus_search.nearest_distance([0.0], variant="clean_big_1024", embedded_by="a@b")
    assert guarded == [("clean_big_1024", "x@y"), ("clean_big_1024", "a@b")]


def test_a_compared_question_is_searched_with_the_embedder_that_embedded_it(monkeypatch):
    from use_cases import retrieval_compare

    seen = {}

    class _Db:
        def hybrid_search(self, *a, **kw):
            seen.update(kw)
            return []

    question = {"original_text": "q", "emb": "[0]", "embedded_by": "bge-m3@ollama"}
    retrieval_compare.ranked_lists(_Db(), question, "clean_big_1024")
    assert seen["embedded_by"] == "bge-m3@ollama"


# the depth script lives outside `app`, and a signature change there broke it without a word
def test_the_depth_script_searches_with_the_embedder_of_each_question(monkeypatch, script):
    ef_latency = script("ef_latency")
    seen = []
    monkeypatch.setattr(corpus_search, "hybrid_search", lambda *a, **kw: seen.append(kw))
    monkeypatch.setattr(ef_latency.llm, "embedder_label", lambda: "bge-m3@ollama")

    ef_latency.timings([("q", "[0]", "bge-m3@vllm-embed"), ("q", "[0]", None)], "clean_big_1024", 100)

    assert [kw["embedded_by"] for kw in seen] == ["bge-m3@vllm-embed", "bge-m3@ollama"]
    assert "embedded_by" in ef_latency.SAMPLE


# a cancel between sources leaves the cut ones replaced: they get their report and the index, the rest are named
def test_a_cancelled_index_reports_and_builds_for_the_sources_it_did_cut(monkeypatch):
    from types import SimpleNamespace

    import job_handlers.indexing as indexing
    import sources.factory
    import use_cases.dedup
    import use_cases.index

    built = [SimpleNamespace(name="a"), SimpleNamespace(name="b"), SimpleNamespace(name="c")]
    queued, indexed = [], []
    monkeypatch.setattr(indexing, "require_embedder_ready", lambda: None)
    monkeypatch.setattr(indexing, "clear_the_engine_for", lambda role: None)
    monkeypatch.setattr(sources.factory, "sources", lambda names: built)
    monkeypatch.setattr(use_cases.index, "collect_data", lambda *a, **k: use_cases.index.IndexResult(
        sources=1, chunks=5, refused={}, left=["b", "c"], model="m"))
    monkeypatch.setattr(use_cases.index, "ensure_vector_index", indexed.append)
    monkeypatch.setattr(indexing, "_report_depth", lambda: None)
    monkeypatch.setattr(indexing.job_queue, "enqueue",
                        lambda kind, opts: queued.append(opts.get("source") or kind))
    out = indexing.index_data({"variant": "v", "_job_id": 1})
    assert queued == ["a", "count_terms"] and indexed == ["v"]
    assert out["left_by_cancel"] == ["b", "c"] and out["sources"] == 1
