import inspect
from types import SimpleNamespace

import pytest
from evals import runner
from use_cases import agent, chat

import db


def _reads_of_data_chunks():
    return [db.hybrid_search, db.nearest_distance, db.corpus_fingerprint,
            db.is_empty, db.list_categories, db.cleanup]


def test_no_reader_of_the_corpus_can_forget_which_variant_it_reads():
    for fn in _reads_of_data_chunks():
        param = inspect.signature(fn).parameters["variant"]
        assert param.kind is inspect.Parameter.KEYWORD_ONLY, fn.__name__
        assert param.default is inspect.Parameter.empty, fn.__name__


def test_a_run_against_an_empty_variant_stops_instead_of_answering_from_nothing(monkeypatch):
    monkeypatch.setattr(runner.db, "is_empty", lambda *, variant: True)
    monkeypatch.setattr(
        runner.db, "corpus_variants", lambda: [{"variant": "baseline", "chunks": 1}]
    )
    with pytest.raises(RuntimeError, match="typo|empty"):
        runner.run("run", set_name="curated", variant="baseline")


def test_a_run_against_a_variant_with_no_declared_policy_stops_before_the_first_question():
    with pytest.raises(ValueError, match="no declared policy"):
        runner.run("run", set_name="curated", variant="nowhere_declared")


def test_the_single_shot_snapshot_names_the_variant_it_read(monkeypatch):
    from use_cases import run_snapshot

    monkeypatch.setattr(run_snapshot.db, "corpus_fingerprint", lambda *, variant: {"chunks": 7})
    monkeypatch.setattr(run_snapshot.llm, "server_context_length", lambda model: None)
    monkeypatch.setattr(run_snapshot.llm, "resolve_name", lambda role: "stub")
    snapshot = chat._config_snapshot(False, 5, True, 0.55, None, "baseline")
    assert snapshot["variant"] == "baseline"
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
        "db.corpus_variants", lambda: [{"variant": "baseline", "chunks": 13068}]
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
    monkeypatch.setattr(db, "_ts_config", lambda *a, **kw: "english")
    monkeypatch.setattr(db.engine, "connect", lambda: _Conn())
    db.hybrid_search("q", [0.0], None, variant="clean_1024", exact=True)

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
        "source": "a.md", "content": "body",
    }

    class _Conn:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def execute(self, statement, *args):
            return SimpleNamespace(mappings=lambda: SimpleNamespace(all=lambda: [scrambled]))

    monkeypatch.setattr(db, "_ts_config", lambda *a, **kw: "english")
    monkeypatch.setattr(db.engine, "connect", lambda: _Conn())
    hit, = db.hybrid_search("q", [0.0], None, variant="clean_1024", exact=True)

    assert (hit.content, hit.source, hit.distance, hit.section) == (
        "body", "a.md", 0.42, "Redis"
    )


def test_the_queries_that_claim_to_read_what_retrieval_reads_filter_the_same_rows():
    # the probe claims the shape `hybrid_search` gives the planner and filtered variant alone
    from evals import build_veto
    from use_cases import search_depth

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
    from use_cases.index import check_variant

    assert check_variant("clean_1024") == "clean_1024"
    for bad in ("clean_1024\n", "clean 1024", "Clean_1024", "", "x" * 37):
        with pytest.raises(ValueError, match="must match"):
            check_variant(bad)
