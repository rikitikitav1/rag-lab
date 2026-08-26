import inspect
from pathlib import Path

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
    monkeypatch.setattr(chat.db, "corpus_fingerprint", lambda *, variant: {"chunks": 7})
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


def test_the_search_mode_is_set_on_checkout_not_on_connect():
    """A pooled connection made before the script ran would come back without the setting."""
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    import retrieval_report
    from orm.sync_db import engine
    from sqlalchemy import event

    retrieval_report.prepare(True)
    listener = retrieval_report._APPLIED
    assert listener is not None
    assert event.contains(engine, "checkout", listener)
    assert not event.contains(engine, "connect", listener)
    event.remove(engine, "checkout", listener)
    retrieval_report._APPLIED = None
