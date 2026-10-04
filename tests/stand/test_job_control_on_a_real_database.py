# holding, releasing, cancelling in bulk and the parent link all live in sql, so they run on a real base
import job_queue
import pytest
from errors import Refusal
from models import FailKind, JobStatus
from real_db import pytestmark  # noqa: F401
from sqlalchemy import text
from sqlalchemy.orm import sessionmaker
from use_cases import job_control
from use_cases.job_control import Scope


@pytest.fixture
def jobs(db, monkeypatch):
    monkeypatch.setattr(job_queue, "Session", sessionmaker(bind=db))
    with db.connect() as c:
        c.execute(text("TRUNCATE jobs RESTART IDENTITY CASCADE"))
        c.execute(text("INSERT INTO jobs (id, type, queue, status, options) VALUES"
                       " (1, 'index_data', 'default', 'running', '{\"source\": \"a\"}'),"
                       " (2, 'index_data', 'default', 'new', '{\"source\": \"b\"}'),"
                       " (3, 'index_data', 'default', 'new', '{\"source\": \"c\"}'),"
                       " (4, 'analyze_source', 'default', 'new', '{\"source\": \"a\"}')"))
        c.execute(text("SELECT setval('jobs_id_seq', 4)"))
    return db


def _statuses(db) -> dict:
    with db.connect() as c:
        return dict(c.execute(text("SELECT id, status FROM jobs ORDER BY id")).all())


def test_a_cancel_of_the_waiting_spares_the_running_job_and_a_dry_run_touches_nothing(jobs):
    scope = Scope(type="index_data", statuses=[JobStatus.new], every=True)
    assert job_control.cancel(scope, dry_run=True) == {"would_cancel": [2, 3]}
    assert _statuses(jobs)[2] == "new"
    assert sorted(job_control.cancel(scope)["cancelled"]) == [2, 3]
    assert _statuses(jobs) == {1: "running", 2: "cancelled", 3: "cancelled", 4: "new"}


def test_a_held_job_is_passed_by_and_comes_back_in_its_place(jobs):
    assert job_control.pause(Scope(type="index_data", every=True)) == {"paused": [2, 3]}
    claimed = job_queue.claim_next(["default"])
    assert claimed.id == 4, "the worker skips what is held"
    assert job_control.resume(Scope(ids=[2, 3])) == {"resumed": [2, 3]}
    assert _statuses(jobs)[2] == "new"


def test_a_held_job_still_counts_as_live_for_the_guards_and_the_cancel(jobs):
    job_queue.pause([2])
    assert job_queue.pending_of_type("index_data", source="b") == 2
    assert job_queue.cancel([2]) == [2]


def test_a_type_alone_is_refused_unless_every_is_said(jobs):
    with pytest.raises(Refusal, match="every=true"):
        job_control.cancel(Scope(type="index_data"))
    with pytest.raises(Refusal, match="name a run_name"):
        job_control.pause(Scope())


def test_a_job_queued_by_a_running_handler_names_it_and_one_queued_from_outside_does_not(jobs, monkeypatch):
    monkeypatch.setattr(job_queue, "_check", lambda type, options: None)
    outside = job_queue.enqueue("analyze_source", {"source": "x"})
    token = job_queue.running_job.set(1)
    try:
        child = job_queue.enqueue("analyze_source", {"source": "y"})
    finally:
        job_queue.running_job.reset(token)
    explicit = job_queue.enqueue("analyze_source", {"source": "z"}, parent_id=3)
    with jobs.connect() as c:
        parents = dict(c.execute(text("SELECT id, parent_id FROM jobs WHERE id > 4")).all())
    assert parents == {outside: None, child: 1, explicit: 3}
    assert job_control.cancel(Scope(parent_id=1), dry_run=True) == {"would_cancel": [child]}


def test_a_failure_says_its_kind_and_a_retry_says_why_it_waits(jobs):
    from datetime import timedelta

    job_queue.fail(2, {"error": "no"}, FailKind.final)
    job_queue.reschedule(3, {"source": "c", "attempts": 1}, timedelta(seconds=60), because="connection reset")
    with jobs.connect() as c:
        error = c.execute(text("SELECT error FROM jobs WHERE id = 2")).scalar()
        waiting = c.execute(text("SELECT options->'waiting_because' FROM jobs WHERE id = 3")).scalar()
    assert error == {"error": "no", "kind": "final"}
    assert waiting["kind"] == "retry" and waiting["error"] == "connection reset" and waiting["retry_at"]
