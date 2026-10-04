# the notice of a finished job goes through postgres, so it is heard on a real connection
import json

import job_queue
import psycopg
import worker
from real_db import pytestmark  # noqa: F401
from sqlalchemy import text
from sqlalchemy.orm import sessionmaker


def _listener(db):
    conn = psycopg.connect(db.url.set(drivername="postgresql").render_as_string(hide_password=False),
                           autocommit=True)
    conn.execute(f"LISTEN {job_queue.FINISHED_CHANNEL}")
    return conn


def _heard(conn) -> list[dict]:
    return [json.loads(n.payload) for n in conn.notifies(timeout=0.5)]


def test_a_finished_job_is_announced_and_a_rescheduled_one_is_not(db, monkeypatch, quiet_announcements):
    monkeypatch.setattr(job_queue, "Session", sessionmaker(bind=db))
    monkeypatch.setattr(job_queue, "announce_finished", quiet_announcements)
    with db.connect() as c:
        c.execute(text("TRUNCATE jobs CASCADE"))
        c.execute(text("INSERT INTO jobs (id, type, queue, status, options) VALUES"
                       " (1, 'index_data', 'default', 'done', '{\"run_name\": \"r\"}'),"
                       " (2, 'index_data', 'default', 'new', '{}')"))
    conn = _listener(db)
    try:
        job_queue.announce_finished([1, 2])
        assert _heard(conn) == [{"id": 1, "type": "index_data", "status": "done", "run_name": "r"}]
    finally:
        conn.close()


def test_a_cancel_announces_the_waiting_jobs_and_leaves_the_running_one_to_the_worker(db, monkeypatch,
                                                                                      quiet_announcements):
    monkeypatch.setattr(job_queue, "Session", sessionmaker(bind=db))
    monkeypatch.setattr(job_queue, "announce_finished", quiet_announcements)
    with db.connect() as c:
        c.execute(text("TRUNCATE jobs CASCADE"))
        c.execute(text("INSERT INTO jobs (id, type, queue, status, options) VALUES"
                       " (1, 'index_data', 'default', 'running', '{}'), (2, 'index_data', 'default', 'new', '{}')"))
    conn = _listener(db)
    try:
        assert sorted(job_queue.cancel([1, 2])) == [1, 2]
        assert [e["id"] for e in _heard(conn)] == [2]
    finally:
        conn.close()


def test_the_worker_announces_whatever_ended_the_job(monkeypatch):
    announced = []
    monkeypatch.setattr(job_queue, "announce_finished", announced.extend)
    monkeypatch.setattr(job_queue, "claim_next", lambda queues: job_queue.ClaimedJob(id=7, type="boom", options={}))
    monkeypatch.setitem(worker.HANDLERS, "boom", lambda options: (_ for _ in ()).throw(worker.Final("no")))
    monkeypatch.setattr(worker.job_specs, "check", lambda *a, **kw: None)
    monkeypatch.setattr(job_queue, "fail", lambda *a, **kw: None)
    monkeypatch.setattr(job_queue, "add_tokens", lambda *a, **kw: None)
    monkeypatch.setattr(worker, "_clouds_of", lambda c: [])
    monkeypatch.setattr(worker, "_fail_the_experiment_waiting_on", lambda c: None)
    assert worker.run_once(["default"])
    assert announced == [7]


def test_every_line_a_handler_logs_carries_its_job_and_a_job_it_queues_names_it(monkeypatch):
    import structlog

    seen = []

    def handler(options):
        seen.append((structlog.contextvars.get_contextvars().get("job_id"), job_queue.running_job.get()))

    monkeypatch.setattr(job_queue, "claim_next", lambda queues: job_queue.ClaimedJob(id=8, type="quiet", options={}))
    monkeypatch.setitem(worker.HANDLERS, "quiet", handler)
    monkeypatch.setattr(worker.job_specs, "check", lambda *a, **kw: None)
    monkeypatch.setattr(job_queue, "complete", lambda *a, **kw: None)
    monkeypatch.setattr(job_queue, "add_tokens", lambda *a, **kw: None)
    monkeypatch.setattr(worker, "_clouds_of", lambda c: [])
    assert worker.run_once(["default"])
    assert seen == [(8, 8)]
    assert "job_id" not in structlog.contextvars.get_contextvars() and job_queue.running_job.get() is None
