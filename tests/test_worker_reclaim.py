from types import SimpleNamespace

import job_queue
import worker


def test_a_job_left_running_goes_back_to_the_queue(monkeypatch):
    jobs = [SimpleNamespace(id=7, status="running"), SimpleNamespace(id=8, status="running")]
    monkeypatch.setattr(job_queue, "requeue_stale", lambda queues: [j.id for j in jobs])
    seen = {}
    monkeypatch.setattr(worker.log, "warning", lambda event, **kw: seen.update(kw))

    worker.reclaim(["default"])

    assert seen == {"ids": [7, 8]}


def test_nothing_stale_stays_quiet(monkeypatch):
    monkeypatch.setattr(job_queue, "requeue_stale", lambda queues: [])
    monkeypatch.setattr(worker.log, "warning", lambda *a, **kw: (_ for _ in ()).throw(AssertionError))

    worker.reclaim(["default"])
