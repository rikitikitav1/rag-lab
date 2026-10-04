from real_db import pytestmark  # noqa: F401
from sqlalchemy import text


# waiting jobs are priced at their type's mean of the last day, and a type with no history is named, not guessed
def test_the_queue_is_counted_and_priced_at_each_types_mean(db, monkeypatch):
    from sqlalchemy.orm import sessionmaker
    from use_cases import queue_stats

    with db.connect() as c:
        c.execute(text("TRUNCATE jobs"))
        for kind, status, elapsed in (("index_data", "done", 30.0), ("index_data", "done", 50.0),
                                      ("index_data", "new", None), ("index_data", "new", None),
                                      ("onboard_source", "new", None), ("accept_questions", "running", None)):
            c.execute(text("INSERT INTO jobs (type, status, options, elapsed)"
                           " VALUES (:t, :s, '{\"source\": \"x\"}', :e)"), {"t": kind, "s": status, "e": elapsed})
    monkeypatch.setattr(queue_stats, "Session", sessionmaker(bind=db))

    out = queue_stats.stats(60)

    assert out["waiting"] == {"index_data": 2, "onboard_source": 1}
    assert [r["type"] for r in out["running"]] == ["accept_questions"] and out["running"][0]["of"] == "x"
    assert out["mean_seconds"]["index_data"] == {"mean": 40, "n": 2}
    assert out["eta_seconds"] == 80 and out["unpriced"] == ["onboard_source"]
    assert out["finished_last_60_min"] == {"index_data": {"done": 2}}


# a job the worker keeps dying under fails on the next restart instead of being claimed first again
def test_a_job_the_worker_dies_under_fails_after_its_reclaims(db, monkeypatch):
    import job_queue
    from sqlalchemy.orm import sessionmaker

    with db.connect() as c:
        c.execute(text("TRUNCATE jobs"))
        c.execute(text("INSERT INTO jobs (type, status, queue, options) VALUES"
                       " ('onboard_source', 'running', 'default', '{\"source\": \"x\", \"reclaims\": 5}'),"
                       " ('index_data', 'running', 'default', '{\"source\": \"y\"}')"))
    monkeypatch.setattr(job_queue, "Session", sessionmaker(bind=db))

    back = job_queue.requeue_stale(["default"])

    with db.connect() as c:
        rows = dict(c.execute(text("SELECT type, status FROM jobs")).all())
    assert rows == {"onboard_source": "error", "index_data": "new"} and len(back) == 1
