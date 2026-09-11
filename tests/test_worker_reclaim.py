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


def test_a_deferral_has_a_ceiling_of_its_own(monkeypatch):
    # a deferral never touched `attempts`, so a job waiting for an absent model held its lane
    import worker

    failed, rescheduled = [], []
    monkeypatch.setattr(worker.job_queue, "fail", lambda id, error, **kw: failed.append(error))
    monkeypatch.setattr(
        worker.job_queue, "reschedule",
        lambda id, options, delay, **kw: rescheduled.append(options.get("deferred_seconds")),
    )
    monkeypatch.setattr(worker, "_fail_the_experiment_waiting_on", lambda claimed: None)

    def _defer(options):
        raise worker.Deferred(30)

    monkeypatch.setitem(worker.HANDLERS, "judge_answers", _defer)
    claimed = SimpleNamespace(
        id=1, type="judge_answers", options={"run_name": "r", "deferred_seconds": 0}
    )
    monkeypatch.setattr(worker.job_queue, "claim_next", lambda queues: claimed)

    worker.run_once(["cpu"])
    assert rescheduled == [30], "an early deferral is rescheduled with the time it waited"

    claimed.options = {"run_name": "r", "deferred_seconds": worker.MAX_DEFERRED_SECONDS}
    worker.run_once(["cpu"])
    assert failed and "gave up" in failed[0]["error"]


def test_a_final_refusal_is_not_tried_again(monkeypatch):
    # a retry of the same refusal only wakes the server and takes the card again
    failed, rescheduled = [], []
    monkeypatch.setattr(worker.job_queue, "fail", lambda id, error, **kw: failed.append(error))
    monkeypatch.setattr(worker.job_queue, "reschedule",
                        lambda id, options, delay, **kw: rescheduled.append(options))
    monkeypatch.setattr(worker, "_fail_the_experiment_waiting_on", lambda claimed: None)

    def refused(options):
        raise worker.Final("Qwen/Q on vllm does not return tool calls")

    def flaky(options):
        raise RuntimeError("connection reset")

    claimed = SimpleNamespace(id=1, type="hand_card", options={"engine_id": 3})
    monkeypatch.setattr(worker.job_queue, "claim_next", lambda queues: claimed)
    monkeypatch.setitem(worker.HANDLERS, "hand_card", refused)
    worker.run_once(["default"])
    assert rescheduled == [] and failed[0]["attempts"] == 1

    monkeypatch.setitem(worker.HANDLERS, "hand_card", flaky)
    worker.run_once(["default"])
    assert len(rescheduled) == 1, "a transient failure still gets its retries"
