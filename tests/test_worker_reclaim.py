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


def test_a_worker_whose_code_differs_still_claims_and_says_so_on_the_row(monkeypatch):
    # refusing to claim turned an edit during a batch into a queue that looked like it had nothing to do
    import version

    monkeypatch.setattr(version, "differs_from_disk", lambda *a, **kw: "loaded aaa, on disk bbb")
    monkeypatch.setattr(worker, "_SAID_STALE", False)
    claimed = SimpleNamespace(id=11, type="index_data", options={})
    monkeypatch.setattr(job_queue, "claim_next", lambda queues: claimed)
    monkeypatch.setattr(worker, "HANDLERS", {"index_data": lambda options: None})
    monkeypatch.setattr(worker.job_specs, "check", lambda *a, **kw: None)
    monkeypatch.setattr(job_queue, "add_tokens", lambda *a, **kw: None)
    monkeypatch.setattr(job_queue, "complete", lambda *a, **kw: None)
    monkeypatch.setattr(worker, "_record_spend", lambda *a, **kw: None)
    monkeypatch.setattr(worker, "_clouds_of", lambda c: [])
    monkeypatch.setattr(worker, "_balances", lambda c: {})
    said = []
    monkeypatch.setattr(worker.log, "warning", lambda event, **kw: said.append(event))

    assert worker.run_once(["default"]) is True
    assert worker.run_once(["default"]) is True
    assert said == ["worker.code_differs"], "loud once, and the row carries it every time"
    assert "differs" in version.mine(), "the stamp of the job says the code was not the tree's"


def test_a_worker_on_the_tree_it_loaded_says_nothing_about_versions():
    import version

    assert version.differs_from_disk() is None


def test_the_stamp_follows_the_bytes_of_the_config_and_not_its_mtime(tmp_path):
    import os

    import version

    first = version.tree_stamp()
    config = version.APP.parent / "config.yaml"
    was, text = config.stat(), config.read_text()
    # a checkout restoring the same bytes must not refuse work; an edited line must
    os.utime(config, ns=(was.st_atime_ns, was.st_mtime_ns + 1000))
    assert version.tree_stamp() == first, "the same bytes are the same stand"
    try:
        config.write_text(text + "\n# a line that changes what is served\n")
        assert version.tree_stamp() != first
    finally:
        config.write_text(text)
        os.utime(config, ns=(was.st_atime_ns, was.st_mtime_ns))
    assert version.tree_stamp() == first


def test_the_stand_says_which_code_the_worker_loaded(tmp_path, monkeypatch):
    # a worker that claims nothing looks from the queue like a worker with nothing to do
    import version
    from use_cases import stand_health

    said = tmp_path / "worker.stamp"
    monkeypatch.setattr(version, "SAID", said)
    version.say_loaded(said)
    clean = stand_health.code()
    assert clean["worker_tree_moved"] is False and clean["worker_loaded_moved"] is None

    # the tree moving beside the worker is hygiene; what decides is a file the worker imported
    monkeypatch.setattr(version, "LOADED_TREE", "older")
    moved = {"count": 1, "files": ["app/llm.py"]}
    monkeypatch.setattr(version, "loaded_differs", lambda every=0: moved)
    version.say_loaded(said)
    code = stand_health.code()
    assert code["worker_tree_moved"] is True and code["worker_loaded_moved"] == moved


def test_a_worker_stamp_older_than_the_loaded_reading_is_not_read_as_clean(tmp_path, monkeypatch):
    import json

    import version
    from use_cases import stand_health

    said = tmp_path / "worker.stamp"
    said.write_text(json.dumps({"stamp": version.LOADED_TREE, "at": "then", "pid": 1}))
    monkeypatch.setattr(version, "SAID", said)
    code = stand_health.code()
    assert code["worker_loaded_moved"] is None and "too_old_to_tell" in code


def test_a_module_imported_through_a_symlink_is_still_looked_at(tmp_path, monkeypatch):
    import sys
    import types

    import version

    link = tmp_path / "app_link"
    link.symlink_to(version.APP)
    fake = types.ModuleType("via_link")
    fake.__file__ = str(link / "version.py")
    monkeypatch.setitem(sys.modules, "via_link", fake)
    assert str(version.APP / "version.py") in version._loaded_paths()


def test_a_stand_whose_worker_never_said_says_so(tmp_path, monkeypatch):
    import version
    from use_cases import stand_health

    monkeypatch.setattr(version, "SAID", tmp_path / "absent.stamp")
    assert "has not said" in stand_health.code()["worker"]


def test_a_claimed_job_carries_the_stamp_of_the_process_that_took_it():
    # "did all the arms run on one code" was a human comparing docker inspect with file mtimes
    import version

    mine = version.mine()
    assert set(mine) == {"stamp", "code_version", "at", "loaded_differs"}
    assert mine["loaded_differs"] is None
    assert mine["stamp"] == version.LOADED_TREE



def test_the_stamp_is_not_recomputed_on_every_poll_of_an_empty_queue():
    # the guard sits before the claim, so an idle worker asked it every three seconds
    import version

    reads = []
    was = version.tree_stamp
    try:
        version.tree_stamp = lambda: reads.append(1) or was()
        version.differs_from_disk(every=0)
        version.differs_from_disk(every=3600)
        version.differs_from_disk(every=3600)
    finally:
        version.tree_stamp = was
    assert len(reads) == 1, "the answer is kept for a while, the tree is not hashed every poll"


def test_comparing_arms_says_whether_they_ran_on_one_code():
    # the stamp was written and nobody read it; two arms on two trees compare two things
    from types import SimpleNamespace as Row

    from evals import compare

    def rows(version):
        return [Row(metrics={"config": {"code_version": version}}) for _ in range(3)]

    same = compare.code_by_run({"a": rows("aaa"), "b": rows("aaa")})
    assert same["one_code"] is True and same["by_run"]["b"] == ["aaa"]
    apart = compare.code_by_run({"a": rows("aaa"), "b": rows("bbb")})
    assert apart["one_code"] is False


def test_the_door_prefers_the_fingerprint_to_the_commit_hash():
    # a commit hash does not see an edit made between the commit and the container's start
    from types import SimpleNamespace as Row

    from evals import compare

    def rows(**config):
        return [Row(metrics={"config": config}) for _ in range(2)]

    one_commit = {"a": rows(code_version="aaa", tree_stamp="one"),
                  "b": rows(code_version="aaa", tree_stamp="two")}
    read = compare.code_by_run(one_commit)
    assert read["one_code"] is False, "same commit, two trees: the hash alone calls this one code"
    assert read["read_from"] == "tree_stamp" and read["by_run"]["a"] == ["one"]

    older = compare.code_by_run({"a": rows(code_version="aaa"), "b": rows(code_version="aaa")})
    assert older["one_code"] is True and older["read_from"] == "code_version"

    # rows too old to carry either field: nothing is known, and that is not "one code"
    blind = compare.code_by_run({"a": rows(), "b": rows()})
    assert blind["one_code"] is None and blind["read_from"] == "nothing"

    # one arm silent and one arm speaking is not agreement either
    half = compare.code_by_run({"a": rows(tree_stamp="one"), "b": rows()})
    assert half["one_code"] is None and half["said_nothing"] == ["b"]

    # a tree that moved beside a run is reported and does not decide: the process ran what it loaded
    walked = compare.code_by_run({
        "a": rows(tree_stamp="one", loaded_differs=None),
        "b": rows(tree_stamp="one", tree_differs="loaded one, on disk two", loaded_differs=None),
    })
    assert walked["one_code"] is True
    assert walked["tree_moved"] == {"b": "loaded one, on disk two"}

    # one arm older than the fingerprint: the two values come from different namespaces
    mixed = compare.code_by_run({"a": rows(code_version="aaa"),
                                 "b": rows(code_version="aaa", tree_stamp="one")})
    assert mixed["one_code"] is None and mixed["read_from"] == "code_version+tree_stamp"


def test_the_fingerprint_can_be_taken_over_a_config_outside_the_tree():
    # an absolute CONFIG_PATH has no name relative to the repository, and the stamp is taken at import
    import config
    import version

    was = config.CONFIG_PATH
    try:
        config.CONFIG_PATH = "/etc/hostname"
        stamp = version.tree_stamp()
    finally:
        config.CONFIG_PATH = was
    assert len(stamp) == 12 and stamp != version.tree_stamp()


def test_a_tree_that_moved_beside_a_run_does_not_veto_its_code():
    # the tree was switched to another branch under a running stand, and the worker kept its code
    from types import SimpleNamespace as Row

    from evals import compare

    def rows(**config):
        return [Row(metrics={"config": {"tree_stamp": "one", **config}}) for _ in range(2)]

    beside = compare.code_by_run({
        "a": rows(loaded_differs=None),
        "b": rows(tree_differs="loaded one, on disk two", loaded_differs=None),
    })
    assert beside["one_code"] is True, "the tree moved, the process kept running what it loaded"
    assert beside["tree_moved"]["b"], "and the reader is still told the tree moved"

    under = compare.code_by_run({
        "a": rows(loaded_differs=None),
        "b": rows(loaded_differs={"count": 1, "files": ["app/llm.py"]}),
    })
    assert under["one_code"] is None, "a file this run imported stopped matching its stamp"
    assert under["loaded_moved"]["b"]["files"] == ["app/llm.py"]


def test_two_processes_writing_one_run_are_named():
    from types import SimpleNamespace as Row

    from evals import compare

    rows = [Row(metrics={"config": {"tree_stamp": "one", "process_started": at}})
            for at in ("2026-09-23T05:00:00+00:00", "2026-09-23T07:00:00+00:00")]
    read = compare.code_by_run({"a": rows})
    assert read["process_started"]["a"] == [
        "2026-09-23T05:00:00+00:00", "2026-09-23T07:00:00+00:00"
    ]


def test_a_run_older_than_the_flag_cannot_be_certified_by_its_silence():
    # the snapshot writes the key with `null` when clean, so a missing key is a row that cannot say
    from types import SimpleNamespace as Row

    from evals import compare

    def rows(**config):
        return [Row(metrics={"config": {"tree_stamp": "one", **config}}) for _ in range(2)]

    old = compare.code_by_run({
        "a": rows(),
        "b": rows(tree_differs="loaded one, on disk two"),
    })
    assert old["one_code"] is None, "its tree moved and its rows predate the flag"
    assert old["too_old_to_tell"] == ["b"]

    new = compare.code_by_run({
        "a": rows(loaded_differs=None),
        "b": rows(tree_differs="loaded one, on disk two", loaded_differs=None),
    })
    assert new["one_code"] is True, "the key is there and says nothing moved under the process"
    assert new["too_old_to_tell"] == [] and new["tree_moved"]["b"]
