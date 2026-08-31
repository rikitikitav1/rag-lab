from job_handlers.judging import _MAX_JUDGE_ATTEMPTS, _errored, _errored_metric


def test_errored_false_until_cap():
    metrics = {}
    for i in range(1, _MAX_JUDGE_ATTEMPTS):
        metrics["relevance"] = _errored_metric(metrics, "relevance", "RuntimeError")
        assert metrics["relevance"]["attempts"] == i
        assert _errored(metrics, "relevance") is False


def test_errored_true_at_cap():
    metrics = {}
    for _ in range(_MAX_JUDGE_ATTEMPTS):
        metrics["relevance"] = _errored_metric(metrics, "relevance", "RuntimeError")
    assert _errored(metrics, "relevance") is True


def test_errored_metric_stores_class_not_message():
    m = _errored_metric({}, "faithfulness", "RuntimeError")
    assert m == {"error": "RuntimeError", "attempts": 1}


class _Log:
    def __init__(self):
        self.id = 1
        self.relevance = None


def _verdict(score=8, **kw):
    from models.registry import Purpose
    from use_cases.judge import Verdict

    return Verdict(
        reason="because", score=score, model="qwen2.5:7b",
        purpose=kw.get("purpose", Purpose.judge_relevance),
        prompt_version=kw.get("prompt_version", 2),
    )


def test_a_written_verdict_names_the_judge_and_its_prompt():
    from job_handlers.judging import _judge_axis, _Snapshot

    snapshot = _Snapshot({}, {"generate_answer": 3}, {"generation": "gemma3:4b"})
    wrote = _judge_axis(
        _Log(), snapshot, "relevance", True, False, lambda *a: _verdict(), (),
    )
    assert wrote is True
    assert snapshot.models["judging"] == "qwen2.5:7b"
    assert snapshot.prompts["judge_relevance"] == 2
    # what produced the answer is not overwritten by what scored it
    assert snapshot.models["generation"] == "gemma3:4b"
    assert snapshot.prompts["generate_answer"] == 3


def test_a_failed_axis_leaves_the_snapshot_alone():
    from job_handlers.judging import _judge_axis, _Snapshot

    def boom(*a):
        raise RuntimeError("judge is down")

    snapshot = _Snapshot({}, {}, {})
    wrote = _judge_axis(_Log(), snapshot, "relevance", True, False, boom, ())
    assert wrote is False
    assert snapshot.models == {} and snapshot.prompts == {}
    assert snapshot.metrics["relevance"]["attempts"] == 1


def test_rejudging_replaces_the_prompt_version_it_names():
    from job_handlers.judging import _judge_axis, _Snapshot
    from models.registry import Purpose

    snapshot = _Snapshot({}, {"judge_relevance": 2}, {})
    _judge_axis(
        _Log(), snapshot, "relevance", True, True,
        lambda *a: _verdict(purpose=Purpose.judge_relevance, prompt_version=3), (),
    )
    assert snapshot.prompts["judge_relevance"] == 3


def test_a_job_without_a_bench_judges_with_whatever_is_active():
    from job_handlers.judging import _bench_from
    from use_cases import judge

    assert _bench_from({}) == judge.Bench(model=None, versions=None)


def test_a_job_can_name_its_judge_without_touching_the_stand():
    from job_handlers.judging import _bench_from
    from models.registry import Purpose

    bench = _bench_from(
        {"judge_model": "qwen3:4b", "judge_prompts": {"judge_faithfulness": 3}}
    )
    assert bench.model == "qwen3:4b"
    assert bench.versions == {Purpose.judge_faithfulness: 3}


def test_an_unnamed_axis_falls_back_to_the_active_prompt(monkeypatch):
    from models.registry import Purpose
    from use_cases import judge

    monkeypatch.setattr(judge.prompt_repo, "active", lambda p: (f"active {p}", 2))
    monkeypatch.setattr(judge.prompt_repo, "template_of", lambda p, v: f"pinned {p} v{v}")

    bench = judge.Bench(versions={Purpose.judge_faithfulness: 3})
    assert bench.template(Purpose.judge_faithfulness) == ("pinned judge.faithfulness v3", 3)
    assert bench.template(Purpose.judge_relevance) == ("active judge.relevance", 2)
