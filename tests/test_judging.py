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
