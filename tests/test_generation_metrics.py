from types import SimpleNamespace

from evals import generation_metrics

_TEXT = {
    "answered": "the corpus says hello",
    "refused": "I cannot answer this from the available sources",
    "unsupported_answer": "ClickHouse is a columnar database",
    "narrated_call": '{"name": "deepwiki__ask_question", "parameters": {"q": "x"}}',
    "error": "",
}


def _log(marked=None, answered=True, faith=None, rel=None, compl=None, sources=(), outcome=None):
    return SimpleNamespace(
        question=SimpleNamespace(original_text="q", marked_sources=marked or [], kind=None),
        metrics={},
        answered=answered,
        answer=_TEXT.get(outcome, "the corpus says hello"),
        faithfulness=faith,
        relevance=rel,
        completeness=compl,
        sources=[{"source": s} for s in sources],
    )


def _evaluate(monkeypatch, logs):
    monkeypatch.setattr(generation_metrics, "load_logs", lambda run_name: logs)
    return generation_metrics.evaluate("run")


def test_axes_stay_on_the_in_corpus_pool(monkeypatch):
    logs = [
        _log(marked=["a.md"], faith=8, rel=9),
        _log(faith=2, rel=3, sources=["mcp:deepwiki__ask_question"]),
    ]
    m = _evaluate(monkeypatch, logs)
    assert (m["faithfulness"], m["relevance"], m["n_scored"]) == (8, 9, 1)


def test_remote_answers_are_scored_separately(monkeypatch):
    logs = [
        _log(faith=6, rel=8, sources=["mcp:deepwiki__ask_question"]),
        _log(faith=4, rel=6, sources=["mcp:deepwiki__ask_question"]),
        _log(marked=["a.md"], faith=10, rel=10),
    ]
    m = _evaluate(monkeypatch, logs)
    assert (m["remote_grounding"], m["remote_relevance"]) == (5, 7)
    assert (m["n_remote_scored"], m["answered_via_remote"]) == (2, 2)
    assert m["faithfulness"] == 10


def test_refusals_are_counted_instead_of_vanishing(monkeypatch):
    logs = [
        _log(marked=["a.md"], answered=False, outcome="refused"),
        _log(marked=["a.md"], answered=True, faith=7, rel=7, outcome="answered"),
        _log(answered=False, outcome="refused"),
        _log(answered=False, outcome="refused"),
    ]
    m = _evaluate(monkeypatch, logs)
    assert m["answer_rate"] == 0.25
    assert m["false_refusal"] == "1/2"
    assert m["refusal_accuracy"] == "2/2"
    assert m["n_logs"] == 4


def test_an_answer_without_sources_is_not_a_refusal(monkeypatch):
    logs = [
        _log(answered=False, outcome="unsupported_answer"),
        _log(answered=False, outcome="narrated_call"),
        _log(answered=False, outcome="refused"),
    ]
    m = _evaluate(monkeypatch, logs)
    assert m["refusal_accuracy"] == "1/3"
    assert m["unsupported_external"] == "1/3"
    assert m["narrated_calls"] == 1


def test_rejected_questions_leave_the_pools(monkeypatch):
    rejected = _log(answered=False, outcome="refused")
    rejected.question.kind = "rejected"
    logs = [rejected, _log(marked=["a.md"], faith=9, rel=9, outcome="answered")]
    m = _evaluate(monkeypatch, logs)
    assert m["n_logs"] == 1
    assert m["refusal_accuracy"] == "0/0"


def _off_domain(answered=True, sources=(), outcome=None, faith=None):
    log = _log(answered=answered, sources=sources, outcome=outcome, faith=faith)
    log.question.kind = "off_domain"
    return log


def test_off_domain_questions_are_their_own_pool(monkeypatch):
    logs = [
        _off_domain(answered=False, outcome="refused"),
        _off_domain(answered=False, outcome="refused"),
        _off_domain(answered=True, sources=["mcp:deepwiki__ask_question"], outcome="answered"),
        _log(marked=["a.md"], faith=8, rel=8, outcome="answered"),
    ]
    m = _evaluate(monkeypatch, logs)
    assert m["off_domain_refusal"] == "2/3"
    assert m["off_domain_via_remote"] == 1
    assert m["refusal_accuracy"] == "0/0"
    assert m["faithfulness"] == 8


def test_a_narrated_call_is_not_laundered_into_a_refusal(monkeypatch):
    log = _log(answered=False)
    log.answer = "No relevant documents found."
    log.metrics = {"outcome": "narrated_call"}
    m = _evaluate(monkeypatch, [log])
    assert m["narrated_calls"] == 1
    assert m["refusal_accuracy"] == "0/1"


def test_prose_narration_is_caught_through_the_configured_integrations(monkeypatch):
    log = _log(answered=False)
    log.answer = 'I will use deepwiki__ask_question(repoName="x") to answer'
    log.metrics = {"config": {"mcp_configured": ["deepwiki"]}}
    m = _evaluate(monkeypatch, [log])
    assert m["narrated_calls"] == 1


def test_off_domain_answers_are_scored_for_grounding(monkeypatch):
    logs = [_off_domain(faith=0, outcome="answered"), _off_domain(faith=2, outcome="answered")]
    for log in logs:
        log.sources = [{"source": "kotlin-interview-questions/README.md"}]
    m = _evaluate(monkeypatch, logs)
    assert m["off_domain_grounding"] == 1
    assert m["off_domain_refusal"] == "0/2"


def test_running_out_of_hops_is_not_an_error(monkeypatch):
    log = _log(answered=False)
    log.answer = ""
    log.metrics = {"outcome": "error", "hops": 4, "config": {"max_hops": 4}}
    broken = _log(answered=False)
    broken.answer = ""
    broken.metrics = {"outcome": "error", "hops": 1, "config": {"max_hops": 4}}
    m = _evaluate(monkeypatch, [log, broken])
    assert m["outcomes"]["exhausted"] == 1
    assert m["outcomes"]["error"] == 1


def test_a_long_answer_is_not_a_refusal_because_of_one_phrase(monkeypatch):
    log = _log(marked=["a.md"], faith=8, rel=8, sources=["a.md"])
    log.answer = (
        "Redis cannot provide strong consistency guarantees across replicas because "
        "replication is asynchronous. " + "The primary acknowledges a write before the "
        "replicas confirm it, so a failover can lose the tail of the stream. " * 4
    )
    m = _evaluate(monkeypatch, [log])
    assert m["false_refusal"] == "0/1"
    assert m["outcomes"]["answered"] == 1
