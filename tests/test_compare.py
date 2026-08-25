from types import SimpleNamespace

from evals import compare

_TEXT = {
    "answered": "the corpus says hello",
    "refused": "I cannot answer this from the available sources",
    "narrated_call": '{"name": "deepwiki__ask_question", "parameters": {"q": "x"}}',
}


def _log(
    question_id=1,
    kind=None,
    marked=None,
    faith=None,
    rel=None,
    sources=(),
    outcome="answered",
    elapsed=None,
    fallback_reason=None,
):
    return SimpleNamespace(
        question_id=question_id,
        question=SimpleNamespace(original_text="q", marked_sources=marked or [], kind=kind),
        metrics={"fallback_reason": fallback_reason} if fallback_reason else {},
        answered=outcome == "answered",
        answer=_TEXT.get(outcome, _TEXT["answered"]),
        faithfulness=faith,
        relevance=rel,
        completeness=None,
        sources=[{"source": s} for s in sources],
        elapsed=elapsed,
    )


def test_arms_are_split_by_pool():
    runs = {
        "a": [
            _log(question_id=1, marked=["a.md"], faith=8, rel=9),
            _log(question_id=2, faith=4, rel=6),
            _log(question_id=3, kind="off_domain", faith=0, rel=9),
        ],
    }
    result = compare.compare(runs)
    assert set(result["pools"]) == {"in_corpus", "out_of_corpus", "off_domain"}
    assert result["pools"]["in_corpus"]["arms"]["a"]["faithfulness"] == 8
    assert result["pools"]["out_of_corpus"]["arms"]["a"]["n"] == 1
    assert result["overall"]["a"]["n"] == 3


def test_rejected_questions_stay_out_of_every_pool():
    rejected = _log(question_id=9, kind="rejected", faith=1)
    result = compare.compare({"a": [rejected, _log(marked=["a.md"], faith=7)]})
    assert result["overall"]["a"]["n"] == 1
    assert "rejected" not in result["pools"]


def test_an_in_corpus_question_without_marked_sources_lands_outside():
    result = compare.compare({"a": [_log(kind="in_corpus", faith=5)]})
    assert result["pools"]["out_of_corpus"]["arms"]["a"]["n"] == 1
    assert "in_corpus" not in result["pools"]


def test_answers_that_never_left_the_corpus_are_the_leak_metric():
    logs = [
        _log(question_id=1, sources=["a.md"]),
        _log(question_id=2, sources=["mcp:deepwiki__ask_question"]),
        _log(question_id=3, outcome="refused"),
    ]
    arm = compare.compare({"a": logs})["pools"]["out_of_corpus"]["arms"]["a"]
    assert (arm["answered_from_corpus"], arm["answered_via_remote"]) == (1, 1)
    assert arm["answered_from_corpus_rate"] == 0.333


def test_a_narrated_call_is_not_an_answer_from_the_corpus():
    logs = [_log(question_id=1, sources=["a.md"], outcome="narrated_call")]
    arm = compare.compare({"a": logs})["pools"]["out_of_corpus"]["arms"]["a"]
    assert arm["answered_from_corpus"] == 0
    assert arm["outcomes"]["narrated_call"] == 1


def test_latency_reports_both_the_mean_and_the_median():
    logs = [_log(question_id=i, elapsed=e) for i, e in enumerate([1.0, 2.0, 30.0])]
    arm = compare.compare({"a": logs})["pools"]["out_of_corpus"]["arms"]["a"]
    assert (arm["latency_avg"], arm["latency_p50"]) == (11.0, 2.0)


def test_the_gate_counts_every_reason_it_can_fire_for():
    logs = [
        _log(question_id=1, fallback_reason="weak"),
        _log(question_id=2, fallback_reason="empty"),
        _log(question_id=3, fallback_reason="off_topic"),
        _log(question_id=4, fallback_reason="none"),
    ]
    arm = compare.compare({"a": logs})["pools"]["out_of_corpus"]["arms"]["a"]
    assert arm["gate_fired"] == 3


def test_paired_test_only_keeps_questions_judged_on_both_sides():
    left = [_log(question_id=1, faith=5), _log(question_id=2, faith=7)]
    right = [_log(question_id=1, faith=8), _log(question_id=3, faith=9)]
    result = compare.paired(left, right, "faithfulness")
    assert (result["n"], result["left"], result["right"]) == (1, 5, 8)
    assert (result["better"], result["worse"]) == (1, 0)


def test_identical_arms_get_no_p_value():
    logs = [_log(question_id=i, faith=6) for i in range(3)]
    result = compare.paired(logs, logs, "faithfulness")
    assert result["p_value"] is None
    assert (result["better"], result["worse"]) == (0, 0)


def test_a_shifted_arm_gets_a_significant_p_value():
    left = [_log(question_id=i, faith=3) for i in range(12)]
    right = [_log(question_id=i, faith=8) for i in range(12)]
    result = compare.paired(left, right, "faithfulness")
    assert result["p_value"] is not None and result["p_value"] < 0.01


def test_pairs_cover_every_combination_of_arms():
    def arm(faith):
        return [_log(question_id=1, marked=["a.md"], faith=faith)]

    result = compare.compare({"a": arm(5), "b": arm(6), "c": arm(7)})
    pairs = result["pools"]["in_corpus"]["pairs"]
    assert [(p["left"], p["right"]) for p in pairs] == [("a", "b"), ("a", "c"), ("b", "c")]
    assert pairs[0]["faithfulness"]["right"] == 6
