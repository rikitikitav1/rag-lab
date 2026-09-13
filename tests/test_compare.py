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
    fallback_opened=False,
):
    metrics = {}
    if fallback_reason:
        metrics["fallback_reason"] = fallback_reason
    if fallback_opened:
        metrics["fallback_opened"] = True
    return SimpleNamespace(
        question_id=question_id,
        question=SimpleNamespace(original_text="q", marked_sources=marked or [], kind=kind),
        metrics=metrics,
        # a real row always has it, and the comparison reads it to say whether one ruler scored both
        prompts={},
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
    assert result["blended_do_not_rank"]["a"]["n"] == 3


def test_rejected_questions_stay_out_of_every_pool():
    rejected = _log(question_id=9, kind="rejected", faith=1)
    result = compare.compare({"a": [rejected, _log(marked=["a.md"], faith=7)]})
    assert result["blended_do_not_rank"]["a"]["n"] == 1
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


def test_the_leak_separates_a_shut_gate_from_a_tool_that_did_not_deliver():
    logs = [
        _log(question_id=1, sources=["a.md"]),
        _log(question_id=2, sources=["a.md"], fallback_reason="weak", fallback_opened=True),
    ]
    arm = compare.compare({"a": logs})["pools"]["out_of_corpus"]["arms"]["a"]
    assert arm["answered_from_corpus"] == 2
    assert arm["answered_from_corpus_gate_shut"] == 1
    assert arm["answered_from_corpus_opened_no_evidence"] == 1


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


def test_identical_arms_report_a_p_of_one_rather_than_nothing():
    # null cannot enter a Holm family, and `stats.wilcoxon_p` already answers this case with 1.0
    logs = [_log(question_id=i, faith=6) for i in range(3)]
    result = compare.paired(logs, logs, "faithfulness")
    assert result["p"] == 1.0
    assert (result["better"], result["worse"]) == (0, 0)
    assert result["ci95"] == [0.0, 0.0]


def test_an_interval_that_spans_zero_is_how_no_difference_is_reported():
    left = [_log(question_id=i, faith=5 + (i % 3)) for i in range(30)]
    right = [_log(question_id=i, faith=5 + ((i + 1) % 3)) for i in range(30)]
    result = compare.paired(left, right, "faithfulness")
    low, high = result["ci95"]
    assert low < 0 < high
    assert abs(result["mean_delta"]) < 0.5


def test_a_shifted_arm_gets_a_significant_p_value():
    left = [_log(question_id=i, faith=3) for i in range(12)]
    right = [_log(question_id=i, faith=8) for i in range(12)]
    result = compare.paired(left, right, "faithfulness")
    assert result["p"] is not None and result["p"] < 0.01


def test_pairs_cover_every_combination_of_arms():
    def arm(faith):
        return [_log(question_id=1, marked=["a.md"], faith=faith)]

    result = compare.compare({"a": arm(5), "b": arm(6), "c": arm(7)})
    pairs = result["pools"]["in_corpus"]["pairs"]
    assert [(p["left"], p["right"]) for p in pairs] == [("a", "b"), ("a", "c"), ("b", "c")]
    assert pairs[0]["faithfulness"]["right"] == 6


def test_one_arm_scored_on_one_axis_cannot_agree_with_an_arm_scored_on_three():
    # the ruler check narrowed to shared axes while its neighbour called partial silence unknown
    from types import SimpleNamespace

    from evals.compare import residencies

    def row(prompts):
        return SimpleNamespace(
            metrics={"faithfulness": {"residency_id": 7, "engine": "ollama:11434"}},
            prompts=prompts, question_id=1, run_name="arm", sources=_sources("a.md"),
        )

    narrow = row({"judge_faithfulness": 2})
    wide = row({f"judge_{axis}": 2 for axis in ("faithfulness", "relevance", "completeness")})
    assert residencies({"a": [narrow], "b": [wide]})["one_judge_prompt"] is None
    assert residencies({"a": [wide], "b": [wide]})["one_judge_prompt"] is True


def _sources(*names):
    return [{"source": n, "hop": None, "vector_rank": i + 1, "keyword_rank": None}
            for i, n in enumerate(names)]


def _judged(engine=None, name=None, rid=7, qid=1, sources=("a.md",)):
    from types import SimpleNamespace

    stamp = {"residency_id": rid, "engine": engine}
    if name:
        stamp["engine_name"] = name
    return SimpleNamespace(
        metrics={"faithfulness": stamp}, prompts={"judge_faithfulness": 1},
        question_id=qid, run_name="arm", sources=_sources(*sources) if sources else None,
    )


def test_the_migration_pair_still_joins(monkeypatch):
    # one arm judged before the engines table, one after, same server: this must stay comparable
    from evals import compare

    monkeypatch.setattr(compare, "registered_names", lambda: {"ollama"})
    got = compare.residencies({
        "before": [_judged(engine="ollama:11434")],
        "after": [_judged(engine="ollama:11434", name="ollama")],
    })
    assert got["one_engine"] is True, "both eras carry the address, so the pair joins"
    assert got["one_engine_name"] is None, "one arm predates the name, and silence is not a match"
    assert "no engine name" in got["read_this_first"], "the reader is told what cannot be said"


def test_two_engines_taking_the_same_port_in_turn_are_told_apart(monkeypatch):
    from evals import compare

    monkeypatch.setattr(compare, "registered_names", lambda: {"ollama", "vllm"})
    got = compare.residencies({
        "left": [_judged(engine="localhost:8000", name="ollama")],
        "right": [_judged(engine="localhost:8000", name="vllm")],
    })
    # the address agrees and the entity does not, which is the case the address cannot show
    assert got["one_engine"] is True and got["one_engine_name"] is False
    assert "same host and port" in got["read_this_first"]


def test_a_pair_judged_on_two_engines_is_refused_and_answering_engines_are_named(monkeypatch):
    # two judges are two rulers
    import pytest
    from evals import compare

    monkeypatch.setattr(compare, "registered_names", lambda: {"ollama", "vllm"})
    with pytest.raises(compare.TwoJudges, match="judged on"):
        compare.compare({"left": [_judged(engine="ollama:11434", name="ollama")],
                         "right": [_judged(engine="vllm:8000", name="vllm")]})
    assert issubclass(compare.TwoJudges, compare.Ambiguous), "every door already answers it 409"

    arm, other = _log(kind="in_corpus", faith=7), _log(kind="in_corpus", faith=8)
    arm.metrics["config"] = {"engines": {"generation": "vllm-cpu", "embedding": "ollama"}}
    other.metrics["config"] = {"engines": {"generation": "ollama", "embedding": "ollama"}}
    got = compare.compare({"cpu": [arm], "gpu": [other]})
    assert got["answering_engines_by_run"] == {
        "cpu": {"embedding": ["ollama"], "generation": ["vllm-cpu"]},
        "gpu": {"embedding": ["ollama"], "generation": ["ollama"]},
    }, "two generators are the treatment: named, not refused"


def test_the_door_carries_every_key_the_comparison_writes():
    # a response model drops any key it does not declare
    from api.v1.eval import CompareResponse
    from evals import compare

    written = set(compare.compare({"a": [_log(kind="in_corpus", faith=7)]}))
    carried = {f.alias or name for name, f in CompareResponse.model_fields.items()}
    assert written <= carried, written - carried


def test_an_engine_deleted_since_the_run_is_named_as_gone(monkeypatch):
    from evals import compare

    monkeypatch.setattr(compare, "registered_names", lambda: set())
    got = compare.residencies({"one": [_judged(engine="localhost:8000", name="vllm")]})
    assert got["engines_gone"] == ["vllm"]


def test_arms_that_retrieved_different_sources_are_not_one_arm_read_twice(monkeypatch):
    # the deterministic half must be equal, not close: the judge's floor covers only the judge
    from evals import compare

    monkeypatch.setattr(compare, "registered_names", lambda: {"ollama"})
    got = compare.residencies({
        "left": [_judged(engine="o:1", name="ollama", sources=("a.md", "b.md"))],
        "right": [_judged(engine="o:1", name="ollama", sources=("b.md", "a.md"))],
    })

    assert got["one_deterministic"] is False, "the same two files in a different order is a change"
    assert "ranked them differently" in got["read_this_first"]
    # a rerank A/B changes retrieval on purpose, and the ruler must not call that arm invalid
    assert "unless retrieval is the treatment" in got["read_this_first"]


def test_the_same_sources_in_the_same_order_leave_the_contrast_readable(monkeypatch):
    from evals import compare

    monkeypatch.setattr(compare, "registered_names", lambda: {"ollama"})
    got = compare.residencies({
        "left": [_judged(engine="o:1", name="ollama", sources=("a.md", "b.md"))],
        "right": [_judged(engine="o:1", name="ollama", sources=("a.md", "b.md"))],
    })

    assert got["one_deterministic"] is True
    assert "one residency is necessary" in got["read_this_first"]


def test_rows_that_recorded_no_sources_say_so_rather_than_agreeing(monkeypatch):
    from evals import compare

    monkeypatch.setattr(compare, "registered_names", lambda: {"ollama"})
    got = compare.residencies({
        "left": [_judged(engine="o:1", name="ollama", sources=None)],
        "right": [_judged(engine="o:1", name="ollama", sources=("a.md",))],
    })

    assert got["one_deterministic"] is None, "silence on one side is not a match"
    assert "cannot be said" in got["read_this_first"]


def test_a_differing_engine_is_read_before_the_sources_that_differ_with_it(monkeypatch):
    # the order matters: a different engine explains different sources, and not the other way round
    from evals import compare

    monkeypatch.setattr(compare, "registered_names", lambda: {"ollama", "vllm"})
    got = compare.residencies({
        "left": [_judged(engine="o:1", name="ollama", sources=("a.md",))],
        "right": [_judged(engine="v:1", name="vllm", sources=("b.md",))],
    })

    assert (got["one_engine"], got["one_deterministic"]) == (False, False)
    assert "differently named engines" in got["read_this_first"]


def test_the_door_carries_the_disqualification_and_not_only_the_means(monkeypatch):
    # the response model listed three fields, so the route dropped the residency without a word
    import bootstrap

    monkeypatch.setattr(bootstrap, "bootstrap_models", lambda: None)

    import server
    from api.v1 import eval as eval_route
    from fastapi.testclient import TestClient

    full = {
        "schema": 9, "runs": ["a", "b"], "pools": {}, "blended_do_not_rank": {},
        "verdicts": {"comparable": 3, "disagree": 1},
        "correlation_population": {"predicate": "p", "n": {}},
        "answering_engines_by_run": {},
        "residency": {"one_residency": False, "read_this_first": "not comparable"},
    }
    monkeypatch.setattr(eval_route.compare_uc, "compare", lambda runs: full)

    from orm.async_db import get_session

    class _Session:
        async def scalars(self, _stmt):
            class _R:
                def all(self):
                    return [object()]

            return _R()

    async def _yield():
        yield _Session()

    server.app.dependency_overrides[get_session] = _yield
    try:
        with TestClient(server.app) as client:
            got = client.get("/v1/eval/compare", params={"runs": ["a", "b"]})
    finally:
        server.app.dependency_overrides.clear()

    assert got.status_code == 200, got.text
    body = got.json()
    assert body["residency"]["read_this_first"] == "not comparable"
    assert body["verdicts"] == {"comparable": 3, "disagree": 1}, "the pair's number reaches the door"
    assert body["schema"] == 9, "a reader cannot tell two eras of this record apart without it"


def test_a_run_that_cannot_be_paired_is_refused_by_name_not_by_a_bare_500(monkeypatch):
    # `tok_probe` holds one question three times; the reason names it and the door swallowed it
    import bootstrap

    monkeypatch.setattr(bootstrap, "bootstrap_models", lambda: None)

    import server
    from api.v1 import eval as eval_route
    from evals.pools import Ambiguous
    from fastapi.testclient import TestClient
    from orm.async_db import get_session

    def boom(_runs):
        raise Ambiguous("question 77529 appears twice in run 'tok_probe'")

    monkeypatch.setattr(eval_route.compare_uc, "compare", boom)

    class _Session:
        async def scalars(self, _stmt):
            class _R:
                def all(self):
                    return [object()]

            return _R()

    async def _yield():
        yield _Session()

    server.app.dependency_overrides[get_session] = _yield
    try:
        with TestClient(server.app) as client:
            got = client.get("/v1/eval/compare", params={"runs": ["a", "b"]})
    finally:
        server.app.dependency_overrides.clear()

    assert got.status_code == 409, got.text
    assert "77529" in got.json()["detail"], "the caller is told which question, not just that it failed"


def _scored(question_id, faith, rel, tokens=100, elapsed=2.0):
    ql = _log(question_id=question_id, faith=faith, rel=rel)
    ql.metrics = {axis: {"judge_prompt_tokens": tokens, "elapsed": elapsed}
                  for axis in ("faithfulness", "relevance")}
    return ql


def test_verdicts_count_a_moved_ruler_that_the_means_hide():
    # 1 to 0 on one row and 0 to 1 on another: the means hold still, two verdicts moved
    left = [_scored(1, 1, 1), _scored(2, 0, 1), _scored(3, 1, 1)]
    right = [_scored(1, 0, 1), _scored(2, 1, 1), _scored(3, 1, 1)]
    seen = compare.verdicts(left, right)

    faith = seen["axes"]["faithfulness"]
    assert faith["left"] == faith["right"]
    assert (faith["comparable"], faith["disagree"]) == (3, 2)
    assert (seen["comparable"], seen["disagree"], seen["disagree_rate"]) == (6, 2, 0.333)
    assert seen["axes"]["completeness"]["comparable"] == 0


def test_a_verdict_scored_on_one_side_is_neither_a_match_nor_a_clash():
    # one side unscored is no clash, and it stays out of the denominator too
    left = [_scored(1, 1, None), _scored(2, 1, 1)]
    right = [_scored(1, 1, 1), _scored(2, 1, 0)]
    rel = compare.verdicts(left, right)["axes"]["relevance"]
    assert (rel["comparable"], rel["disagree"], rel["one_sided"]) == (1, 1, 1)


def test_verdicts_pair_only_shared_questions_and_say_where_the_judge_read_less():
    left = [_scored(1, 1, 1, tokens=900, elapsed=2.0), _scored(2, 1, 1), _scored(9, 0, 0)]
    right = [_scored(1, 1, 1, tokens=400, elapsed=4.0), _scored(2, 1, 1), _scored(8, 1, 1)]
    seen = compare.verdicts(left, right)
    assert seen["questions"] == 2, "a question one arm never asked is not a verdict to compare"
    faith = seen["axes"]["faithfulness"]
    assert faith["prompt_tokens_differ"] == 1
    assert (faith["seconds_left"], faith["seconds_right"]) == (2.0, 3.0)


def test_compare_carries_verdicts_for_a_pair_and_none_for_more():
    rows = [_scored(1, 1, 1)]
    assert compare.compare({"a": rows, "b": rows})["verdicts"]["comparable"] == 2
    assert compare.compare({"a": rows, "b": rows, "c": rows})["verdicts"] is None


def test_a_remote_judge_is_read_as_having_no_residency_and_not_as_an_old_row():
    from evals import compare

    said = compare._what_to_read_first(True, True, None, True, True, remote_judge=True)
    assert "remote judge has no residency" in said
    assert "before this was recorded" in compare._what_to_read_first(True, True, None, True, True)
