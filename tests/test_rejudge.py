from types import SimpleNamespace

import pytest
from models.eval import QuestionLog
from use_cases import rejudge


def _picked(source="a", target="b") -> dict[str, str]:
    """What each carried column is filled from, by name rather than by position."""
    statement = rejudge.copy_statement(source, target)
    carried = rejudge.carried_columns()
    chosen = list(statement.select.selected_columns)
    assert len(chosen) == len(carried), "insert-from-select is positional; lengths must match"
    return {
        name: str(col.compile(compile_kwargs={"literal_binds": True}))
        for name, col in zip(carried, chosen, strict=True)
    }


def test_a_copy_carries_every_column_but_the_key():
    carried = rejudge.carried_columns()
    every = {c.name for c in QuestionLog.__table__.columns}
    assert set(carried) == every - {"id"}
    # a column added by a migration must appear in the copy without anyone editing this
    assert "created_at" in carried and "context" in carried and "sources" in carried


def test_the_copy_is_unjudged():
    picked = _picked()
    for axis in rejudge.AXES:
        assert picked[axis] == "NULL", f"{axis} must arrive empty, got {picked[axis]}"


def test_the_copy_keeps_what_produced_the_answer_and_drops_what_scored_it():
    picked = _picked()
    for axis in rejudge.AXES:
        assert f"'{axis}'" in picked["metrics"], f"metrics must lose the {axis} verdict"
        assert f"'judge_{axis}'" in picked["prompts"], f"prompts must lose judge_{axis}"
    assert "'judging'" in picked["models"], "models must lose the judge"
    # config and retrieval describe the run that produced the answer, and stay
    assert "'config'" not in picked["metrics"] and "'retrieval'" not in picked["metrics"]
    # and the answer itself is carried untouched
    assert picked["answer"] == "question_logs.answer"
    assert picked["context"] == "question_logs.context"
    assert picked["created_at"] == "question_logs.created_at"


def test_a_copy_names_its_target_run():
    # insert-from-select pairs the two lists by position, so a value lands in the wrong column
    assert _picked(target="b")["run_name"] == "'b'"


@pytest.mark.parametrize(
    "source,target,why",
    [("", "b", "no source"), ("a", "", "no target"), ("a", "a", "same name")],
)
def test_a_copy_refuses_before_touching_the_database(source, target, why):
    with pytest.raises(ValueError):
        rejudge.copy_run(source, target)


def test_the_strip_list_matches_what_the_judge_writes():
    # the judge writes `prompts[purpose.name]` and the reader looked for another key
    from models.registry import Purpose

    written = {Purpose[f"judge_{axis}"].name for axis in rejudge.AXES}
    removed = {f"judge_{axis}" for axis in rejudge.AXES}
    assert written == removed


def test_a_rejudge_refuses_an_axis_that_is_not_the_judge():
    with pytest.raises(ValueError, match="cannot move"):
        rejudge.validate_axes({"variant": ["clean_1024", "baseline"]})


def test_a_rejudge_with_no_axes_would_compare_an_arm_against_itself():
    with pytest.raises(ValueError, match="no axes"):
        rejudge.validate_axes({})


def test_a_prompt_axis_takes_a_version_rather_than_a_name():
    with pytest.raises(ValueError, match="prompt versions"):
        rejudge.validate_axes({"judge_relevance": ["v2"]})


def test_a_judge_model_takes_the_shape_its_sibling_door_demands():
    with pytest.raises(ValueError, match="model names"):
        rejudge.validate_axes({"judge_model": [7]})
    with pytest.raises(ValueError, match="model names"):
        rejudge.validate_axes({"judge_model": ["qwen\n2.5:7b"]})
    # the case a `match` on a `$`-anchored pattern lets through, hence `fullmatch`
    with pytest.raises(ValueError, match="model names"):
        rejudge.validate_axes({"judge_model": ["qwen2.5:7b\n"]})
    with pytest.raises(ValueError, match="model names"):
        rejudge.validate_axes({"judge_model": ["q" * 129]})
    rejudge.validate_axes({"judge_model": ["qwen2.5:7b", "qwen3:4b"]})


def test_the_judge_axes_are_accepted():
    rejudge.validate_axes({"judge_faithfulness": [2, 3]})


def test_an_arm_becomes_a_bench_and_a_job():
    from models.registry import Purpose

    arm = {"judge_model": "qwen3:4b", "judge_faithfulness": 3}
    bench = rejudge.arm_bench(arm)
    assert bench.model == "qwen3:4b"
    assert bench.versions == {Purpose.judge_faithfulness: 3}
    assert rejudge.arm_options(arm, "arm_1") == {
        "run_name": "arm_1",
        "judge_model": "qwen3:4b",
        "judge_prompts": {"judge_faithfulness": 3},
    }


def test_an_arm_without_a_model_leaves_the_role_alone():
    bench = rejudge.arm_bench({"judge_relevance": 2})
    assert bench.model is None


def test_a_paired_delta_reports_its_own_spread():
    before = {i: {"relevance": 5} for i in range(40)}
    after = {i: {"relevance": 5 + (i % 2)} for i in range(40)}
    got = rejudge._paired(before, after, "relevance")
    assert got["n"] == 40
    assert got["delta"] == 0.5
    assert got["better"] == 20 and got["worse"] == 0
    assert got["ci95"][0] > 0


def test_a_delta_over_nothing_is_nothing_rather_than_zero():
    assert rejudge._paired({1: {}}, {1: {}}, "relevance") is None


def test_the_control_arm_is_expressible():
    # the first arm anyone wants is one run judged twice; without a label the two collapse
    rejudge.validate_axes({"repeat": [1, 2]})
    arm = {"repeat": 2}
    assert rejudge.arm_bench(arm) == rejudge.judge.Bench(model=None, versions=None)
    assert rejudge.arm_options(arm, "arm_2") == {
        "run_name": "arm_2",
        "judge_model": None,
        "judge_prompts": {},
    }


def test_repeat_labels_have_to_differ():
    with pytest.raises(ValueError, match="must differ"):
        rejudge.validate_axes({"repeat": [1, 1]})


def test_repeat_rides_along_with_a_real_axis_without_reaching_the_bench():
    from models.registry import Purpose

    bench = rejudge.arm_bench({"repeat": 2, "judge_faithfulness": 3})
    assert bench.versions == {Purpose.judge_faithfulness: 3}


def test_a_copy_may_not_claim_a_judge_the_row_has_not_had():
    # the judge writes `models["judging"]` and the reader looked for another key
    from job_handlers import judging

    snapshot = judging._Snapshot({}, {}, {})
    judging._judge_axis(
        type("L", (), {"id": 1, "relevance": None})(), snapshot, "relevance", True, False,
        lambda *a: _verdict_for_contract(), (),
    )
    assert set(snapshot.models) == {rejudge.JUDGE_MODEL_KEY}


def _verdict_for_contract():
    from models.registry import Purpose
    from use_cases.judge import Verdict

    return Verdict(
        reason="because", score=8, model="qwen2.5:7b",
        purpose=Purpose.judge_relevance, prompt_version=2,
    )


def test_a_fanout_is_capped_by_the_work_it_makes_not_the_arms_it_names(monkeypatch):
    # the count is stubbed: a real run would make this pass or fail on the machine
    monkeypatch.setattr(rejudge, "_source_rows", lambda source, ids=None: 800)
    rejudge.refuse_oversized_fanout("any", 5)
    with pytest.raises(ValueError, match="over the cap"):
        rejudge.refuse_oversized_fanout("any", 6)


def test_a_rejudge_names_a_source_run_where_other_kinds_name_a_dataset():
    from api.v1.experiment import ExperimentCreate
    from models.experiment import ExperimentKind

    # the call printed in both readmes: it names no dataset, and used to be a 422
    made = ExperimentCreate(
        kind=ExperimentKind.rejudge, name="judge_noise", source_run="some_run",
        param="repeat", axes={"repeat": [1, 2]},
    )
    assert made.dataset is None
    assert made.param_values == [1, 2]


def test_every_other_kind_still_has_to_name_its_dataset():
    from api.v1.experiment import ExperimentCreate

    with pytest.raises(ValueError, match="dataset is required"):
        ExperimentCreate(param="k", param_values=[1, 3])


def test_arms_are_paired_with_their_runs_by_the_record_not_by_order():
    # adding a value to one axis moves the product, and the old derivation zipped positions
    exp = SimpleNamespace(
        axes={"judge_faithfulness": [2, 3], "repeat": [1]},
        run_names=["r_a", "r_b", "r_c"],
        procedure={
            "arms": [
                {"arm": {"judge_faithfulness": 2, "repeat": 1}, "run": "r_a"},
                {"arm": {"judge_faithfulness": 3, "repeat": 1}, "run": "r_b"},
                {"arm": {"judge_faithfulness": 2, "repeat": 2}, "run": "r_c"},
            ]
        },
    )
    assert rejudge.paired_arms(exp) == [
        ({"judge_faithfulness": 2, "repeat": 1}, "r_a"),
        ({"judge_faithfulness": 3, "repeat": 1}, "r_b"),
        ({"judge_faithfulness": 2, "repeat": 2}, "r_c"),
    ]


def test_an_experiment_recorded_before_the_mapping_still_reads():
    exp = SimpleNamespace(
        axes={"repeat": [1, 2]}, run_names=["r_1", "r_2"], procedure={"source_run": "s"}
    )
    assert rejudge.paired_arms(exp) == [({"repeat": 1}, "r_1"), ({"repeat": 2}, "r_2")]


def test_added_arms_are_checked_like_a_declared_grid():
    with pytest.raises(ValueError, match="cannot move"):
        rejudge.validate_arms([{"k": 5}])
    with pytest.raises(ValueError, match="distinct names"):
        rejudge.validate_arms([{"judge_relevance": 3}, {"judge_relevance": 3}])
    with pytest.raises(ValueError, match="no arms"):
        rejudge.validate_arms([])
    rejudge.validate_arms([{"judge_relevance": 3}, {"judge_model": "qwen2.5:7b"}])


def test_folded_axes_keep_the_order_they_were_named_in():
    assert rejudge.folded_axes(
        [{"repeat": 2, "judge_relevance": 3}, {"repeat": 3, "judge_relevance": 3}]
    ) == {"repeat": [2, 3], "judge_relevance": [3]}


def _fake_reads(monkeypatch, scored: dict[str, dict]):
    monkeypatch.setattr(rejudge, "_scored", lambda name: scored.get(name, {}))
    monkeypatch.setattr(rejudge, "answers_digest", lambda name, ids=None: "sha256:same:2")
    monkeypatch.setattr(rejudge, "_judged_by", lambda name: {"model": ["stub"], "prompts": None})


def test_one_arm_is_compared_against_the_run_it_copied(monkeypatch):
    # the type takes two arms, so a single reading was paid for with an arm nobody wanted
    _fake_reads(
        monkeypatch,
        {
            "src": {1: {"relevance": 5}, 2: {"relevance": 5}},
            "arm_v3": {1: {"relevance": 6}, 2: {"relevance": 7}},
        },
    )
    got = rejudge.compute_results("src", "judge_relevance", [({"judge_relevance": 3}, "arm_v3")])
    assert got["source_scored"] == {
        "n": 2, "judge": {"model": ["stub"], "prompts": None},
        "faithfulness": None, "relevance": 5.0, "completeness": None,
    }
    assert list(got["deltas"]) == ["src_vs_arm_v3"]
    assert got["deltas"]["src_vs_arm_v3"]["relevance"]["delta"] == 1.5
    assert got["deltas"]["src_vs_arm_v3"]["same_answers"]


def test_neighbouring_arms_are_compared_to_each_other(monkeypatch):
    _fake_reads(
        monkeypatch,
        {
            "a": {1: {"relevance": 5}},
            "b": {1: {"relevance": 6}},
            "c": {1: {"relevance": 8}},
        },
    )
    got = rejudge.compute_results(
        "src", "repeat", [({"repeat": 1}, "a"), ({"repeat": 2}, "b"), ({"repeat": 3}, "c")]
    )
    assert got["pairing"] == "every pair"
    assert list(got["deltas"]) == ["a_vs_b", "a_vs_c", "b_vs_c"]
    assert got["deltas"]["b_vs_c"]["relevance"]["delta"] == 2.0


def test_a_wide_grid_falls_back_to_one_base_and_says_so(monkeypatch):
    scored = {f"a{i}": {1: {"relevance": i}} for i in range(rejudge.PAIR_EVERY_UP_TO + 1)}
    _fake_reads(monkeypatch, scored)
    got = rejudge.compute_results(
        "src", "repeat", [({"repeat": i}, name) for i, name in enumerate(scored)]
    )
    assert got["pairing"] == "against a0"
    assert len(got["deltas"]) == len(scored) - 1


def test_the_source_arm_names_the_judge_that_scored_it(monkeypatch):
    # the source is read as an arm, so a pair not naming its judge compares an unnamed one
    rows = [
        ({"judging": "qwen2.5:7b"}, {"judge_relevance": 2}),
        ({"judging": "qwen2.5:7b"}, {"judge_relevance": 2}),
        ({}, {}),
    ]

    class _Session:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def execute(self, *a, **kw):
            return SimpleNamespace(all=lambda: rows)

    monkeypatch.setattr(rejudge, "Session", _Session)
    assert rejudge._judged_by("any") == {
        "model": ["qwen2.5:7b"],
        "prompts": {"relevance": [2]},
    }


def test_the_cap_counts_the_arms_the_experiment_already_holds(monkeypatch):
    # counting only the new arms let a caller post them one at a time and walk past the cap
    monkeypatch.setattr(rejudge, "_source_rows", lambda source, ids=None: 800)
    rejudge.refuse_oversized_fanout("run", 2, existing=0)
    with pytest.raises(ValueError, match="over the cap"):
        rejudge.refuse_oversized_fanout("run", 2, existing=4)


def test_an_arm_label_that_would_not_fit_a_run_name_is_refused():
    assert rejudge.LABEL_RE.fullmatch("strict_v2")
    assert not rejudge.LABEL_RE.fullmatch("v" * 65)
    assert not rejudge.LABEL_RE.fullmatch("_leading")
    assert not rejudge.LABEL_RE.fullmatch("has space")


def test_the_family_of_a_rejudge_report_is_corrected_and_named():
    # four hypotheses and one control read at 0.05 each is how a family becomes significant
    from use_cases.rejudge import _annotate_family

    deltas = {
        "repeat_vs_a": {
            "faithfulness": {"p": 0.004},
            "relevance": {"p": 0.0006},
            "completeness": None,
            "halves": {},
            "same_answers": True,
        },
        "repeat_vs_b": {
            "faithfulness": {"p": 0.0000007},
            "relevance": {"p": 0.28},
            "completeness": None,
            "halves": {},
            "same_answers": True,
        },
    }
    out = _annotate_family(deltas)

    assert (out["tests"], out["method"]) == (4, "holm")
    assert out["family"], "the family is stated, so a narrower pre-registered one can be seen"
    kept = {
        f"{pair}/{axis}": stats["significant_holm"]
        for pair, axes in deltas.items()
        for axis, stats in axes.items()
        if axis in ("faithfulness", "relevance") and stats
    }
    assert kept == {
        "repeat_vs_a/faithfulness": True,
        "repeat_vs_a/relevance": True,
        "repeat_vs_b/faithfulness": True,
        "repeat_vs_b/relevance": False,
    }
    # neither `halves` nor `same_answers` is a hypothesis, and a None axis is not a test
    assert deltas["repeat_vs_a"]["completeness"] is None


def _judged(model=("qwen2.5:7b",), faith=(2,), rel=(2,)):
    return {"model": list(model), "prompts": {"faithfulness": list(faith), "relevance": list(rel)}}


def test_a_rejudge_without_an_arm_reproducing_the_source_judge_is_refused(monkeypatch):
    # a source judged days earlier carries that pass's drift into the delta
    from use_cases import rejudge

    monkeypatch.setattr(rejudge, "_judged_by", lambda run: _judged())
    monkeypatch.setattr(
        rejudge, "_effective_judge",
        lambda arm: {
            "model": arm.get("judge_model") or "qwen2.5:7b",
            "prompts": {
                axis: arm.get(f"judge_{axis}") or 3 for axis in ("faithfulness", "relevance")
            },
        },
    )

    with pytest.raises(ValueError, match="no arm reproduces the judge"):
        rejudge.refuse_unpaired_rejudge("src", [{"judge_faithfulness": 4, "judge_relevance": 4}])

    # the control is an arm judging the same rows with the source's own versions
    rejudge.refuse_unpaired_rejudge(
        "src",
        [{"judge_faithfulness": 4, "judge_relevance": 4},
         {"repeat": "2", "judge_faithfulness": 2, "judge_relevance": 2}],
    )


def test_a_source_carrying_no_judge_at_all_says_so_rather_than_guessing(monkeypatch):
    # rows written before the judge was stamped: the record cannot name what scored them
    from use_cases import rejudge

    monkeypatch.setattr(rejudge, "_judged_by", lambda run: {"model": None, "prompts": None})

    with pytest.raises(ValueError, match="carry no judge at all"):
        rejudge.refuse_unpaired_rejudge("src", [{"judge_faithfulness": 4}])
    rejudge.refuse_unpaired_rejudge("src", [{"judge_faithfulness": 4}], unpaired=True)

    # a `repeat` arm is the human saying which judge they believe produced the source
    rejudge.refuse_unpaired_rejudge(
        "src", [{"judge_faithfulness": 4}, {"repeat": "2", "judge_faithfulness": 2}]
    )


def test_a_source_judged_by_two_versions_cannot_be_paired_at_all(monkeypatch):
    from use_cases import rejudge

    monkeypatch.setattr(rejudge, "_judged_by", lambda run: _judged(faith=(2, 3)))

    with pytest.raises(ValueError, match="more than one judge prompt version"):
        rejudge.refuse_unpaired_rejudge("src", [{"judge_faithfulness": 2, "judge_relevance": 2}])


def test_same_answers_asks_about_the_rows_a_pair_shares(monkeypatch):
    # an arm judged on a sample differs in size alone, and whole-run digests then disagree
    asked = []

    def digest(name, ids=None):
        asked.append((name, tuple(ids or ())))
        return "sha256:same:10"

    monkeypatch.setattr(rejudge, "answers_digest", digest)
    loaded = {"arm": {1: {}, 2: {}, 3: {}}, "ctl": {2: {}, 3: {}}, "src": {}}

    assert rejudge._same_answers("arm", "ctl", "src", loaded)
    assert all(ids == (2, 3) for _, ids in asked), "the shared rows, not either whole run"
    assert {name for name, _ in asked} == {"arm", "ctl", "src"}


def test_a_pair_with_nothing_in_common_is_not_the_same_answers(monkeypatch):
    monkeypatch.setattr(rejudge, "answers_digest", lambda name, ids=None: "sha256:same:0")
    assert not rejudge._same_answers("a", "b", "src", {"a": {1: {}}, "b": {2: {}}, "src": {}})


def test_the_axes_an_arm_does_not_move_are_its_controls():
    # a round changing the faithfulness prompt is checked by completeness not moving
    assert rejudge.control_axes({"judge_faithfulness": 4, "judge_relevance": 4}) == [
        "completeness"
    ]
    assert rejudge.control_axes({"repeat": "2"}) == list(rejudge.AXES)

    plain = rejudge.arm_options({"judge_faithfulness": 4}, "arm")
    assert "control_axes" not in plain, "nothing changes unless a sample is asked for"

    sampled = rejudge.arm_options({"judge_faithfulness": 4}, "arm", control_sample=200)
    assert sampled["control_axes"] == ["relevance", "completeness"]
    assert sampled["control_sample"] == 200


def test_an_arm_reproduces_nothing_when_the_source_never_said_what_judged_it():
    from use_cases import rejudge

    unstamped = {"model": ["qwen2.5:7b"], "prompts": None}
    assert rejudge._reproduces({"judge_model": "qwen2.5:7b"}, unstamped) is False

    no_model = {"model": None, "prompts": {"faithfulness": [2]}}
    assert rejudge._reproduces({"judge_faithfulness": 2}, no_model) is False
