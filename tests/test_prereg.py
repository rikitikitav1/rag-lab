from types import SimpleNamespace

import job_specs
import pytest
from evals import columns
from pydantic import ValidationError
from use_cases import prereg


def test_every_outcome_the_stand_writes_has_a_name_in_the_registry():
    # a row the stand can produce and the registry cannot name is a row nobody could preregister on
    assert columns.check_outcomes_are_all_registered() == []


def _row(answer="some text", sources=(), faithfulness=None, **metrics):
    return SimpleNamespace(metrics=metrics, answer=answer, sources=list(sources),
                           faithfulness=faithfulness, question=None)


def test_the_two_readings_of_exhaustion_are_two_names():
    # the ambiguity this registry exists to kill: `finished_by` and `outcome` both say "exhausted"
    row = _row(answer="Не могу ответить.", finished_by="hops_exhausted", outcome="refused")
    assert columns.read("hops_exhausted", row) == 1.0
    assert columns.read("exhausted", row) == 0.0


def test_an_outcome_column_reads_the_outcome_the_stand_settles_not_the_one_it_wrote():
    # the judge found nothing grounded, so the row the answer called answered is answered_ungrounded
    row = _row(sources=["a"], faithfulness=0, outcome="answered")
    assert columns.read("answered_ungrounded", row) == 1.0
    assert columns.read("answered", row) == 0.0
    assert columns.read("refused", _row(outcome="answered", settled_outcome="refused")) == 1.0


def test_the_oldest_rows_read_the_ceiling_the_way_the_outcome_does():
    # rows written before `finished_by` existed carry only their hops
    assert columns.read("hops_exhausted", _row(hops=4, outcome="answered")) == 1.0
    assert columns.read("hops_exhausted", _row(hops=4, failed=True, outcome="error")) == 0.0
    assert columns.read("hops_exhausted", _row(finished_by="hops_exhausted", failed=True)) == 0.0


def test_a_column_the_registry_cannot_name_is_refused_with_the_known_ones():
    with pytest.raises(KeyError, match="known:"):
        columns.read("waste", _row(outcome="refused"))


def test_a_union_of_columns_reads_as_one_predicate():
    exhausted = _row(answer="Не могу ответить.", finished_by="hops_exhausted", outcome="refused")
    assert columns.read_any(["narrated_call", "hops_exhausted"], exhausted) == 1.0
    assert columns.read_any(["narrated_call"], _row(answer="Не могу ответить.", outcome="refused")) == 0.0


def test_a_column_carries_no_direction_of_its_own():
    # `refused` is right out of corpus and a guard in corpus: the declaration says which way is worse
    assert not hasattr(columns.REGISTRY["refused"], "better")


def test_a_closing_run_names_the_promise_it_was_made_under():
    with pytest.raises(ValidationError, match="closing run names"):
        job_specs.EvalRun(run_name="r", set_name="s", purpose=job_specs.Purpose.closing)


def test_a_smoke_owes_nothing_and_is_the_default():
    spec = job_specs.EvalRun(run_name="r", set_name="s")
    assert spec.purpose is job_specs.Purpose.smoke and spec.prereg is None
    named = job_specs.EvalRun(run_name="r", set_name="s",
                              purpose=job_specs.Purpose.closing, prereg="mr4_sgr")
    assert named.prereg == "mr4_sgr"


def test_the_gate_lives_in_the_spec_so_both_doors_get_it():
    # the REST door and the queue both build this model; a gate on a route would cover one of them
    from api.v1 import eval as eval_route

    assert issubclass(eval_route.EvalRunRequest, job_specs.EvalRunFields)
    assert "purpose" in job_specs.EvalRunFields.model_fields


def test_a_promise_refuses_a_column_nobody_can_recompute():
    with pytest.raises(prereg.Refused, match="Known:"):
        prereg._known_or_refuse(["waste"], "closing")
    # and naming nothing is refused too, because a promise without a column promises nothing
    with pytest.raises(prereg.Refused, match="at least one column"):
        prereg._known_or_refuse([], "closing")
    assert prereg._known_or_refuse(["hops_exhausted"], "closing") == ["hops_exhausted"]


def test_a_guard_that_is_not_an_object_or_names_no_column_is_refused_not_crashed():
    with pytest.raises(prereg.Refused, match="each guard is an object"):
        prereg._guard_or_refuse(None, "unsupported_answer")
    with pytest.raises(prereg.Refused, match="named by a string"):
        prereg._guard_or_refuse(None, {"must_not": "rise"})
    with pytest.raises(prereg.Refused, match="must_not"):
        prereg._guard_or_refuse(None, {"column": "unsupported_answer", "direction": "not above"})
    with pytest.raises(prereg.Refused, match="margin"):
        prereg._guard_or_refuse(None, {"column": "unsupported_answer", "must_not": "rise", "margin": -1})


def test_the_effect_is_signed_by_the_declared_direction():
    control = {1: _row(outcome="narrated_call"), 2: _row(outcome="narrated_call")}
    arm = {1: _row(outcome="narrated_call"), 2: _row(sources=["a"], outcome="answered")}
    # the arm lowered the column on one question of two: positive when lowering was promised
    assert prereg._paired(control, arm, {1, 2}, ["narrated_call"], -1) == [0.0, 1.0]
    assert prereg._paired(control, arm, {1, 2}, ["narrated_call"], 1) == [0.0, -1.0]
    # a question outside the declared population is not paired
    assert prereg._paired(control, arm, {1}, ["narrated_call"], -1) == [0.0]


def test_a_floor_run_that_shares_no_question_is_refused_not_crashed():
    with pytest.raises(prereg.Refused, match="floor: no question is shared"):
        prereg._band_or_refuse([], "floor")


def test_a_guard_older_than_its_direction_is_unreadable_not_passed():
    got = prereg._guard({"column": "unsupported_answer", "direction": "not above its floor"}, {}, set())
    assert got["state"] == "unreadable"


def test_cleared_is_always_a_value_with_its_reason():
    holds = {"column": "unsupported_answer", "state": "holds"}
    broken = {"column": "unsupported_answer", "state": "broken"}
    undecided = {"column": "unsupported_answer", "state": "undecided"}
    assert prereg._verdict("cleared", [holds])[0] is True
    # a broken guard is a veto even where the closing column clears
    assert prereg._verdict("cleared", [broken]) == (False, "guard broken: unsupported_answer")
    assert prereg._verdict("cleared", [undecided])[0] is None
    for state in ("no_floor", "no_direction"):
        cleared, why = prereg._verdict(state, [holds])
        assert cleared is None and why
    assert prereg._verdict("missed", [])[0] is False


def test_a_closing_run_under_a_promise_that_does_not_exist_is_refused(monkeypatch):
    import job_queue

    monkeypatch.setattr(prereg, "exists", lambda name: name == "mr4_sgr")
    with pytest.raises(job_specs.Refused, match="no preregistration named 'typo'"):
        job_queue._promise({"purpose": "closing", "prereg": "typo"})
    assert job_queue._promise({"purpose": "closing", "prereg": "mr4_sgr"}) == "mr4_sgr"
    assert job_queue._promise({"purpose": "smoke"}) is None


def test_a_guard_reads_its_bar_on_top_of_the_nights_floor():
    # the night's in-corpus refusals: control 2, arm 7, the control repeat 2 of 820 on other questions
    refused, answered = "Не могу ответить.", "some text"

    def run(refusing):
        return {q: _row(answer=refused if q in refusing else answered, sources=["a"], outcome="answered")
                for q in range(820)}

    rows = {"control": run({0, 1}), "arm": run(set(range(7))), "floor": run({2, 3})}
    guard = {"column": "refused", "must_not": "rise", "margin": 0.0}
    ids = set(range(820))
    alone = prereg._guard(guard, {k: rows[k] for k in ("control", "arm")}, ids)
    assert alone["state"] == "broken"
    # five rows over the noise no longer veto, and they do not pass either: the arm's upper edge is above it
    with_floor = prereg._guard(guard, rows, ids)
    assert with_floor["state"] == "undecided" and with_floor["bar"] == with_floor["floor"]["ci95"][1] > 0
