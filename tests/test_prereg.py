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
    alone = prereg._guard(guard, {"run": {k: rows[k] for k in ("control", "arm")}}, ids)
    assert alone["state"] == "broken"
    # five rows over the noise no longer veto, and they do not pass either: the arm's upper edge is above it
    with_floor = prereg._guard(guard, {"run": rows}, ids)
    assert with_floor["state"] == "undecided" and with_floor["bar"] == with_floor["floor"]["ci95"][1] > 0


def _measured(classes, kept, chars=None):
    addresses = [f"a{n}" for n in range(len(classes))]
    row = {"question_id": 1, "classes": classes, "addresses": addresses, "kept": kept,
           "arm": set(addresses)}
    return row | ({"chars": chars} if chars else {})


def test_the_grader_columns_read_what_the_curve_reads():
    row = _measured(["gold_section", "stranger", "stranger"], kept=[0, 1])
    assert columns.read("gold_retained", row) == 1.0
    assert columns.read("strangers_dropped", row) == 0.5
    # a row with no stranger has no share of strangers dropped, not a zero
    assert columns.read("strangers_dropped", _measured(["gold_section"], kept=[0])) is None
    # a chunk outside the serving arm is not counted, whatever the grader said of it
    outside = _measured(["gold_section", "stranger"], kept=[0]) | {"arm": {"a0"}}
    assert columns.read("strangers_dropped", outside) is None


def test_a_row_the_strip_never_touched_has_no_chars_removed_rather_than_none_removed():
    assert columns.read("chars_removed_gold", _measured(["gold_section"], kept=[0])) is None
    stripped = _measured(["gold_section", "stranger", "stranger"], kept=[0, 1],
                         chars={"before": [1000, 800, 900], "after": [900, 200, 900]})
    assert columns.read("chars_removed_gold", stripped) == 0.1
    # the dropped stranger is counted by strangers_dropped, not twice as characters removed
    assert columns.read("chars_removed_stranger", stripped) == 0.75
    assert columns.read("chars_removed_neighbour", stripped) is None


def test_a_judge_score_column_leaves_an_abstained_row_out():
    assert columns.read("faithfulness", _row(faithfulness="7")) == 7.0
    assert columns.read("faithfulness", _row(faithfulness=None)) is None
    control = {1: _row(faithfulness="6"), 2: _row(faithfulness=None)}
    arm = {1: _row(faithfulness="8"), 2: _row(faithfulness="9")}
    assert prereg._paired(control, arm, {1, 2}, ["faithfulness"], 1) == [2.0]


def test_a_closing_refuses_a_score_in_a_union_and_columns_from_two_sources():
    with pytest.raises(prereg.Refused, match="closes alone"):
        prereg._closing_or_refuse({"columns": ["faithfulness", "refused"], "arm_should": "raise"})
    with pytest.raises(prereg.Refused, match="one source"):
        prereg._closing_or_refuse({"columns": ["refused", "gold_retained"], "arm_should": "raise"})
    with pytest.raises(prereg.Refused, match="floor_value"):
        prereg._closing_or_refuse({"columns": ["faithfulness"], "arm_should": "raise", "floor_value": -1})


def test_a_veto_is_read_inside_one_arm_by_point():
    veto = prereg._veto_or_refuse({"column": "chars_removed_gold", "above": "chars_removed_stranger",
                                   "margin": 0.05, "min_rows": 1})
    assert veto["on"] == "arm"
    cuts_gold = _measured(["gold_section", "stranger"], kept=[0, 1],
                          chars={"before": [100, 100], "after": [50, 90]})
    rows = {"run": {}, "measurement": {"arm": {1: cuts_gold}}}
    assert prereg._veto(veto, rows, {1})["state"] == "fired"
    gentle = cuts_gold | {"chars": {"before": [100, 100], "after": [96, 90]}}
    assert prereg._veto(veto, {"run": {}, "measurement": {"arm": {1: gentle}}}, {1})["state"] == "quiet"
    unstripped = _measured(["gold_section", "stranger"], kept=[0, 1])
    got = prereg._veto(veto, {"run": {}, "measurement": {"arm": {1: unstripped}}}, {1})
    assert got["state"] == "unreadable"
    assert prereg._veto(veto, {"run": {}, "measurement": {}}, {1})["state"] == "not_run"
    with pytest.raises(prereg.Refused, match="one source"):
        prereg._veto_or_refuse({"column": "chars_removed_gold", "above": "refused", "min_rows": 1})


def test_a_veto_read_on_fewer_rows_than_it_declared_does_not_fire():
    # the smoke is ten rows, and a close on it must not decide a veto declared on two hundred
    with pytest.raises(prereg.Refused, match="min_rows"):
        prereg._veto_or_refuse({"column": "chars_removed_gold", "above": "chars_removed_stranger"})
    veto = prereg._veto_or_refuse({"column": "chars_removed_gold", "above": "chars_removed_stranger",
                                   "margin": 0.05, "min_rows": 200})
    cuts_gold = _measured(["gold_section", "stranger"], kept=[0, 1],
                          chars={"before": [100, 100], "after": [50, 90]})
    got = prereg._veto(veto, {"run": {}, "measurement": {"arm": {1: cuts_gold}}}, {1})
    assert got["state"] == "undecided"


def test_a_chunk_stripped_to_nothing_leaves_kept_and_stays_in_the_characters():
    row = _measured(["gold_section", "gold_section", "stranger"], kept=[0],
                    chars={"before": [100, 100, 100], "after": [100, 0, 0]})
    row["kept_before_strip"] = [0, 1, 2]
    # the stranger was kept by the chunk pass and lost every strip: dropped, and cut whole
    assert columns.read("strangers_dropped", row) == 1.0
    assert columns.read("chars_removed_stranger", row) == 1.0
    assert columns.read("chars_removed_gold", row) == 0.5
    assert columns.read("gold_retained", row) == 1.0


def test_a_union_where_no_member_has_a_value_has_none():
    no_gold = _measured(["stranger"], kept=[0])
    assert columns.value(["gold_retained", "strangers_dropped"], no_gold) == 0.0
    assert columns.value(["gold_retained"], no_gold) is None
    assert columns.value(["gold_retained", "chars_removed_gold"], no_gold) is None


def test_a_fired_veto_stops_the_promise_before_its_arms_are_run():
    fired = {"column": "chars_removed_gold", "above": "chars_removed_stranger", "state": "fired"}
    cleared, why = prereg._verdict("not_run", [], [fired])
    assert cleared is False and "veto fired" in why
    quiet = fired | {"state": "quiet"}
    assert prereg._verdict("not_run", [], [quiet]) == (None, prereg._OPEN["not_run"])


def test_a_declared_floor_value_is_the_bar_when_no_floor_run_is_named(monkeypatch):
    promise = {"closing": {"columns": ["faithfulness"], "arm_should": "raise", "floor_value": 0.18},
               "population": {"sets": ["s"]}, "guards": [], "vetoes": []}
    control = {q: _row(faithfulness="6") for q in range(50)}
    arm = {q: _row(faithfulness="7" if q % 2 else "6") for q in range(50)}
    monkeypatch.setattr(prereg, "read", lambda name: promise)
    monkeypatch.setattr(prereg, "_question_ids", lambda sets: set(range(50)))
    monkeypatch.setattr(prereg, "_rows", {"c": control, "a": arm}.get)
    monkeypatch.setattr(prereg, "_closed_with", lambda name, runs, result: None)
    out = prereg.close("p", runs={"control": "c", "arm": "a"})
    assert out["bar"] == 0.18 and out["effect"]["ci95"][0] > 0.18 and out["cleared"] is True
    assert out["means"] == {"control": 6.0, "arm": 6.5}


def test_a_chunk_the_strip_never_asked_is_not_in_the_characters():
    row = _measured(["gold_section", "gold_section"], kept=[0, 1],
                    chars={"before": [100, 100], "after": [100, 50]})
    row["strips"] = [1, 3]
    # the one-strip gold chunk was decided by the chunk pass and would dilute the gold share toward quiet
    assert columns.read("chars_removed_gold", row) == 0.5


def test_a_declared_draw_narrows_the_population_to_its_questions(monkeypatch):
    promise = {"closing": {"columns": ["faithfulness"], "arm_should": "raise"},
               "population": {"sets": ["s"], "question_ids": [1, 2]}, "guards": [], "vetoes": []}
    control = {q: _row(faithfulness="6") for q in range(5)}
    arm = {q: _row(faithfulness="7") for q in range(5)}
    monkeypatch.setattr(prereg, "read", lambda name: promise)
    monkeypatch.setattr(prereg, "_question_ids", lambda sets: set(range(5)))
    monkeypatch.setattr(prereg, "_rows", {"c": control, "a": arm}.get)
    monkeypatch.setattr(prereg, "_closed_with", lambda name, runs, result: None)
    assert prereg.close("p", runs={"control": "c", "arm": "a"})["n"] == 2


def test_a_closing_recorded_under_the_nested_key_still_reads_as_what_it_named():
    named = {"runs": {"control": "c", "arm": "a"}, "measurements": {"arm": "m.json"}}
    # the first closing through the door wrote both maps under `runs`
    assert prereg._named({"runs": named, "cleared": False}) == named
    assert prereg._named(named | {"cleared": False}) == named
    assert prereg._named({"runs": {"control": "c", "arm": "a"}}) == {"runs": {"control": "c", "arm": "a"},
                                                                     "measurements": {}}
