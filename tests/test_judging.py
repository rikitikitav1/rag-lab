from types import SimpleNamespace

import pytest
from conftest import FakeSession
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
    from job_handlers.judging import _apply_axis, _run_axis, _Snapshot

    snapshot = _Snapshot({}, {"generate_answer": 3}, {"generation": "gemma3:4b"})
    v, err = _run_axis(1, "relevance", lambda *a: _verdict())
    wrote = _apply_axis(_Log(), snapshot, "relevance", v, err)
    assert wrote is True
    assert snapshot.models["judging"] == "qwen2.5:7b"
    assert snapshot.prompts["judge_relevance"] == 2
    # what produced the answer is not overwritten by what scored it
    assert snapshot.models["generation"] == "gemma3:4b"
    assert snapshot.prompts["generate_answer"] == 3


def test_a_failed_axis_leaves_the_snapshot_alone():
    from job_handlers.judging import _apply_axis, _run_axis, _Snapshot

    def boom(*a):
        raise RuntimeError("judge is down")

    snapshot = _Snapshot({}, {}, {})
    v, err = _run_axis(1, "relevance", boom)
    wrote = _apply_axis(_Log(), snapshot, "relevance", v, err)
    assert wrote is False
    assert snapshot.models == {} and snapshot.prompts == {}
    assert snapshot.metrics["relevance"]["attempts"] == 1


def test_rejudging_replaces_the_prompt_version_it_names():
    from job_handlers.judging import _apply_axis, _run_axis, _Snapshot
    from models.registry import Purpose

    snapshot = _Snapshot({}, {"judge_relevance": 2}, {})
    v, err = _run_axis(
        1, "relevance", lambda *a: _verdict(purpose=Purpose.judge_relevance, prompt_version=3)
    )
    _apply_axis(_Log(), snapshot, "relevance", v, err)
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


def test_the_judge_runs_one_row_at_a_time_unless_the_width_says_otherwise(monkeypatch):
    # every number read before this came from a width of one, and slots the server lacks queue
    from job_handlers import judging

    monkeypatch.delenv("JUDGE_WIDTH", raising=False)
    monkeypatch.delenv("OLLAMA_NUM_PARALLEL", raising=False)
    assert judging.judge_width() == 1

    monkeypatch.setenv("JUDGE_WIDTH", "4")
    monkeypatch.setenv("OLLAMA_NUM_PARALLEL", "4")
    assert judging.judge_width() == 4

    # four rows against one slot queue inside the server while the stamp still reads four
    monkeypatch.setenv("OLLAMA_NUM_PARALLEL", "1")
    assert judging.judge_width() == 1

    monkeypatch.setenv("JUDGE_WIDTH", "0")
    assert judging.judge_width() == 1, "a width of nothing judges nothing"


def test_every_row_is_judged_once_whatever_the_width(monkeypatch):
    from job_handlers import judging

    seen = []
    monkeypatch.setattr(judging, "_target_log_ids", lambda session, options: list(range(20)))
    monkeypatch.setattr(
        judging, "_judge_log", lambda log_id, **kw: seen.append(log_id)
    )
    monkeypatch.setattr(judging, "Session", FakeSession)
    monkeypatch.setattr(judging.experiment, "try_aggregate_for_run", lambda run: None)
    monkeypatch.setattr(judging, "require_role_ready", lambda role: None)
    monkeypatch.setattr(judging.rejudge, "arm_bench", lambda arm: judging.judge.Bench())
    monkeypatch.setattr(judging.judge.Bench, "template", lambda self, purpose: ("t", 1))

    monkeypatch.setenv("JUDGE_WIDTH", "4")
    monkeypatch.setenv("OLLAMA_NUM_PARALLEL", "4")
    judging.judge_answers({})

    assert sorted(seen) == list(range(20)), "a fan-out must not drop or double a row"


def test_a_row_records_what_judged_it_beside_the_model(monkeypatch):
    # five passes over the same fifty rows agreed byte for byte with a seed pinned
    from job_handlers import judging

    monkeypatch.setattr(judging.llm, "sampler_of", lambda role: {"temperature": 0, "seed": 0})
    monkeypatch.setattr(judging.llm, "engine", lambda: "ollama:11434")
    verdict = SimpleNamespace(reason="because", elapsed=1.5, model="qwen2.5:7b")

    monkeypatch.setenv("OLLAMA_NUM_PARALLEL", "4")
    written = judging._axis_metric(verdict, judging._stamp(4))
    # the time, because a pair of arms is comparable only inside one residency
    when = written.pop("judged_at")
    assert written.pop("residency_id") is None, "no pass named it, so the stamp says so"
    assert written == {
        "reason": "because", "elapsed": 1.5, "model": "qwen2.5:7b", "seed": 0, "width": 4,
        "slots_believed": 4, "on_card": None, "engine": "ollama:11434",
    }
    assert when.endswith("+00:00"), "the stamp must say when in utc, or two arms cannot be paired"

    # a role with no seed says so rather than implying one: the passes before 31.08 had none
    monkeypatch.setattr(judging.llm, "sampler_of", lambda role: {"temperature": 0})
    bare = judging._stamp(1)
    bare.pop("judged_at")
    bare.pop("residency_id")
    assert bare == {
        "seed": None, "width": 1, "slots_believed": 4, "on_card": None,
        "engine": "ollama:11434",
    }


def test_a_row_the_judge_failed_on_is_swept_again_instead_of_stranding_the_series(monkeypatch):
    # one row of 823 stayed owed, nothing came back for it, and the experiment sat in running
    from job_handlers import judging

    enqueued = []
    monkeypatch.setattr(judging.job_queue, "enqueue", lambda t, o: enqueued.append((t, o)))
    monkeypatch.setattr(judging, "Session", FakeSession)

    monkeypatch.setattr(judging, "_target_log_ids", lambda session, options: [7])
    judging._sweep_again_if_rows_are_still_owed({"run_name": "arm"}, "arm")
    assert enqueued == [("judge_answers", {"run_name": "arm", "sweep": 1})]

    # and it stops: a row that spent its attempts is no longer owed, so the next sweep finds none
    enqueued.clear()
    monkeypatch.setattr(judging, "_target_log_ids", lambda session, options: [])
    judging._sweep_again_if_rows_are_still_owed({"run_name": "arm", "sweep": 1}, "arm")
    assert enqueued == []

    # a hand-dispatched subset never loops: it names its rows and is done when they are
    monkeypatch.setattr(judging, "_target_log_ids", lambda session, options: [7])
    judging._sweep_again_if_rows_are_still_owed({"run_name": "arm", "log_ids": [7]}, "arm")
    assert enqueued == []


def test_the_judge_reads_a_verdict_out_of_whatever_the_server_wrapped_it_in():
    # one row lost its relevance to a JSONDecodeError three times and the record kept the name
    from use_cases.judge import _verdict_of

    plain = _verdict_of('{"reason": "grounded", "score": 8}')
    assert (plain.score, plain.reason) == (8, "grounded")

    fenced = _verdict_of('```json\n{"reason": "grounded", "score": 8}\n```')
    assert fenced.score == 8, "a fence around the object is not a reason to lose the row"

    with_prose = _verdict_of('Here is my verdict:\n{"reason": "no", "score": 0}\nHope it helps')
    assert with_prose.score == 0


def test_a_verdict_that_is_not_one_says_what_the_judge_answered():
    from use_cases.judge import _verdict_of

    with pytest.raises(ValueError, match="no JSON object"):
        _verdict_of("I cannot score this")

    with pytest.raises(ValueError, match="score"):
        _verdict_of('{"reason": "fine", "score": 42}')

    with pytest.raises(ValueError, match="reason"):
        _verdict_of('{"score": 5}')


def test_a_row_outside_the_control_sample_is_not_owed_that_axis():
    # nobody is coming for the rows a control sample left out, and the series waits
    from job_handlers.judging import still_to_judge

    paths = set(still_to_judge().compile().params.values())

    for axis in ("relevance", "faithfulness", "completeness"):
        assert (axis, "skipped") in paths, f"{axis} would wait for a row nobody is judging"
        assert (axis, "attempts") in paths, "and the attempts cap still ends a failing row"


def test_the_sweep_ends_even_when_the_rows_stay_owed_for_ever(monkeypatch):
    # the test above proves it stops by handing the second sweep an empty list
    from job_handlers import judging

    enqueued, failed = [], []
    monkeypatch.setattr(judging.job_queue, "enqueue", lambda t, o: enqueued.append(o))
    monkeypatch.setattr(judging, "Session", FakeSession)
    monkeypatch.setattr(judging, "_target_log_ids", lambda session, options: [7])
    monkeypatch.setattr(judging.experiment, "mark_failed_for_run", failed.append)

    options = {"run_name": "arm"}
    for _ in range(10):
        judging._sweep_again_if_rows_are_still_owed(options, "arm")
        if not enqueued:
            break
        options = enqueued.pop()

    assert options["sweep"] == judging._MAX_SWEEPS, "the sweeps stop at their own cap"
    # and the series is not left waiting for a verdict nobody will produce
    assert failed == ["arm"]


def test_a_row_that_broke_before_any_axis_ran_stops_being_owed(monkeypatch):
    # `attempts` only grew inside the axis writer, so a failure elsewhere left it untouched
    from types import SimpleNamespace

    from job_handlers import judging

    def row(**over):
        base = dict(
            metrics={"relevance": {"attempts": 1}}, relevance=None, faithfulness="7",
            completeness=None, context="ctx",
            question=SimpleNamespace(reference_answer="ref"),
        )
        return SimpleNamespace(**{**base, **over})

    def count(ql, skip):
        class _Session:
            def __enter__(self):
                return self

            def __exit__(self, *_):
                return False

            def execute(self, statement, params=None):
                return None

            def get(self, model, ident, **kw):
                return ql

            def commit(self):
                pass

        monkeypatch.setattr(judging, "Session", _Session)
        judging._count_the_attempt(9, skip, "connection reset")
        return ql.metrics

    metrics = count(row(), ("completeness",))
    assert metrics["relevance"]["attempts"] == 2, "the axis it owed counts the try"
    assert "faithfulness" not in metrics, "an axis already scored is not owed"
    assert metrics["completeness"] == {"skipped": "outside the control sample"}, (
        "a skipped axis must be marked here too, or the row stays owed and the sweep loops"
    )

    # nothing to judge it against: never owed, so never charged an attempt
    bare = count(row(context=None, question=SimpleNamespace(reference_answer=None)), ())
    assert "faithfulness" not in bare and "completeness" not in bare
def test_the_control_sample_is_drawn_by_question_not_sliced_off_a_scan(monkeypatch):
    # `log_ids[:sample]` off a query with no ORDER BY, and each arm is its own scan
    from job_handlers import judging

    issued = []

    class _Session:
        def scalars(self, stmt):
            issued.append(str(stmt.compile(compile_kwargs={"literal_binds": True})))
            return iter([1, 2])

    judging._control_sample(_Session(), "arm_a", [9, 8, 7], 2, seed=0)
    sql = issued[-1]

    assert "ORDER BY md5" in sql, "the sample is drawn, not taken in scan order"
    assert "question_logs.question_id" in sql.split("ORDER BY")[1], (
        "the draw is over the question, which the copies share, not over the row id"
    )
    # over the whole run: a later sweep sees fewer owed rows and would draw a different sample
    assert "run_name = 'arm_a'" in sql
    assert "9, 8, 7" not in sql
    # `concat` ignores NULL, so every row without a question would collapse to one hash
    assert "question_id IS NOT NULL" in sql


def test_a_skipped_axis_does_not_erase_a_verdict_the_row_already_carries(monkeypatch):
    # the marker replaced the whole per-axis dict, losing the reason of a judged control
    from types import SimpleNamespace

    from job_handlers import judging

    ql = SimpleNamespace(
        id=5, answered=True, answer="a", relevance="8", faithfulness=None, completeness=None,
        context="ctx", metrics={"relevance": {"reason": "grounded", "model": "qwen2.5:7b"}},
        prompts={}, models={}, question=SimpleNamespace(original_text="q", reference_answer=None),
    )

    class _Session:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def execute(self, statement, params=None):
            return None

        def get(self, model, ident, **kw):
            return ql

        def commit(self):
            pass

    monkeypatch.setattr(judging, "Session", _Session)
    monkeypatch.setattr(judging, "_apply_axis", lambda *a, **kw: False)
    monkeypatch.setattr(judging, "_stamp", lambda width, residency=None: {"seed": 0, "width": width})

    judging._judge_log(5, skip=("relevance", "completeness"))

    assert ql.metrics["relevance"] == {"reason": "grounded", "model": "qwen2.5:7b"}
    assert ql.metrics["completeness"] == {"skipped": "outside the control sample"}


def test_the_verdict_is_the_last_object_that_validates():
    # a nested object always sorts last, and `SCORE_SCHEMA` allows one
    from use_cases.judge import _verdict_of

    shown_then_answered = (
        'Format:\n```json\n{"reason": "example", "score": 0}\n```\n'
        'My verdict:\n{"reason": "grounded", "score": 8}'
    )
    assert _verdict_of(shown_then_answered).score == 8
    assert _verdict_of('{"reason": "g", "score": 8, "details": {"chunks": 3}}').score == 8
    assert _verdict_of('{"reason": "see {inner: 1}", "score": 7}').score == 7


def test_a_failure_written_to_the_row_is_bounded_and_keeps_what_was_there():
    # one writer bounded the text and the other handed the raw `str(e)` of a SQLAlchemy error
    from job_handlers import judging

    was = {"relevance": {"judge": "qwen2.5:7b", "attempts": 1}}
    written = judging._errored_metric(was, "relevance", "x" * 900)

    assert len(written["error"]) == judging._ERROR_CHARS
    assert written["attempts"] == 2
    assert written["judge"] == "qwen2.5:7b", "an earlier marker survives"

    # a row being judged again is not skipped, and saying both reads as neither
    forced = judging._errored_metric(
        {"relevance": {"skipped": "outside the control sample"}}, "relevance", "boom"
    )

    assert "skipped" not in forced
    assert forced["attempts"] == 1


def test_a_sweep_does_not_carry_the_finished_job_id_into_the_next_one(monkeypatch):
    from job_handlers import judging

    enqueued = []
    monkeypatch.setattr(judging.job_queue, "enqueue", lambda t, o: enqueued.append(o))
    monkeypatch.setattr(judging, "Session", FakeSession)
    monkeypatch.setattr(judging, "_target_log_ids", lambda session, options: [7])

    judging._sweep_again_if_rows_are_still_owed({"run_name": "arm", "_job_id": 41}, "arm")

    assert enqueued == [{"run_name": "arm", "sweep": 1}]


def test_our_judge_scores_outside_the_lock_and_writes_under_it():
    # three model calls under `FOR UPDATE` timed out whoever lost the race, closing an axis on it
    import inspect

    from job_handlers import judging

    scoring = inspect.getsource(judging._judge_log)
    assert "with_for_update" not in scoring, "the lock is back around the model calls"
    assert "_run_axis" in scoring and "_merge_our_scores" in scoring

    merging = inspect.getsource(judging._merge_our_scores)
    assert "with_for_update" in merging and "_run_axis" not in merging
    # a verdict taken meanwhile wins: the merge asks again what is owed
    assert "_owed" in merging


def test_a_refusal_owes_no_axis_and_both_halves_of_the_rule_say_so():
    # the owner's rule of 06.09: abstain, rather than bend the judge prompt towards the guest
    from types import SimpleNamespace

    from job_handlers.judging import _owed, still_to_judge

    refused = SimpleNamespace(
        metrics={"refusal": True}, relevance=None, faithfulness=None, completeness=None,
        context="ctx", question=SimpleNamespace(reference_answer="ref"),
    )
    assert _owed(refused) == ()

    answered = SimpleNamespace(**{**vars(refused), "metrics": {"refusal": False}})
    assert set(_owed(answered)) == {"relevance", "faithfulness", "completeness"}

    # a row with no such fact keeps the old behaviour rather than being silenced on a guess
    silent = SimpleNamespace(**{**vars(refused), "metrics": {}})
    assert _owed(silent) != ()

    assert "true" in str(still_to_judge().compile().params.values()), "the sql half is missing it"


def test_a_comparison_says_when_two_arms_were_judged_across_a_reload():
    # the rule lived in a log, and two people took the wrong pair on the same day because of it
    from types import SimpleNamespace

    from evals.compare import residencies

    def row(rid):
        return SimpleNamespace(metrics={"faithfulness": {"residency_id": rid}} if rid else {})

    same = residencies({"a": [row(7), row(7)], "b": [row(7)]})
    assert same["one_residency"] is True
    assert "necessary, not sufficient" in same["read_this_first"]

    split = residencies({"a": [row(7)], "b": [row(9)]})
    assert split["one_residency"] is False and "not comparable" in split["read_this_first"]

    old = residencies({"a": [row(None)], "b": [row(None)]})
    assert old["one_residency"] is None and "nothing can be said" in old["read_this_first"]

    # two backends are not one instrument at all, and that outranks any residency reading
    def on(rid, engine):
        return SimpleNamespace(metrics={"faithfulness": {"residency_id": rid, "engine": engine}})

    split = residencies({"a": [on(7, "ollama:11434")], "b": [on(7, "api.example:443")]})
    assert split["one_engine"] is False
    assert "different engines" in split["read_this_first"], "the engine speaks before residency"
    assert split["one_residency"] is True, "the ids match, and that is exactly why it misleads"


def test_a_pass_names_the_residency_it_caused_or_inherits_the_last(monkeypatch):
    # `/api/ps` has no load moment, so the pass that found the card empty is the one that names it
    import job_handlers.judging as j

    on_card, last, disturbed = [None], [None], [False]
    monkeypatch.setattr(j, "judge_on_card", lambda: on_card[0])
    monkeypatch.setattr(j, "_last_residency", lambda: last[0])
    monkeypatch.setattr(j, "_loaded_since", lambda prev, job_id: disturbed[0])

    assert j._residency(42) == j.Residency(42, None), "the card was empty, so this pass loads it"
    on_card[0], last[0] = True, 7
    assert j._residency(42) == j.Residency(7, True), "it was resident, so the older one holds"
    last[0] = None
    assert j._residency(42) == j.Residency(42, True)

    # half on the card is another instrument: the cpu layers answer with other kernels
    on_card[0], last[0] = False, 7
    assert j._residency(42) == j.Residency(42, False), "a partial load never inherits"

    # `/api/ps` cannot see a neighbour loaded between two passes, but the queue can
    on_card[0], disturbed[0] = True, True
    assert j._residency(42) == j.Residency(42, True), "something loaded since, so this is new"


def test_a_job_type_nobody_classified_is_assumed_to_evict_the_judge():
    # the safe way round: a new type is a stranger, and a stranger is assumed to take the card
    import job_specs

    assert not job_specs.disturbs_the_judge("judge_answers")
    assert not job_specs.disturbs_the_judge("judge_guest_axes"), "guests ride the judge's model"
    assert job_specs.disturbs_the_judge("eval_run")
    assert job_specs.disturbs_the_judge("judge_language"), "the probe answers on the generator"
    assert job_specs.disturbs_the_judge("a_type_invented_next_year")


def test_a_pass_walks_the_rows_in_the_order_it_was_given():
    # the order of requests is what the drift measurement moves, so postgres may not choose it
    from job_handlers import judging

    class _Session:
        def scalars(self, stmt):
            return [3, 1, 2]

    got = judging._target_log_ids(_Session(), {"log_ids": [2, 3, 1, 99]})
    assert got == [2, 3, 1], "the caller's order, and nothing it did not ask for"
