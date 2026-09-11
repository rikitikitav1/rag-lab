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
    from job_handlers.judging import Snapshot, _apply_axis, _run_axis

    snapshot = Snapshot({}, {"generate_answer": 3}, {"generation": "gemma3:4b"})
    v, err = _run_axis(1, "relevance", lambda *a: _verdict())
    wrote = _apply_axis(_Log(), snapshot, "relevance", v, err)
    assert wrote is True
    assert snapshot.models["judging"] == "qwen2.5:7b"
    assert snapshot.prompts["judge_relevance"] == 2
    # what produced the answer is not overwritten by what scored it
    assert snapshot.models["generation"] == "gemma3:4b"
    assert snapshot.prompts["generate_answer"] == 3


def test_a_failed_axis_leaves_the_snapshot_alone():
    from job_handlers.judging import Snapshot, _apply_axis, _run_axis

    def boom(*a):
        raise RuntimeError("judge is down")

    snapshot = Snapshot({}, {}, {})
    v, err = _run_axis(1, "relevance", boom)
    wrote = _apply_axis(_Log(), snapshot, "relevance", v, err)
    assert wrote is False
    assert snapshot.models == {} and snapshot.prompts == {}
    assert snapshot.metrics["relevance"]["attempts"] == 1


def test_a_lost_card_on_an_axis_ends_the_pass_instead_of_failing_the_row():
    import pytest
    from engines.card import CardNotHanded
    from job_handlers.judging import _run_axis

    def lost(*a):
        raise CardNotHanded("vllm is down; the card stays where it is")

    with pytest.raises(CardNotHanded):
        _run_axis(1, "relevance", lost)


def test_rejudging_replaces_the_prompt_version_it_names():
    from job_handlers.judging import Snapshot, _apply_axis, _run_axis
    from models.registry import Purpose

    snapshot = Snapshot({}, {"judge_relevance": 2}, {})
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
    monkeypatch.setattr(judging, "require_role_ready", lambda role, **kw: None)
    monkeypatch.setattr(judging, "require_card", lambda role, model=None, asked_by=None: None)
    monkeypatch.setattr(judging.rejudge, "arm_bench", lambda arm: judging.judge.Bench())
    monkeypatch.setattr(judging.judge.Bench, "template", lambda self, purpose: ("t", 1))

    monkeypatch.setenv("JUDGE_WIDTH", "4")
    monkeypatch.setenv("OLLAMA_NUM_PARALLEL", "4")
    judging.judge_answers({})

    assert sorted(seen) == list(range(20)), "a fan-out must not drop or double a row"


def test_a_row_records_what_judged_it_beside_the_model(monkeypatch):
    # five passes over the same fifty rows agreed byte for byte with a seed pinned
    from job_handlers import judging

    monkeypatch.setattr(
        judging.llm, "sampler",
        lambda role, spec=None: judging.engines.Sampler({"temperature": 0, "seed": 0}, {}),
    )
    _judged_on(monkeypatch, "ollama")
    verdict = SimpleNamespace(reason="because", elapsed=1.5, model="qwen2.5:7b",
                              prompt_tokens=2317)

    monkeypatch.setenv("OLLAMA_NUM_PARALLEL", "4")
    written = judging._axis_metric(verdict, judging.stamp_of(4))
    # the time, because a pair of arms is comparable only inside one residency
    when = written.pop("judged_at")
    assert written.pop("residency_id") is None, "no pass named it, so the stamp says so"
    assert written == {
        "reason": "because", "elapsed": 1.5, "model": "qwen2.5:7b", "seed": 0, "width": 4,
        "slots_believed": 4, "on_card": None, "engine": "ollama:11434",
        # null with no instrument named would mean nowhere to ask, and ollama is where we ask
        "on_card_read_from": "ollama /api/ps",
        # ollama trims to its window silently, so the count is the only witness that it did not
        "judge_prompt_tokens": 2317,
        # the address and the entity together: one survives the migration, the other names it
        "engine_name": "ollama", "engine_refused": {}, "engine_added": {},
        # no pass named a residency, so no instrument minted one either
        "residency_source": None,
    }
    assert when.endswith("+00:00"), "the stamp must say when in utc, or two arms cannot be paired"
    named = judging.stamp_of(4, judging.Residency(7, True, "ollama /api/ps and the queue"))
    assert named["residency_source"] == "ollama /api/ps and the queue", "what minted it, verbatim"

    # a role with no seed says so rather than implying one: the early passes had none
    monkeypatch.setattr(
        judging.llm, "sampler",
        lambda role, spec=None: judging.engines.Sampler({"temperature": 0}, {}),
    )
    bare = judging.stamp_of(1)
    bare.pop("judged_at")
    bare.pop("residency_id")
    assert bare == {
        "seed": None, "width": 1, "slots_believed": 4, "on_card": None,
        "engine": "ollama:11434", "engine_name": "ollama", "engine_refused": {},
        "engine_added": {}, "on_card_read_from": "ollama /api/ps", "residency_source": None,
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
    monkeypatch.setattr(
        judging, "stamp_of",
        lambda width, residency=None, model=None: {"seed": 0, "width": width},
    )

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
    # abstain, rather than bend the judge prompt towards the guest
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

    # every row that carries a residency carries an engine too: they were stamped together
    def row(rid, version=2, qid=1):
        stamp = {"residency_id": rid, "engine": "ollama:11434", "engine_name": "ollama"}
        return SimpleNamespace(
            metrics={"faithfulness": stamp} if rid else {},
            prompts={"judge_faithfulness": version}, question_id=qid, run_name="arm",
            sources=[{"source": "a.md", "hop": None, "vector_rank": 1, "keyword_rank": None}],
        )

    same = residencies({"a": [row(7), row(7, qid=2)], "b": [row(7)]})
    assert same["one_residency"] is True
    assert "necessary, not sufficient" in same["read_this_first"]

    split = residencies({"a": [row(7)], "b": [row(9)]})
    assert split["one_residency"] is False and "not comparable" in split["read_this_first"]

    old = residencies({"a": [row(None)], "b": [row(None)]})
    assert old["one_residency"] is None and "nothing can be said" in old["read_this_first"]

    # one arm silent and the other not: the union has one id, and that used to read as agreement
    half = residencies({"a": [row(None)], "b": [row(7)]})
    assert half["one_residency"] is None, "silence cannot agree with an arm that recorded"
    assert "nothing can be said" in half["read_this_first"], "an unknown residency comes first"

    # residency agrees and one arm never recorded an engine: that outranks any residency reading
    quiet = SimpleNamespace(metrics={"faithfulness": {"residency_id": 7}},
                            prompts={"judge_faithfulness": 2}, question_id=1, run_name="arm",
                            sources=row(7).sources)
    mixed = residencies({"a": [quiet], "b": [row(7)]})
    assert mixed["one_residency"] is True and mixed["one_engine"] is None
    assert "recorded no engine" in mixed["read_this_first"]

    # two backends are not one instrument at all, and that outranks any residency reading
    def on(rid, engine):
        return SimpleNamespace(metrics={"faithfulness": {"residency_id": rid, "engine": engine}},
                               prompts={"judge_faithfulness": 2}, question_id=1, run_name="arm",
                               sources=row(rid).sources)

    split = residencies({"a": [on(7, "ollama:11434")], "b": [on(7, "api.example:443")]})
    assert split["one_engine"] is False
    assert "different engines" in split["read_this_first"], "the engine speaks before residency"
    assert split["one_residency"] is True, "the ids match, and that is exactly why it misleads"

    # one residency, one engine, two rulers: the prompt version is the difference nobody read
    rulers = residencies({"a": [row(7, version=2)], "b": [row(7, version=5)]})
    assert rulers["one_judge_prompt"] is False and rulers["one_residency"] is True
    assert "two rulers" in rulers["read_this_first"], "the ruler speaks before the reload"
    assert rulers["judge_prompts_by_run"] == {
        "a": {"faithfulness": [2]}, "b": {"faithfulness": [5]},
    }

    # three axes carry three versions, and their union is three even when both arms agree
    def all_axes(version):
        return SimpleNamespace(
            metrics={"faithfulness": {"residency_id": 7, "engine": "ollama:11434"}},
            prompts={f"judge_{axis}": version
                     for axis in ("faithfulness", "relevance", "completeness")},
            question_id=1, run_name="arm", sources=row(7).sources,
        )

    wide = residencies({"a": [all_axes(2)], "b": [all_axes(2)]})
    assert wide["one_judge_prompt"] is True, "one version per axis is one ruler, not three"

    # a version nobody recorded cannot be compared, and that is not the same as a match
    blank = SimpleNamespace(
        metrics={"faithfulness": {"residency_id": 7, "engine": "ollama:11434"}}, prompts={},
        question_id=1, run_name="arm", sources=row(7).sources,
    )
    unknown = residencies({"a": [blank], "b": [row(7)]})
    assert unknown["one_judge_prompt"] is None
    assert "cannot be said" in unknown["read_this_first"]


def test_a_pass_names_the_residency_it_caused_or_inherits_the_last(monkeypatch):
    # `/api/ps` has no load moment, so the pass that found the card empty is the one that names it
    import job_handlers.judging as j

    _judged_on(monkeypatch, "ollama")
    on_card, last, disturbed, handed = [None], [None], [False], [False]
    asked = []
    monkeypatch.setattr(j, "judge_on_card", lambda _model=None: on_card[0])
    monkeypatch.setattr(
        j, "_last_residency", lambda name, started=None: asked.append(name) or last[0]
    )
    monkeypatch.setattr(j, "_loaded_since", lambda prev, job_id: disturbed[0])
    monkeypatch.setattr(j, "_card_changed_hands", lambda since, name: handed[0])
    queue = "ollama /api/ps and the queue"

    assert j._residency(42) == j.Residency(42, None, "ollama /api/ps"), "the card was empty"
    on_card[0], last[0] = True, (7, "2026-09-10T00:00:00+00:00")
    assert j._residency(42) == j.Residency(7, True, queue), "it was resident, the older one holds"
    assert asked[-1] == "ollama", "only this engine's rows may lend it a number"
    last[0] = None
    assert j._residency(42) == j.Residency(42, True, queue)

    # half on the card is another instrument: the cpu layers answer with other kernels
    on_card[0], last[0] = False, (7, "2026-09-10T00:00:00+00:00")
    assert j._residency(42) == j.Residency(42, False, "ollama /api/ps"), "partial never inherits"

    # `/api/ps` cannot see a neighbour loaded between two passes, but the queue can
    on_card[0], disturbed[0] = True, True
    assert j._residency(42) == j.Residency(42, True, queue), "something loaded since, this is new"

    # a shell handed the card to vLLM and back, which the queue never saw but the verdicts did
    disturbed[0], handed[0] = False, True
    assert j._residency(42) == j.Residency(42, True, queue), "the card changed hands in between"


def test_a_vllm_pass_is_named_by_the_process_that_holds_the_judge(monkeypatch):
    # `created` in `/v1/models` is the reply's clock, so the process start is what opens a residency
    import engines
    import job_handlers.judging as j
    from models.registry import EngineKind, Placement

    spec = engines.EngineSpec(3, "vllm", EngineKind.vllm, "VLLM", Placement.gpu)
    monkeypatch.setattr(j.llm, "resolve_for", lambda role, model=None: engines.Resolved("q", spec))
    monkeypatch.setattr(j, "judge_on_card", lambda _model=None: pytest.fail("vLLM has no /api/ps"))
    monkeypatch.setattr(j, "_loaded_since", lambda *a: pytest.fail("one process, whatever queued"))
    started, last, asked = ["2026-09-10T15:26:53.970000+00:00"], [None], []
    monkeypatch.setattr(j.engines, "started_at", lambda _spec: started[0])
    monkeypatch.setattr(j, "_last_residency",
                        lambda name, at=None: asked.append((name, at)) or last[0])
    source = "vllm /metrics process start"

    assert j._residency(42) == j.Residency(42, None, source), "the first pass on a process names it"
    assert asked[-1] == ("vllm", started[0]), "only rows judged under this very start may lend"
    last[0] = (30, started[0])
    assert j._residency(42) == j.Residency(30, None, source), "same process, so the older one holds"

    # a server that cannot say when it started names nothing, and the pass stands alone
    started[0] = None
    assert j._residency(42) == j.Residency(42, None, "the pass, vllm /metrics unreachable")


def test_the_process_start_is_read_from_metrics_and_not_from_created(monkeypatch):
    import engines
    from engines import vllm as core
    from models.registry import EngineKind, Placement

    spec = engines.EngineSpec(3, "vllm", EngineKind.vllm, "VLLM", Placement.gpu)
    monkeypatch.setenv("VLLM_BASE_URL", "http://vllm:8000")
    body = ["# HELP process_start_time_seconds Start time\n"
            "process_start_time_seconds 1.78905401397e+09\n"]
    seen = []

    class _R:
        def __init__(self, url):
            seen.append(url)
            self.text = body[0]

    import requests

    monkeypatch.setattr(requests, "get", lambda url, headers=None, timeout=None: _R(url))
    assert core.started_at(spec) == "2026-09-10T15:26:53.970000+00:00"
    assert seen == ["http://vllm:8000/metrics"]

    body[0] = "vllm:num_requests_running 0\n"
    assert core.started_at(spec) is None, "a server that does not say must not be given a start"


def test_the_judge_settles_the_outcome_it_alone_can_know():
    from types import SimpleNamespace

    from job_handlers import judging

    def row(outcome, faithfulness):
        snap = judging.Snapshot({"outcome": outcome} if outcome else {}, {}, {})
        judging._settle_outcome(SimpleNamespace(faithfulness=faithfulness), snap)
        return snap.metrics

    zero = row("answered", "0")
    assert zero["settled_outcome"] == "answered_ungrounded", "zero means it stood on nothing"
    assert zero["outcome"] == "answered", "what the answer knew never changes, or a replay breaks"

    # everything else keeps deriving: the derivation is still wider than what the writer saw
    assert "settled_outcome" not in row("answered", "7")
    assert "settled_outcome" not in row("refused", "0"), "a refusal is not ours to freeze"
    assert "settled_outcome" not in row("error", "0"), "an error may still re-derive as exhausted"
    assert row(None, "7") == {}
    assert "settled_outcome" not in row("answered", None)


def test_a_job_type_nobody_classified_is_assumed_to_evict_the_judge():
    # the safe way round: a new type is a stranger, and a stranger is assumed to take the card
    import job_specs

    assert not job_specs.disturbs_the_judge("judge_answers")
    assert job_specs.disturbs_the_judge("judge_guest_axes"), "relevancy loads the embedder too"
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


def test_a_rejudge_that_raises_the_score_takes_the_settlement_back():
    # the key was only ever written, so one zero froze the outcome against every later pass
    from types import SimpleNamespace

    from job_handlers import judging

    metrics = {"outcome": "answered", "settled_outcome": "answered_ungrounded"}
    snap = judging.Snapshot(metrics, {}, {})
    judging._settle_outcome(SimpleNamespace(faithfulness="8"), snap)
    assert "settled_outcome" not in snap.metrics, "the judge said eight, and the freeze must go"

    judging._settle_outcome(SimpleNamespace(faithfulness="0"), snap)
    assert snap.metrics["settled_outcome"] == "answered_ungrounded"


def test_the_language_probe_records_what_judged_it(monkeypatch):
    # the control read out of regime twice, and nothing in the file said whether the judge was whole
    from evals import judge_language
    from job_handlers import judging

    seen = {}
    monkeypatch.setattr(judging, "_residency", lambda job_id: judging.Residency(77, True))
    _judged_on(monkeypatch, "ollama")
    monkeypatch.setattr(
        judging.llm, "sampler",
        lambda role, spec=None: judging.engines.Sampler({"seed": 0}, {}),
    )
    monkeypatch.setattr(judging, "_parallel_slots", lambda: 1)
    monkeypatch.setattr(judging, "require_role_ready", lambda role, **kw: None)
    monkeypatch.setattr(judging, "require_card", lambda role, model=None, asked_by=None: None)
    monkeypatch.setattr(judge_language, "measure", lambda *a, **kw: seen.update(kw) or {})
    monkeypatch.setattr(judging.measurements, "record", lambda *a, **kw: "nowhere")
    judging.judge_language({"run_name": "r", "_job_id": 1})

    stamp = seen.get("stamp") or {}
    # the record, not the source text: the old assertion passed on an empty stamp
    assert stamp.get("residency_id") == 77, "the pass names the residency it ran in"
    assert stamp.get("on_card") is True and stamp.get("engine_name") == "ollama"


def _judged_on(monkeypatch, name):
    import engines
    from job_handlers import judging
    from models.registry import EngineKind, Placement

    spec = engines.EngineSpec(1, name, EngineKind.ollama, name.upper(), Placement.gpu)
    monkeypatch.setattr(
        judging.llm, "resolve_for", lambda role, model=None: engines.Resolved("qwen2.5:7b", spec)
    )
    monkeypatch.setattr(judging.engines, "address_of", lambda _spec: "ollama:11434")


def test_the_card_is_read_after_the_judge_answered_and_not_before(monkeypatch):
    # the generator evicts the judge, so a probe taken first sees an empty card and splits the pass
    from types import SimpleNamespace

    from job_handlers import judging

    order = []
    monkeypatch.setattr(judging, "_residency", lambda job_id, model=None: order.append("probe")
                        or judging.Residency(job_id, True))

    late = judging.LateResidency(7)
    assert order == [], "building the holder must not touch the card"

    class _Session:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def get(self, _model, _id):
            return SimpleNamespace(
                answered=True, metrics={}, relevance=None, faithfulness=None,
                completeness=None, context="ctx",
                question=SimpleNamespace(original_text="q", reference_answer="ref"),
                answer="a",
            )

    monkeypatch.setattr(judging, "Session", _Session)
    monkeypatch.setattr(judging, "_run_axis",
                        lambda *a, **kw: order.append("call") or SimpleNamespace(score=1))
    monkeypatch.setattr(judging, "stamp_of", lambda *a, **kw: {})
    monkeypatch.setattr(judging, "_merge_our_scores", lambda *a, **kw: True)

    judging._judge_log(1, residency=late)

    assert order and order[-1] == "probe", f"the probe must come last, got {order}"
    assert order.count("probe") == 1, "one probe per pass, not per row"


def test_a_vllm_judge_is_read_by_its_own_door_and_never_by_ollama_s(monkeypatch):
    # the one instrument: the judge on vLLM wrote null while the stand said true
    import engines
    from job_handlers import judging
    from models.registry import EngineKind, Placement

    spec = engines.EngineSpec(3, "vllm", EngineKind.vllm, "VLLM", Placement.gpu)
    asked = []
    monkeypatch.setattr(judging.card.ollama, "residency", lambda s: asked.append(s) or [])
    monkeypatch.setattr(judging.card.vllm, "card_state", lambda s: "awake")
    monkeypatch.setattr(judging.card.vllm, "served", lambda s: ["Qwen/Qwen2.5-7B-Instruct-AWQ"])
    monkeypatch.setattr(
        judging.llm, "resolve", lambda role: engines.Resolved("Qwen/Qwen2.5-7B-Instruct-AWQ", spec)
    )

    assert judging.residency_instrument(spec) == "vllm /is_sleeping"
    assert judging.judge_on_card() is True
    assert asked == [], "`/api/ps` is ollama's door and 404s on a vLLM"


def test_the_next_sweep_does_not_inherit_the_retry_counter(monkeypatch):
    # a pass that was ever deferred carries `attempts`, and the door refuses it from a caller
    import job_specs
    from job_handlers import judging

    seen = {}
    monkeypatch.setattr(judging.job_queue, "enqueue",
                        lambda t, o, **kw: seen.update(type=t, options=o) or 1)
    monkeypatch.setattr(judging, "Session", _session_yielding(None))
    monkeypatch.setattr(judging, "_target_log_ids", lambda _s, _o: [1, 2])

    judging._sweep_again_if_rows_are_still_owed(
        {"run_name": "r", "attempts": 2, "deferred_seconds": 90, "_job_id": 7}, "r"
    )

    assert seen["options"] == {"run_name": "r", "sweep": 1}
    # the real refusal, run against what the real function built
    job_specs.check("judge_answers", seen["options"])


def _session_yielding(value):
    class _S:
        def __enter__(self):
            return value

        def __exit__(self, *_):
            return False

    return _S


def test_the_card_is_read_for_the_engine_that_judges_not_the_one_the_role_names(monkeypatch):
    # a bench override put `on_card` from ollama's /api/ps next to a vllm engine name
    import engines
    from job_handlers import judging
    from models.registry import EngineKind, Placement

    asked = []
    monkeypatch.setattr(
        judging.llm, "resolve_for",
        lambda role, model=None: asked.append(model) or engines.Resolved(
            model or "qwen2.5:7b",
            engines.EngineSpec(3, "vllm", EngineKind.vllm, "VLLM", Placement.gpu)
            if model else engines.EngineSpec(1, "ollama", EngineKind.ollama, "OLLAMA",
                                             Placement.gpu),
        ),
    )
    monkeypatch.setattr(judging.card.vllm, "card_state", lambda _spec: "asleep")

    assert judging.judge_on_card("Qwen/Qwen2.5-7B-Instruct-AWQ") is False
    assert asked == ["Qwen/Qwen2.5-7B-Instruct-AWQ"], "the override must reach the card probe"
