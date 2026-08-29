import pytest
from use_cases import retrieval_compare as rc


def _row(qid, file_rank=None, section_rank=None, scorable=True):
    return {
        "id": qid,
        "file_rank": file_rank,
        "section_rank": section_rank,
        "section_scorable": scorable,
    }


def test_the_grid_is_the_product_of_the_axes():
    grid = rc.arms({"variant": ["a", "b"], "rerank_top": [0, 20]})
    assert len(grid) == 4
    assert {"variant": "a", "rerank_top": 0} in grid
    assert {"variant": "b", "rerank_top": 20} in grid


def test_an_arm_is_named_from_its_own_values():
    # position in the grid is not a name: a reordered axis would rename every arm
    assert rc.arm_name({"variant": "clean_1024", "rerank_top": 20}) == "rerank_top=20_variant=clean_1024"


def test_a_delta_is_paired_on_the_questions_both_arms_measured():
    before = [_row(1, section_rank=2), _row(2, section_rank=None), _row(3, section_rank=1)]
    after = [_row(1, section_rank=1), _row(2, section_rank=4), _row(9, section_rank=1)]
    out = rc.paired_delta(before, after, "section")
    assert out["questions"] == 2, "question 9 is in one arm only and cannot be paired"
    # 1 moved 2 -> 1, and 2 went from not found at all to rank 4, which is also better
    assert out["better"] == 2 and out["worse"] == 0


def test_a_question_no_chunk_can_answer_is_out_of_the_section_level():
    before = [_row(1, section_rank=2), _row(2, section_rank=3, scorable=False)]
    after = [_row(1, section_rank=1), _row(2, section_rank=1, scorable=False)]
    assert rc.paired_delta(before, after, "section")["questions"] == 1


def test_two_arms_with_nothing_in_common_say_so_instead_of_averaging_nothing():
    assert "error" in rc.paired_delta([_row(1)], [_row(2)], "file")


def test_the_interval_is_reproducible():
    deltas = [0.1, -0.2, 0.3, 0.0, 0.5]
    assert rc.bootstrap_ci(deltas) == rc.bootstrap_ci(deltas)


def test_a_summary_counts_only_what_it_could_score():
    rows = [_row(1, section_rank=1), _row(2, section_rank=None), _row(3, section_rank=9, scorable=False)]
    stats = rc.summarise(rows, "section")
    assert stats["n"] == 2
    assert stats["hit@1"] == 0.5
    assert stats["MRR@20"] == 0.5


def test_an_unknown_axis_is_refused_rather_than_ignored():
    from types import SimpleNamespace

    plan = SimpleNamespace(axes={"chunker": ["rooted"]}, dataset="s", sample_size=None, question_ids=None)
    with pytest.raises(ValueError, match="unknown axes"):
        rc.run(plan)


def test_a_comparison_must_name_the_axis_it_is_reported_along(client):
    body = {
        "kind": "retrieval",
        "dataset": "paraphrased_ru",
        "param": "k",
        "axes": {"variant": ["baseline", "clean_1024"]},
    }
    out = client.post("/v1/experiment", json=body)
    assert out.status_code == 422
    assert "param must name one of the axes" in out.text


def test_an_axis_nobody_applies_is_refused_at_the_route(client):
    body = {
        "kind": "retrieval",
        "dataset": "paraphrased_ru",
        "param": "chunker",
        "axes": {"chunker": ["rooted"]},
    }
    out = client.post("/v1/experiment", json=body)
    assert out.status_code == 422
    assert "unknown axes" in out.text


def test_the_search_mode_does_not_outlive_the_comparison(monkeypatch):
    # the listener sits on the process-wide engine: left installed, it hands this job's
    # search mode to every later job in the same worker
    from orm.sync_db import engine
    from sqlalchemy import event

    rc.prepare(True)
    assert rc._APPLIED is not None
    assert event.contains(engine, "checkout", rc._APPLIED)

    installed = rc._APPLIED
    rc.release()
    assert rc._APPLIED is None
    assert not event.contains(engine, "checkout", installed)


def test_the_guard_releases_even_when_an_arm_raises():
    # the same guard the grid uses, so the shipped path is the tested one
    with pytest.raises(RuntimeError):
        with rc.search_mode(True):
            raise RuntimeError("an arm died")
    assert rc._APPLIED is None


def test_the_grid_guard_releases_a_mode_set_by_any_arm():
    with pytest.raises(RuntimeError):
        with rc.search_mode_restored():
            rc.prepare(True)
            rc.prepare(False, ef=200)
            raise RuntimeError("the second arm died")
    assert rc._APPLIED is None


def test_a_delta_moves_only_the_axis_of_record():
    axes = {"variant": ["baseline", "clean_1024"], "rerank_top": [0, 20]}
    # every arm that is not already at the first value of the axis of record has a
    # reference differing from it in that axis alone
    pairs = {
        rc.arm_name(a): rc.arm_name(rc._reference_for(a, "variant", axes))
        for a in rc.arms(axes)
        if rc._reference_for(a, "variant", axes)
    }
    assert pairs == {
        "rerank_top=0_variant=clean_1024": "rerank_top=0_variant=baseline",
        "rerank_top=20_variant=clean_1024": "rerank_top=20_variant=baseline",
    }, "with two axes one reference for the whole grid would move both at once"


def test_an_arm_already_at_the_reference_value_has_no_delta():
    axes = {"variant": ["baseline", "clean_1024"]}
    assert rc._reference_for({"variant": "baseline"}, "variant", axes) is None


def test_without_an_axis_of_record_nothing_is_compared():
    assert rc._reference_for({"variant": "x"}, None, {"variant": ["x"]}) is None


def test_an_axis_value_is_checked_as_well_as_its_name(client):
    body = {
        "kind": "retrieval",
        "dataset": "paraphrased_ru",
        "param": "variant",
        "axes": {"variant": ["baseline", "no_such_cut"]},
    }
    out = client.post("/v1/experiment", json=body)
    assert out.status_code == 400
    assert "declared" in out.text


def test_an_axis_with_no_values_is_refused_before_the_row_is_running(client):
    body = {
        "kind": "retrieval",
        "dataset": "paraphrased_ru",
        "param": "variant",
        "axes": {"variant": []},
    }
    out = client.post("/v1/experiment", json=body)
    assert out.status_code == 422
    assert "no values" in out.text


def _stub_session(exp, rows_won: int | None = None):
    """rows_won=None makes the stub decide the way Postgres would: the update lands only
    while the row is still running, which is the transition the CAS exists for."""
    from models.experiment import ExperimentStatus

    class _Result:
        def __init__(self, count):
            self.rowcount = count

    class _Session:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def get(self, model, ident):
            return exp

        def execute(self, statement):
            # the WHERE is read from the statement, not decided here: a stub that computes
            # the winner itself lets the compare-and-swap be deleted with the suite green,
            # and that swap is the whole safety property of this state machine
            if rows_won is not None:
                won = rows_won
            else:
                where = str(statement.compile()).split("WHERE", 1)
                guarded = len(where) > 1 and "status" in where[1]
                # an unguarded UPDATE lands, it does not abstain. Modelling a missing
                # guard as "nothing happened" let the guard be deleted with the tests
                # named after it still green
                won = int(exp.status == ExperimentStatus.running) if guarded else 1
            if won:
                for key, value in statement.compile().params.items():
                    if hasattr(exp, key):
                        setattr(exp, key, value)
            return _Result(won)

        def commit(self):
            pass

        def rollback(self):
            pass

        def refresh(self, obj):
            pass

    return _Session


class _Exp:
    def __init__(self, status, results=None):
        from models.experiment import ExperimentStatus  # noqa: F401

        self.id = 1
        self.status = status
        self.axes = {"variant": ["baseline"]}
        self.param = "variant"
        self.dataset = "s"
        self.sample_size = None
        self.question_ids = None
        self.started_at = None
        self.results = results
        self.finished_at = None
        self.elapsed = None


def test_a_retry_reclaims_the_row_before_it_measures_again(monkeypatch):
    # the attempt before it left the row failed, and aggregating from failed is refused,
    # so without this the grid is measured and thrown away
    from job_handlers import evaluation
    from models.experiment import ExperimentStatus

    exp = _Exp(ExperimentStatus.failed)
    # the stub swaps only while the row reads running, so without the reclaim the update
    # lands on nothing and the measured grid goes to the not-advanced branch
    monkeypatch.setattr(evaluation, "Session", _stub_session(exp))
    monkeypatch.setattr(rc, "run", lambda plan: {"arms": {}})
    evaluation.compare_retrieval({"experiment_id": 1})
    assert exp.status == ExperimentStatus.aggregated
    assert exp.started_at is not None, "elapsed must not span the failed attempt"


def test_a_finished_record_is_never_written_over_by_a_late_comparison(monkeypatch):
    # a requeued job on a concluded row would put fresh numbers under a written conclusion
    from job_handlers import evaluation
    from models.experiment import ExperimentStatus

    standing = {"arms": {"the numbers the conclusion was written about": 1}}
    exp = _Exp(ExperimentStatus.concluded, results=standing)
    monkeypatch.setattr(evaluation, "Session", _stub_session(exp))
    monkeypatch.setattr(rc, "run", lambda plan: {"arms": {"fresh": 2}})
    evaluation.compare_retrieval({"experiment_id": 1})
    assert exp.results is standing
    assert exp.status == ExperimentStatus.concluded


def test_a_grid_measured_against_an_empty_record_is_kept(monkeypatch):
    # it destroys nothing there, and an hour of measuring is not a log line
    from job_handlers import evaluation
    from models.experiment import ExperimentStatus

    exp = _Exp(ExperimentStatus.aggregated, results=None)
    monkeypatch.setattr(evaluation, "Session", _stub_session(exp))
    measured = {"arms": {"fresh": 2}}
    monkeypatch.setattr(rc, "run", lambda plan: measured)
    evaluation.compare_retrieval({"experiment_id": 1})
    assert exp.results is measured


def test_a_depth_postgres_would_refuse_is_refused_before_the_arms_are_measured(client):
    body = {
        "kind": "retrieval",
        "dataset": "paraphrased_ru",
        "param": "ef_search",
        "axes": {"ef_search": [100, 5000]},
    }
    out = client.post("/v1/experiment", json=body)
    assert out.status_code == 400
    assert "1..1000" in out.text


def test_a_depth_of_zero_is_refused_rather_than_measured_at_the_configured_one(client):
    # `ef or EF_SEARCH` made the arm search at 100 under a name saying nought
    body = {
        "kind": "retrieval",
        "dataset": "paraphrased_ru",
        "param": "ef_search",
        "axes": {"ef_search": [0, 100]},
    }
    assert client.post("/v1/experiment", json=body).status_code == 400


def test_a_switch_is_not_a_depth(client):
    body = {
        "kind": "retrieval",
        "dataset": "paraphrased_ru",
        "param": "rerank_top",
        "axes": {"rerank_top": [0, True]},
    }
    assert client.post("/v1/experiment", json=body).status_code == 400


def test_every_axis_measure_applies_has_a_rule_and_a_message():
    # the rules are the subject AXES is derived from, so the thing worth asserting is that
    # each of them is refusable and printable, and that measure() reads exactly these
    import inspect

    taken = set(inspect.signature(rc.measure).parameters)
    # `ef_search` reaches measure as `ef`, resolved by depth_of; the rest by their own name
    knobs = {"variant", "ef_search", "limit_vector", "limit_keyword",
             "distance_threshold", "rerank_top", "source"}
    unapplied = {k for k in knobs if k not in taken} - {"ef_search"}
    assert not unapplied, f"an axis measure cannot apply is a knob nobody turns: {unapplied}"
    assert set(rc.AXIS_RULES) == knobs
    assert set(rc.AXIS_LIMITS) == knobs


def test_a_pool_of_nothing_cannot_disarm_the_instrument_that_reports_a_capped_pool(client):
    body = {
        "kind": "retrieval",
        "dataset": "paraphrased_ru",
        "param": "limit_vector",
        "axes": {"limit_vector": [0, 20]},
    }
    assert client.post("/v1/experiment", json=body).status_code == 400


def test_an_empty_axis_stops_the_grid_wherever_it_came_from():
    plan = rc.ComparisonPlan(axes={"variant": []}, param="variant", dataset="s")
    with pytest.raises(ValueError, match="no arms to measure"):
        rc.run(plan)


def test_source_cannot_be_the_axis_of_record():
    # arms along it measure different questions, so every delta is "no shared questions"
    plan = rc.ComparisonPlan(axes={"source": ["a", "b"]}, param="source", dataset="s")
    with pytest.raises(ValueError, match="stratifies"):
        rc.run(plan)


def test_a_cancelled_comparison_stops_instead_of_measuring_the_whole_grid(monkeypatch):
    import job_queue

    monkeypatch.setattr(job_queue, "is_cancelled", lambda job_id: True)
    monkeypatch.setattr(rc, "measure", lambda *a, **kw: pytest.fail("measured after cancel"))
    plan = rc.ComparisonPlan(
        axes={"variant": ["baseline"]}, param="variant", dataset="s", job_id=7
    )
    with pytest.raises(RuntimeError, match="cancelled"):
        rc.run(plan)


def test_the_procedure_of_an_arm_is_the_shape_the_report_writes():
    arm = {"variant": "baseline", "rerank_top": 20, "ef_search": 100}
    proc = rc.arm_procedure(arm, [{"id": 1}, {"id": 2}], "paraphrased_ru")
    missing = [f for f in rc.COMPARABLE if f not in proc]
    assert missing == [], "a record the comparability check cannot read is not a record"
    assert proc["search"] == "hnsw ef_search=100"


def test_a_grid_nobody_meant_to_ask_for_is_refused():
    axes = {"variant": ["a", "b"], "rerank_top": [0, 5, 10, 20],
            "ef_search": [50, 100, 200, 400], "limit_vector": [10, 20]}
    with pytest.raises(ValueError, match="over the cap"):
        rc.arms(axes)


def test_the_stored_axes_are_validated_where_a_retry_reads_them():
    # the route checked them on the way in; the rules moved afterwards and the row did not
    plan = rc.ComparisonPlan(axes={"ef_search": [0, 100]}, param="ef_search", dataset="s")
    with pytest.raises(ValueError, match="1..1000"):
        rc.run(plan)


def test_a_record_says_whether_its_two_arms_were_comparable():
    # the axis of record is allowed to differ and nothing else is; `ef_search` is the axis
    # whose name in the procedure is `search`, so without the map it would flag itself
    base = rc.arm_procedure({"variant": "baseline", "ef_search": 100}, [{"id": 1}], "s")
    arm = rc.arm_procedure({"variant": "baseline", "ef_search": 200}, [{"id": 1}], "s")
    field = rc.AXIS_FIELD.get("ef_search", "ef_search")
    assert rc.comparable({**base, field: None}, {**arm, field: None}) == []
    # and a pair that also moved the candidate pool is not comparable, axis or no axis
    wider = rc.arm_procedure(
        {"variant": "baseline", "ef_search": 200, "limit_vector": 20}, [{"id": 1}], "s"
    )
    assert [f for f, _, _ in rc.comparable({**base, field: None}, {**wider, field: None})] == [
        "limit_vector"
    ]


def test_two_prepares_leave_exactly_one_listener(monkeypatch):
    # the swap read the old listener, removed it, and installed the new one in three
    # steps; two lanes interleaving there left one listener on the engine forever
    from orm.sync_db import engine
    from sqlalchemy import event
    from use_cases import retrieval_compare as rc

    installed = []
    monkeypatch.setattr(event, "listen", lambda t, n, fn: installed.append(fn))
    monkeypatch.setattr(
        event, "remove", lambda t, n, fn: installed.remove(fn) if fn in installed else None
    )
    monkeypatch.setattr(engine, "dispose", lambda: None)
    monkeypatch.setattr(rc, "_APPLIED", None)

    rc.prepare(exact=True)
    rc.prepare(exact=False, ef=200)
    assert len(installed) == 1
    rc.release()
    assert installed == []
