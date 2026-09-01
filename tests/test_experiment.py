import re
from types import SimpleNamespace

from models.experiment import ExperimentStatus, can_advance


def test_can_advance_valid():
    assert can_advance(ExperimentStatus.draft, ExperimentStatus.running)
    assert can_advance(ExperimentStatus.running, ExperimentStatus.aggregated)
    assert can_advance(ExperimentStatus.running, ExperimentStatus.failed)
    assert can_advance(ExperimentStatus.aggregated, ExperimentStatus.concluded)
    assert can_advance(ExperimentStatus.failed, ExperimentStatus.running)


def test_can_advance_invalid():
    assert not can_advance(ExperimentStatus.draft, ExperimentStatus.aggregated)
    assert not can_advance(ExperimentStatus.running, ExperimentStatus.concluded)
    assert not can_advance(ExperimentStatus.concluded, ExperimentStatus.running)


def test_a_read_experiment_can_take_another_arm():
    assert can_advance(ExperimentStatus.aggregated, ExperimentStatus.running)


def _gen(faith, rel, compl):
    return {
        "faithfulness": faith,
        "relevance": rel,
        "completeness": compl,
        "faithfulness_0_1": faith / 10,
        "relevance_0_1": rel / 10,
        "completeness_0_1": compl / 10,
        "refusal_accuracy": "0/0",
        "off_domain_refusal": "0/0",
        "off_domain_grounding": None,
        "off_domain_refusal_rate": None,
        "supported_rate": 1.0,
        "n_off_domain_scored": 0,
        "unsupported_external": "0/0",
        "unsupported_off_domain": "0/0",
        "narrated_calls": 0,
        "outcomes": {},
        "false_refusal": "0/10",
        "answer_rate": 1.0,
        "answered_via_remote": 0,
        "remote_grounding": None,
        "remote_relevance": None,
        "n_scored": 10,
        "n_remote_scored": 0,
    }


def test_compute_results_rrf_winner_ignores_retrieval(monkeypatch):
    from use_cases import experiment as exp

    gen = {"run_hi": _gen(9, 9, 9), "run_lo": _gen(5, 5, 5)}
    # retrieval favors run_lo to prove hit@k does not decide the winner
    ret = {
        "run_hi": {"hit_at_k": 0.1, "mrr": 0.1, "hits": 1, "n": 10, "misses": 9},
        "run_lo": {"hit_at_k": 0.9, "mrr": 0.9, "hits": 9, "n": 10, "misses": 1},
    }
    monkeypatch.setattr(exp.generation_metrics, "evaluate", lambda rn: gen[rn])
    monkeypatch.setattr(exp.retrieval_metrics, "evaluate", lambda rn: ret[rn])
    monkeypatch.setattr(exp, "load_logs", lambda rn: [])

    results = exp.compute_results("run", ["run_hi", "run_lo"], ["run_hi", "run_lo"])

    assert results["composite"]["winner"] == "run_hi"
    assert results["composite"]["axes"][:3] == ["faithfulness", "relevance", "completeness"]
    assert results["per_value"]["run_lo"]["hit_at_k"] == 0.9
    assert results["composite"]["pairwise"]["tests"] == 0


def test_experiment_empty_values_422(client):
    r = client.post("/v1/experiment", json={"dataset": "s", "param": "k", "param_values": []})
    assert r.status_code == 422


def test_experiment_bad_param_422(client):
    r = client.post(
        "/v1/experiment", json={"dataset": "s", "param": "bogus", "param_values": [3]}
    )
    assert r.status_code == 422


def test_experiment_model_values_must_be_valid_names(client):
    r = client.post(
        "/v1/experiment",
        json={"dataset": "s", "param": "model", "param_values": ["bad name!"]},
    )
    assert r.status_code == 400


def test_experiment_model_values_reject_ints(client):
    r = client.post(
        "/v1/experiment", json={"dataset": "s", "param": "model", "param_values": [5]}
    )
    assert r.status_code == 400


def test_experiment_int_param_rejects_strings(client):
    r = client.post(
        "/v1/experiment",
        json={"dataset": "s", "param": "k", "param_values": ["llama3.1:8b"]},
    )
    assert r.status_code == 400


def _generation_fan_out(monkeypatch, body) -> list[dict]:
    """The options each arm is queued with, without a database behind the route."""
    from types import SimpleNamespace

    import api.v1.experiment as mod
    from api.v1.experiment import ExperimentCreate

    seen: list[dict] = []
    monkeypatch.setattr(
        mod.job_queue, "add_job", lambda s, t, o: seen.append(o) or SimpleNamespace(id=1)
    )
    request = ExperimentCreate(**body)
    exp = SimpleNamespace(id=1, run_names=[], status=None, started_at=None)
    for value in request.param_values:
        mod.job_queue.add_job(
            None,
            "eval_run",
            {
                "run_name": f"base_{request.param}_{value}",
                "set_name": request.dataset,
                "language": request.language,
                "variant": request.variant,
                request.param: value,
                "experiment_id": exp.id,
            },
        )
    return seen


def test_a_generation_experiment_can_sweep_the_corpus():
    # a branch about corpus variants could not vary one in the kind that answers with a judge
    import pytest as _pytest
    from api.v1.experiment import ExperimentCreate

    ExperimentCreate(
        dataset="s", param="variant", param_values=["baseline", "clean_1024"]
    )
    # and the gate still gates
    with _pytest.raises(ValueError, match="param must be one of"):
        ExperimentCreate(dataset="s", param="bogus", param_values=[1])


def test_a_generation_experiment_pins_the_corpus_it_did_not_sweep(monkeypatch):
    # every arm carried no variant, so the fan-out read the default and recorded nothing
    seen = _generation_fan_out(
        monkeypatch,
        {"dataset": "s", "param": "k", "param_values": [3, 5], "variant": "clean_1024"},
    )
    assert [o["variant"] for o in seen] == ["clean_1024", "clean_1024"]


def test_a_swept_variant_is_not_overwritten_by_the_pinned_one(monkeypatch):
    seen = _generation_fan_out(
        monkeypatch,
        {
            "dataset": "s", "param": "variant",
            "param_values": ["baseline", "clean_1024"], "variant": "baseline",
        },
    )
    assert [o["variant"] for o in seen] == ["baseline", "clean_1024"]


def test_a_grid_over_the_cap_is_refused_at_the_door(client):
    # `arms()` ran in the route body, outside the validator, so its ValueError was a 500
    r = client.post(
        "/v1/experiment",
        json={
            "kind": "retrieval", "dataset": "s", "param": "ef_search",
            "axes": {"variant": ["baseline"], "ef_search": list(range(1, 34))},
        },
    )
    assert r.status_code == 422, r.text
    assert "cap" in r.text


def test_source_as_the_axis_of_record_is_refused_at_the_door(client):
    # the job refuses it after the row reached running and the worker retried three times
    r = client.post(
        "/v1/experiment",
        json={
            "kind": "retrieval", "dataset": "s", "param": "source",
            "axes": {"source": ["a", "b"]},
        },
    )
    assert r.status_code == 422, r.text


def test_arms_that_share_a_name_are_refused_at_the_door(client):
    # two arms with one name overwrite each other and the grid reports fewer than it ran
    r = client.post(
        "/v1/experiment",
        json={
            "kind": "retrieval", "dataset": "s", "param": "variant",
            "axes": {"variant": ["baseline", "baseline"]},
        },
    )
    assert r.status_code == 422, r.text
    assert "distinct names" in r.text


def test_value_suffix_formats():
    from api.v1.eval import value_suffix

    assert value_suffix(5) == "05"
    assert value_suffix("llama3.1:70b") == "llama3.1_70b"


def _log(qid, faith=None, rel=None, compl=None):
    from types import SimpleNamespace

    return SimpleNamespace(
        question_id=qid, faithfulness=faith, relevance=rel, completeness=compl
    )


def test_paired_logs_matches_by_question_id():
    from use_cases.experiment import _paired_logs

    a = [_log(1), _log(2), _log(3)]
    b = [_log(3), _log(1)]
    pairs = _paired_logs(a, b)
    assert [(x.question_id, y.question_id) for x, y in pairs] == [(1, 1), (3, 3)]


def test_axis_deltas_skips_missing_scores():
    from use_cases.experiment import _axis_deltas, _paired_logs

    a = [_log(1, faith="5"), _log(2, faith=None), _log(3, faith="7")]
    b = [_log(1, faith="8"), _log(2, faith="9"), _log(3, faith=None)]
    deltas = _axis_deltas(_paired_logs(a, b), "faithfulness")
    assert deltas == [3]


def test_compare_identical_runs_gives_p_one():
    from use_cases.experiment import _compare_question_sets

    logs = [_log(i, faith="7", rel="8", compl="6") for i in range(10)]
    out = _compare_question_sets(logs, logs)
    assert out["faithfulness"]["p"] == 1.0
    assert out["faithfulness"]["mean_delta"] == 0.0
    assert out["faithfulness"]["ci95"] == [0.0, 0.0]


def test_holm_keeps_what_bonferroni_would_have_thrown_away():
    from use_cases.experiment import _annotate_significance

    comparisons = {
        "a_vs_b": {
            "faithfulness": {"p": 0.02},
            "relevance": {"p": 0.001},
            "completeness": None,
        },
        "a_vs_c": {
            "faithfulness": {"p": 0.5},
            "relevance": None,
            "completeness": None,
        },
    }
    out = _annotate_significance(comparisons)
    assert (out["tests"], out["method"]) == (3, "holm")
    faith_b = out["comparisons"]["a_vs_b"]["faithfulness"]
    rel_b = out["comparisons"]["a_vs_b"]["relevance"]

    # 0.001 survives its step and lets the next be judged against 0.05/2, which Bonferroni held
    assert rel_b["significant_holm"] and rel_b["holm_threshold"] == round(0.05 / 3, 5)
    assert faith_b["significant_holm"] and faith_b["holm_threshold"] == round(0.05 / 2, 5)
    assert faith_b["p"] > 0.05 / 3, "this is the test bonferroni would have thrown away"
    assert not out["comparisons"]["a_vs_c"]["faithfulness"]["significant_holm"]


def test_holm_stops_at_the_first_failure():
    from use_cases.experiment import _annotate_significance

    # the third passes its own step, but the second already failed and holds everything after
    comparisons = {
        "a_vs_b": {"faithfulness": {"p": 0.001}, "relevance": {"p": 0.04}},
        "a_vs_c": {"faithfulness": {"p": 0.02}, "relevance": {"p": 0.9}},
    }
    out = _annotate_significance(comparisons)
    kept = {
        f"{pair}/{axis}": s["significant_holm"]
        for pair, axes in out["comparisons"].items()
        for axis, s in axes.items()
    }
    assert kept == {
        "a_vs_b/faithfulness": True,
        "a_vs_b/relevance": False,
        "a_vs_c/faithfulness": False,
        "a_vs_c/relevance": False,
    }


def test_annotate_significance_empty():
    from use_cases.experiment import _annotate_significance

    out = _annotate_significance({})
    assert out["tests"] == 0 and out["comparisons"] == {}


def test_compare_detects_consistent_shift():
    from use_cases.experiment import _compare_question_sets

    a = [_log(i, faith="5") for i in range(20)]
    b = [_log(i, faith="7") for i in range(20)]
    out = _compare_question_sets(a, b)
    stats = out["faithfulness"]
    assert stats["mean_delta"] == 2.0
    assert stats["p"] < 0.05
    assert stats["ci95"] == [2.0, 2.0]
    assert out["relevance"] is None


def test_a_run_that_ran_out_of_attempts_moves_its_experiment_to_failed(monkeypatch):
    # nothing traversed `running -> failed`, so an experiment whose runs died waited for ever
    from models.experiment import ExperimentStatus
    from use_cases import experiment as uc

    seen = {}

    # the stub hardcoded rowcount and never read the WHERE, so deleting a guard stayed green
    def _session_returning(rowcount):
        class _Result:
            pass

        class _Session:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def execute(self, statement):
                seen["sql"] = str(statement)
                seen["values"] = statement.compile().params
                result = _Result()
                result.rowcount = rowcount
                return result

            def commit(self):
                seen["committed"] = True

            def rollback(self):
                seen["committed"] = False

        return _Session

    monkeypatch.setattr(uc, "Session", _session_returning(1))
    uc.mark_failed_for_run("some_run")
    assert seen["committed"] is True
    assert ExperimentStatus.failed in seen["values"].values()
    # the three guards the update may not lose: the kind, the state and the run
    where = seen["sql"].split("WHERE", 1)[1]
    for column in ("kind", "status", "run_names"):
        assert column in where, f"the update stopped constraining {column}: {where}"

    # and the other branch: an experiment that already moved on is not touched again
    monkeypatch.setattr(uc, "Session", _session_returning(0))
    uc.mark_failed_for_run("some_run")
    assert seen["committed"] is False


def test_the_worker_fails_experiments_for_the_jobs_that_carry_them(monkeypatch):
    import worker
    from use_cases import experiment as uc

    called = []
    monkeypatch.setattr(uc, "mark_failed_for_run", lambda run: called.append(run))

    worker._fail_the_experiment_waiting_on(
        SimpleNamespace(type="index_data", options={"run_name": "r"})
    )
    assert called == [], "an index job has no experiment waiting on it"

    worker._fail_the_experiment_waiting_on(
        SimpleNamespace(type="eval_run", options={"run_name": "r"})
    )
    assert called == ["r"]

    # aggregation is reachable only through judging, so a dead judge strands the experiment
    worker._fail_the_experiment_waiting_on(
        SimpleNamespace(type="judge_answers", options={"run_name": "r2"})
    )
    assert called == ["r", "r2"]


def test_a_bulk_cancel_takes_each_run_judge_with_it(monkeypatch):
    # the single door was given the rule and the bulk door was not: `every: true` filters by type
    import job_queue
    from models import JobStatus

    jobs = [
        SimpleNamespace(id=1, type="eval_run", options={"run_name": "a"}, status=JobStatus.new),
        SimpleNamespace(
            id=9, type="judge_answers", options={"run_name": "a"}, status=JobStatus.new
        ),
    ]

    class _Result(list):
        def all(self):
            return list(self)

    class _Session:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def scalars(self, statement):
            # the fake honours `id.in_(...)`, or the widening it is here to prove is invisible
            if [c["name"] for c in statement.column_descriptions] == ["id"]:
                return _Result([9])
            asked = [
                v
                for value in statement.compile().params.values()
                for v in (value if isinstance(value, (list, tuple, set)) else [value])
            ]
            return _Result([j for j in jobs if j.id in asked])

        def commit(self):
            pass

    monkeypatch.setattr(job_queue, "Session", _Session)

    assert sorted(job_queue.cancel([1])) == [1, 9], "the judge of a cancelled run goes too"


def test_cancelling_an_arm_does_not_leave_its_experiment_running(monkeypatch):
    # a cancelled arm strands the row in `running` unless something fails it
    import job_queue
    from models import JobStatus
    from use_cases import experiment as uc

    failed = []
    monkeypatch.setattr(uc, "mark_failed_for_run", lambda run: failed.append(run))

    jobs = [
        SimpleNamespace(id=1, type="eval_run", options={"run_name": "a"}, status=JobStatus.new),
        SimpleNamespace(
            id=2, type="judge_answers", options={"run_name": "b"}, status=JobStatus.running
        ),
        SimpleNamespace(id=3, type="index_data", options={}, status=JobStatus.new),
    ]

    class _Result(list):
        def all(self):
            return list(self)

    class _Session:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def scalars(self, statement):
            # `select(Job.id)` and `select(Job)` reach the same fake and want different rows
            wanted = [c["name"] for c in statement.column_descriptions]
            return _Result([j.id for j in jobs] if wanted == ["id"] else jobs)

        def commit(self):
            pass

    monkeypatch.setattr(job_queue, "Session", _Session)
    assert job_queue.cancel([1, 2, 3]) == [1, 2, 3]
    assert [j.status for j in jobs] == [JobStatus.cancelled] * 3
    assert failed == ["a", "b"], "an index job carries no experiment"


def test_every_kind_of_report_declares_its_schema():
    # a record written before a field existed is indistinguishable from one where it is absent
    from evals import generation_metrics, retrieval_metrics
    from use_cases import experiment, rejudge, retrieval_compare, run_snapshot

    assert (experiment.SCHEMA, rejudge.SCHEMA, retrieval_compare.SCHEMA) == (3, 4, 2)
    # the summaries the report is computed from, and the row snapshot they are computed over
    assert (generation_metrics.SCHEMA, retrieval_metrics.SCHEMA, run_snapshot.SCHEMA) == (2, 3, 2)


def test_pending_counts_the_rows_the_judge_would_pick_up():
    # counting faithfulness-with-context read a run whose rows have none as fully judged
    from use_cases import experiment

    seen = []

    class _Session:
        def scalar(self, statement):
            seen.append(statement)
            return 0

    experiment._run_pending(_Session(), "some_run")
    sql = str(seen[0])

    # completeness reads a column of `questions`, so without the join it is a cross product
    assert "JOIN questions" in sql
    for axis in ("relevance", "faithfulness", "completeness"):
        assert f"question_logs.{axis} IS NULL" in sql


def _failed(run_names=("a", "b")):
    from models.experiment import ExperimentStatus

    return SimpleNamespace(status=ExperimentStatus.failed, run_names=list(run_names))


def test_a_sibling_arm_revives_a_row_only_when_every_arm_is_judged(monkeypatch):
    # cancelling one arm fails the row, and the arm still judging finished into that status
    from models.experiment import ExperimentStatus
    from use_cases import experiment

    monkeypatch.setattr(experiment, "_series_complete", lambda session, runs: True)
    assert experiment.revive_if_complete(None, _failed())

    monkeypatch.setattr(experiment, "_series_complete", lambda session, runs: False)
    assert not experiment.revive_if_complete(None, _failed()), (
        "a row whose arms are genuinely unjudged keeps the status a human can see"
    )

    monkeypatch.setattr(experiment, "_series_complete", lambda session, runs: True)
    running = SimpleNamespace(status=ExperimentStatus.running, run_names=["a"])
    assert not experiment.revive_if_complete(None, running)
    assert not experiment.revive_if_complete(None, None)


def test_a_rejudge_can_name_its_arms_instead_of_multiplying_axes(client):
    # two axes moving together are a diagonal, and `axes` describes only the cross product
    body = {
        "name": "split", "kind": "rejudge", "source_run": "src", "param": "judge_faithfulness",
        "axes": {"judge_faithfulness": [4, 5], "judge_relevance": [4, 5]},
        "arms": [{"judge_faithfulness": 4, "judge_relevance": 4}],
    }
    out = client.post("/v1/experiment", json=body)
    assert out.status_code == 422
    assert "either axes or arms" in out.text

    body["axes"] = {}
    body["arms"] = [{"judge_faithfulness": 4, "judge_relevance": 4}, {"nonsense": 1}]
    out = client.post("/v1/experiment", json=body)
    assert out.status_code == 422, "an arm is validated the same way an axis is"


def test_every_swept_parameter_takes_a_value_of_its_own_kind():
    # `variant` was checked before the chain, so its name fell through to the integer branch
    import typing

    import config
    from api.v1.eval import ExperimentRequest, validate_param_values
    from models.registry import Pipeline
    from use_cases.agent_policy import GONE, FallbackPolicy, GateSignal, Orchestrator

    live = sorted({o.value for o in Orchestrator} - {o.value for o in GONE})
    good = {
        "k": [3, 5],
        "max_hops": [2, 4],
        "model": ["llama3.1:8b"],
        "fallback_policy": [next(iter(FallbackPolicy)).value],
        "gate_signal": [next(iter(GateSignal)).value],
        "weak_distance": [0.5],
        "topic_threshold": [0.5, 1],
        "orchestrator": live[:1],
        "variant": [config.settings.corpus.variant],
    }
    declared = set(typing.get_args(ExperimentRequest.model_fields["param"].annotation))

    assert declared == set(good), "a parameter was added to the door and not to this table"
    for param, values in good.items():
        validate_param_values(param, values, Pipeline.agent)


def test_an_arm_added_later_is_built_the_way_the_arms_before_it_were(monkeypatch, client):
    # creation carried the sample and the control size; this door carried neither
    from types import SimpleNamespace

    from api.v1 import experiment as door
    from models.experiment import ExperimentKind, ExperimentStatus

    exp = SimpleNamespace(
        id=1, kind=ExperimentKind.rejudge, status=ExperimentStatus.aggregated,
        name="r", run_names=["r_a"], axes={"judge_faithfulness": [2]},
        param="judge_faithfulness", param_values=[2], question_ids=[11, 12, 13],
        results={"old": 1},
        procedure={"source_run": "src", "base": "r", "control_sample": 7,
                   "control_seed": 4,
                   "arms": [{"arm": {"judge_faithfulness": 2}, "run_name": "r_a"}]},
        started_at=None, finished_at=None, elapsed=None,
    )

    class _Session:
        async def scalars(self, _stmt):
            return SimpleNamespace(first=lambda: exp)

        async def commit(self):
            pass

        async def refresh(self, _obj):
            pass

    seen = {}
    monkeypatch.setattr(door.rejudge, "judges_not_ready", lambda arms: [])
    monkeypatch.setattr(door.rejudge, "unseeded_prompt_versions", lambda axes: [])
    monkeypatch.setattr(door.rejudge, "refuse_unpaired_rejudge", lambda *a: None)
    monkeypatch.setattr(door.rejudge, "paired_arms", lambda e: [])
    monkeypatch.setattr(door.rejudge, "stored_arms", lambda pairs: [])
    monkeypatch.setattr(
        door.rejudge, "refuse_oversized_fanout",
        lambda src, n, existing=0, question_ids=None: seen.update(fanout=question_ids),
    )
    monkeypatch.setattr(
        door.rejudge, "copy_runs",
        lambda src, names, question_ids=None: seen.update(copied=question_ids) or {},
    )
    monkeypatch.setattr(
        door.rejudge, "arm_options",
        lambda arm, run_name, control_sample=None, control_seed=0: seen.update(
            control=control_sample, seed=control_seed
        ) or {},
    )
    monkeypatch.setattr(door.job_queue, "add_job", lambda session, type, options: None)

    import asyncio

    asyncio.run(door.add_arms(1, door.ArmsAdd(arms=[{"judge_faithfulness": 3}]), _Session()))

    assert seen["copied"] == [11, 12, 13], "the new arm copies the rows the others read"
    assert seen["fanout"] == [11, 12, 13], "and the size guard measures those rows"
    assert seen["control"] == 7, "and judges the control axis on the same sample size"
    assert seen["seed"] == 4, "drawn with the seed the experiment was created with"


def test_the_arm_that_cannot_be_merged_gives_its_copied_rows_back(monkeypatch, client):
    # the copies commit in their own session, so a refusal after them burns the names for ever
    import asyncio
    from types import SimpleNamespace

    import pytest
    from api.v1 import experiment as door
    from fastapi import HTTPException
    from models.experiment import ExperimentKind, ExperimentStatus

    exp = SimpleNamespace(
        id=1, kind=ExperimentKind.rejudge, status=ExperimentStatus.aggregated,
        name="r", run_names=["r_a"], axes={},
        param="judge_faithfulness", param_values=[2], question_ids=[11],
        results={"old": 1},
        procedure={"source_run": "src", "base": "r"},
        started_at=None, finished_at=None, elapsed=None,
    )

    class _Session:
        async def scalars(self, _stmt):
            return SimpleNamespace(first=lambda: exp)

        async def commit(self):
            pass

        async def refresh(self, _obj):
            pass

    deleted = []
    monkeypatch.setattr(door.rejudge, "judges_not_ready", lambda arms: [])
    monkeypatch.setattr(door.rejudge, "unseeded_prompt_versions", lambda axes: [])
    monkeypatch.setattr(door.rejudge, "refuse_unpaired_rejudge", lambda *a: None)
    monkeypatch.setattr(door.rejudge, "refuse_oversized_fanout", lambda *a, **kw: None)
    monkeypatch.setattr(door.rejudge, "paired_arms", lambda e: [])
    monkeypatch.setattr(door.rejudge, "stored_arms", lambda pairs: [])
    monkeypatch.setattr(door.rejudge, "copy_runs", lambda *a, **kw: {})
    monkeypatch.setattr(door.rejudge, "arm_options", lambda *a, **kw: {})
    monkeypatch.setattr(door.rejudge, "delete_runs", lambda names: deleted.extend(names))
    monkeypatch.setattr(door.job_queue, "add_job", lambda session, type, options: None)

    with pytest.raises(HTTPException) as raised:
        asyncio.run(
            door.add_arms(1, door.ArmsAdd(arms=[{"judge_relevance": 3}]), _Session())
        )

    assert raised.value.status_code == 409
    assert deleted == ["r_judge_relevance=3"], "the refusal takes the copied rows back"


def test_every_kind_of_report_answers_the_reader_with_the_same_keys():
    # the reading tool projected the rejudge shape by hand, so generation read back empty
    from mcp_ops import READING
    from models.experiment import READING_KEYS, ExperimentKind

    assert set(READING) == set(ExperimentKind), "a kind of experiment has no reader"
    for kind, reader in READING.items():
        assert set(reader({})) == set(READING_KEYS), f"{kind} answers with another shape"


def test_a_generation_report_reads_back_with_its_arms_and_deltas():
    from use_cases import experiment as experiment_uc

    report = {
        "schema": 3,
        "param": "variant",
        "per_value": {"clean_1024": {"run_name": "r1", "n_scored": 60, "relevance": 8.83}},
        "composite": {
            "method": "rrf", "winner": "clean_1024",
            "ranking": [{"value": "clean_1024", "rrf": 0.065}],
            "pairwise": {
                "method": "holm", "alpha": 0.05, "family": "every pair on every axis",
                "comparisons": {"a_vs_b": {"relevance": {"p": 0.3}}},
            },
        },
    }
    read = experiment_uc.for_reading(report)
    assert read["arms"]["clean_1024"]["n_scored"] == 60
    assert read["deltas"] == {"a_vs_b": {"relevance": {"p": 0.3}}}
    assert read["multiplicity"]["method"] == "holm", "the correction it promises is reported"
    assert read["ranking"]["winner"] == "clean_1024"


def test_a_retrieval_report_reads_back_with_the_arms_it_has():
    from use_cases import retrieval_compare

    read = retrieval_compare.for_reading(
        {"reference_axis": "variant", "arms": {"a": {"file": {}}}, "deltas": {"b": {"against": "a"}}}
    )
    assert read["arms"] == {"a": {"file": {}}}, "its per-arm summaries are under `arms`"
    assert read["deltas"] == {"b": {"against": "a"}}


# every key the generation report carries under schema 3
SCHEMA_3_SHAPE = [
    ".composite.axes[]",
    ".composite.k",
    ".composite.method",
    ".composite.pairwise.alpha",
    ".composite.pairwise.comparisons.<pair>.completeness",
    ".composite.pairwise.comparisons.<pair>.faithfulness",
    ".composite.pairwise.comparisons.<pair>.relevance",
    ".composite.pairwise.family",
    ".composite.pairwise.method",
    ".composite.pairwise.tests",
    ".composite.ranking[].rrf",
    ".composite.ranking[].value",
    ".composite.winner",
    ".param",
    ".per_value.<arm>.answer_rate",
    ".per_value.<arm>.answered_via_remote",
    ".per_value.<arm>.completeness",
    ".per_value.<arm>.faithfulness",
    ".per_value.<arm>.false_refusal",
    ".per_value.<arm>.hit_at_k",
    ".per_value.<arm>.mrr",
    ".per_value.<arm>.n_off_domain_scored",
    ".per_value.<arm>.n_remote_scored",
    ".per_value.<arm>.n_scored",
    ".per_value.<arm>.narrated_calls",
    ".per_value.<arm>.off_domain_grounding",
    ".per_value.<arm>.off_domain_refusal",
    ".per_value.<arm>.off_domain_refusal_rate",
    ".per_value.<arm>.outcomes",
    ".per_value.<arm>.refusal_accuracy",
    ".per_value.<arm>.relevance",
    ".per_value.<arm>.remote_grounding",
    ".per_value.<arm>.remote_relevance",
    ".per_value.<arm>.run_name",
    ".per_value.<arm>.supported_rate",
    ".per_value.<arm>.unsupported_external",
    ".per_value.<arm>.unsupported_off_domain",
    ".schema",
]


def _shape_of(value, prefix="") -> list[str]:
    # an empty container is a key the record carries, so it names itself rather than vanishing
    if isinstance(value, dict) and value:
        return sorted(p for k, v in value.items() for p in _shape_of(v, f"{prefix}.{k}"))
    if isinstance(value, list) and value:
        return sorted({p for v in value for p in _shape_of(v, f"{prefix}[]")})
    return [prefix]


def test_the_generation_report_declares_a_new_schema_when_its_shape_moves(monkeypatch):
    # the check beside this compares two constants: it catches a bump that forgot the test
    from use_cases import experiment as exp

    gen = {"a": _gen(9, 9, 9), "b": _gen(5, 5, 5)}
    ret = dict.fromkeys(gen, {"hit_at_k": 0.5, "mrr": 0.5, "hits": 5, "n": 10, "misses": 5})
    monkeypatch.setattr(exp.generation_metrics, "evaluate", lambda rn: gen[rn])
    monkeypatch.setattr(exp.retrieval_metrics, "evaluate", lambda rn: ret[rn])
    monkeypatch.setattr(exp, "load_logs", lambda rn: [])

    report = exp.compute_results("run", ["a", "b"], ["a", "b"])
    shape = sorted({
        re.sub(r"\.per_value\.[ab]\.", ".per_value.<arm>.", p).replace("a_vs_b", "<pair>")
        for p in _shape_of(report)
    })

    # every key the record carries under schema 3
    assert (exp.SCHEMA, shape) == (3, SCHEMA_3_SHAPE)


def test_a_hop_budget_is_bounded_at_every_door_that_takes_one(client):
    from use_cases.agent_policy import MAX_HOPS

    over = MAX_HOPS + 1
    assert client.post("/v1/agent/question", json={"text": "q", "max_hops": over}).status_code == 422
    assert client.post(
        "/v1/eval/run", json={"run_name": "r", "pipeline": "agent", "max_hops": over}
    ).status_code == 422
    swept = client.post(
        "/v1/eval/experiment",
        json={"param": "max_hops", "values": [over], "pipeline": "agent"},
    )
    assert swept.status_code == 400 and "1..10" in swept.text


def test_a_field_only_a_rejudge_reads_is_refused_by_the_other_kinds(client):
    out = client.post(
        "/v1/experiment",
        json={"kind": "generation", "dataset": "s", "param": "k", "param_values": [3],
              "control_sample": 200},
    )
    assert out.status_code == 422 and "apply to a rejudge" in out.text
