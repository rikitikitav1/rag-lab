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
    assert not can_advance(ExperimentStatus.aggregated, ExperimentStatus.running)


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
    # a branch whose subject is corpus variants could not vary one in the kind of
    # experiment that answers with a judge: `variant` was not in GENERATION_PARAMS, so
    # the validator refused it before anything else could
    import pytest as _pytest
    from api.v1.experiment import ExperimentCreate

    ExperimentCreate(
        dataset="s", param="variant", param_values=["baseline", "clean_1024"]
    )
    # and the gate still gates
    with _pytest.raises(ValueError, match="param must be one of"):
        ExperimentCreate(dataset="s", param="bogus", param_values=[1])


def test_a_generation_experiment_pins_the_corpus_it_did_not_sweep(monkeypatch):
    # every arm carried no variant at all, so the fan-out read the configured default and
    # the record said nothing about which corpus answered
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
    # `arms()` was called in the route body, outside the validator, so its ValueError
    # reached the client as a 500 for what is plainly a bad request
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
    # two arms with one name overwrite each other in `measured`, and the grid silently
    # reports fewer arms than it was asked for
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


def test_annotate_significance_bonferroni():
    from use_cases.experiment import _annotate_significance

    comparisons = {
        "a_vs_b": {
            "faithfulness": {"p": 0.03},
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
    assert out["tests"] == 3
    assert out["threshold"] == round(0.05 / 3, 5)
    faith_b = out["comparisons"]["a_vs_b"]["faithfulness"]
    rel_b = out["comparisons"]["a_vs_b"]["relevance"]
    assert faith_b["significant_raw"] and not faith_b["significant_bonferroni"]
    assert rel_b["significant_raw"] and rel_b["significant_bonferroni"]


def test_annotate_significance_empty():
    from use_cases.experiment import _annotate_significance

    out = _annotate_significance({})
    assert out["tests"] == 0 and out["threshold"] is None


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
    # nothing traversed `running -> failed` on the generation side, so an experiment whose
    # runs all died waited for a sibling that was not coming
    from models.experiment import ExperimentStatus
    from use_cases import experiment as uc

    seen = {}

    # the stub used to hardcode rowcount and never look at the WHERE, so deleting every
    # guard on the update left this test green while the statement failed experiments
    # belonging to other runs
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
    # the three guards the update is not allowed to lose: the kind, the state it is
    # moving out of, and the run that died
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

    # aggregation is reachable only through judging, so a dead judge strands the
    # experiment even when every answer landed
    worker._fail_the_experiment_waiting_on(
        SimpleNamespace(type="judge_answers", options={"run_name": "r2"})
    )
    assert called == ["r", "r2"]


def test_cancelling_an_arm_does_not_leave_its_experiment_running(monkeypatch):
    # aggregation is reachable only through judging, so a cancelled arm strands the row
    # in `running` for ever unless somebody says so
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

    class _Session:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def scalars(self, statement):
            return SimpleNamespace(all=lambda: jobs)

        def commit(self):
            pass

    monkeypatch.setattr(job_queue, "Session", _Session)
    assert job_queue.cancel([1, 2, 3]) == [1, 2, 3]
    assert [j.status for j in jobs] == [JobStatus.cancelled] * 3
    assert failed == ["a", "b"], "an index job carries no experiment"
