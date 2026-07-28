import pytest
from fastapi.testclient import TestClient
from models.experiment import ExperimentStatus, can_advance


@pytest.fixture
def client(monkeypatch):
    import bootstrap

    monkeypatch.setattr(bootstrap, "bootstrap_models", lambda: None)

    import server
    from orm.async_db import get_session

    async def _dummy_session():
        yield None

    server.app.dependency_overrides[get_session] = _dummy_session
    with TestClient(server.app) as c:
        yield c
    server.app.dependency_overrides.clear()


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
        "n_scored": 10,
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
    assert results["composite"]["axes"] == ["faithfulness", "relevance", "completeness"]
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
