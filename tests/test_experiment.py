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

    results = exp.compute_results("run", ["run_hi", "run_lo"], ["run_hi", "run_lo"])

    assert results["composite"]["winner"] == "run_hi"
    assert results["composite"]["axes"] == ["faithfulness", "relevance", "completeness"]
    assert results["per_value"]["run_lo"]["hit_at_k"] == 0.9


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
