import rerank


class _FakeReranker:
    def __init__(self, scores):
        self._scores = scores

    def predict(self, pairs):
        return self._scores


def test_rerank_orders_by_score_and_takes_top(monkeypatch):
    rows = [("doc a",), ("doc b",), ("doc c",)]
    monkeypatch.setattr(rerank, "_model", lambda: _FakeReranker([0.1, 0.9, 0.5]))
    assert rerank.rerank("q", rows, top=2) == [(("doc b",), 0.9), (("doc c",), 0.5)]


def test_rerank_empty_rows():
    assert rerank.rerank("q", [], top=3) == []


def test_rerank_single_row(monkeypatch):
    monkeypatch.setattr(rerank, "_model", lambda: _FakeReranker([0.7]))
    assert rerank.rerank("q", [("only",)], top=3) == [(("only",), 0.7)]


def test_device_reports_the_loaded_model_not_the_intent(monkeypatch):
    monkeypatch.setattr(rerank, "requested_device", lambda: "cuda")
    monkeypatch.setattr(rerank, "_reranker", None)
    assert rerank.device() == "cuda"

    loaded = type("M", (), {"model": type("Inner", (), {"device": "cpu"})()})()
    monkeypatch.setattr(rerank, "_reranker", loaded)
    assert rerank.device() == "cpu"


def test_cuda_oom_falls_back_to_cpu(monkeypatch):
    import torch

    calls = []

    def boom():
        raise torch.cuda.OutOfMemoryError("no room")

    class _Cpu:
        def __init__(self, name, device=None, **kw):
            calls.append(device)

        def predict(self, pairs):
            return [0.5] * len(pairs)

    monkeypatch.setattr(rerank, "_model", boom)
    monkeypatch.setattr(rerank, "unload", lambda: calls.append("unload"))
    monkeypatch.setattr("sentence_transformers.CrossEncoder", _Cpu)

    scores = rerank.score_pairs([("q", "chunk")])

    assert scores == [0.5]
    assert calls == ["unload", "cpu"]
    assert isinstance(rerank._reranker, _Cpu)
