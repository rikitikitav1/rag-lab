import rerank


class _FakeReranker:
    def __init__(self, scores):
        self._scores = scores

    def predict(self, pairs):
        return self._scores


def test_rerank_orders_by_score_and_takes_top(monkeypatch):
    rows = [("doc a",), ("doc b",), ("doc c",)]
    monkeypatch.setattr(rerank, "_model", lambda: _FakeReranker([0.1, 0.9, 0.5]))
    assert rerank.rerank("q", rows, top=2) == [("doc b",), ("doc c",)]


def test_rerank_empty_rows():
    assert rerank.rerank("q", [], top=3) == []


def test_rerank_single_row(monkeypatch):
    monkeypatch.setattr(rerank, "_model", lambda: _FakeReranker([0.7]))
    assert rerank.rerank("q", [("only",)], top=3) == [("only",)]
