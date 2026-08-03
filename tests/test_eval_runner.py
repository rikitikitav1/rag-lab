from evals import runner


def _rows(marker, n=3):
    return [(f"chunk {marker} {i}", f"{marker}.md", "cat", i, 1, None, 0.1, 0.5) for i in range(n)]


def _stub_phases(monkeypatch, use_rerank_expected=None):
    calls = []
    monkeypatch.setattr(
        runner.llm, "request_embeddings_batch", lambda texts: [[0.1]] * len(texts)
    )
    monkeypatch.setattr(
        runner.db, "hybrid_search",
        lambda text, vector, category, limit: calls.append(("search", text, limit)) or _rows(text),
    )
    monkeypatch.setattr(
        runner.rerank, "rerank",
        lambda text, rows, top: calls.append(("rerank", text, top)) or rows[:top],
    )
    monkeypatch.setattr(runner.llm, "unload", lambda role: calls.append(("unload", role)))
    monkeypatch.setattr(runner.rerank, "unload", lambda: calls.append(("unload", "reranker")))
    monkeypatch.setattr(
        runner.chat, "answer_from_rows",
        lambda text, rows, **kw: calls.append(("generate", text, kw.get("phased"))),
    )
    return calls


def test_phases_run_in_order_and_free_vram(monkeypatch):
    calls = _stub_phases(monkeypatch)
    answered, cancelled = runner.run_phased(
        "run", ["q1", "q2"], use_rerank=True, language=None, k=2, model=None, job_id=None
    )
    assert (answered, cancelled) == (2, False)
    kinds = [c[0] for c in calls]
    assert kinds == [
        "search", "search",
        "unload", "unload",
        "rerank", "rerank",
        "unload",
        "generate", "generate",
    ]


def test_retrieval_widens_only_when_reranking(monkeypatch):
    calls = _stub_phases(monkeypatch)
    runner.run_phased("run", ["q"], use_rerank=True, language=None, k=3, model=None, job_id=None)
    wide = [c[2] for c in calls if c[0] == "search"][0]

    calls.clear()
    runner.run_phased("run", ["q"], use_rerank=False, language=None, k=3, model=None, job_id=None)
    narrow = [c[2] for c in calls if c[0] == "search"][0]

    assert wide == runner.config.settings.rerank.candidates
    assert narrow == 3
    assert not [c for c in calls if c[0] in ("rerank", "unload")]


def test_generation_marks_logs_as_phased(monkeypatch):
    calls = _stub_phases(monkeypatch)
    runner.run_phased("run", ["q"], use_rerank=False, language=None, k=2, model=None, job_id=None)
    assert [c[2] for c in calls if c[0] == "generate"] == [True]


def test_cancel_stops_generation_midway(monkeypatch):
    calls = _stub_phases(monkeypatch)
    monkeypatch.setattr(runner.job_queue, "is_cancelled", lambda job_id: True)
    answered, cancelled = runner.run_phased(
        "run", ["q1", "q2"], use_rerank=False, language=None, k=2, model=None, job_id=7
    )
    assert (answered, cancelled) == (0, True)
    assert not [c for c in calls if c[0] == "generate"]


def _scored_rows(marker, scores):
    return [
        (f"chunk {marker} {i}", f"{marker}.md", "cat", i, 1, None, 0.1, s)
        for i, s in enumerate(scores)
    ]


def _rerank_by_content(monkeypatch, ranking):
    # ranking: content substring -> score; a real reranker scores the (question, chunk) pair
    def fake_predict(pairs):
        return [ranking.get(chunk, 0.0) for _, chunk in pairs]

    monkeypatch.setattr(runner.rerank, "_model", lambda: type("M", (), {"predict": staticmethod(fake_predict)})())
    monkeypatch.setattr(runner.rerank, "_predict", fake_predict)


def test_rerank_phase_keeps_candidates_with_their_own_question(monkeypatch):
    # the classic batching bug: chunks leaking between questions
    retrieved = [("q1", _scored_rows("a", [0.1, 0.2])), ("q2", _scored_rows("b", [0.3, 0.4]))]
    _rerank_by_content(monkeypatch, {"chunk a 0": 9, "chunk a 1": 1, "chunk b 0": 8, "chunk b 1": 2})

    out = runner._phase_rerank(retrieved, k=1)

    assert [text for text, _ in out] == ["q1", "q2"]
    assert [rows[0][0] for _, rows in out] == ["chunk a 0", "chunk b 0"]


def test_rerank_phase_orders_within_each_question(monkeypatch):
    retrieved = [("q", _scored_rows("a", [0.1, 0.2, 0.3]))]
    _rerank_by_content(monkeypatch, {"chunk a 0": 1, "chunk a 1": 5, "chunk a 2": 3})

    (_, rows), = runner._phase_rerank(retrieved, k=3)

    assert [r[0] for r in rows] == ["chunk a 1", "chunk a 2", "chunk a 0"]


def test_rerank_phase_handles_ragged_and_empty_pools(monkeypatch):
    retrieved = [
        ("q1", _scored_rows("a", [0.1])),
        ("q2", []),
        ("q3", _scored_rows("c", [0.1, 0.2, 0.3])),
    ]
    _rerank_by_content(monkeypatch, {"chunk a 0": 1, "chunk c 0": 1, "chunk c 1": 9, "chunk c 2": 5})

    out = runner._phase_rerank(retrieved, k=2)

    assert [len(rows) for _, rows in out] == [1, 0, 2]
    assert [r[0] for r in out[2][1]] == ["chunk c 1", "chunk c 2"]
