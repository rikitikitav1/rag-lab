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
        runner.rerank, "score_pairs",
        lambda pairs: calls.append(("rerank", len(pairs))) or [1.0] * len(pairs),
    )
    monkeypatch.setattr(
        runner.llm, "unload",
        lambda role="embedding", model=None: calls.append(("unload", role)),
    )
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
        "rerank",
        "unload",
        "generate", "generate",
    ]


def test_rerank_runs_once_for_the_whole_set(monkeypatch):
    calls = _stub_phases(monkeypatch)
    runner.run_phased(
        "run", ["q1", "q2", "q3"], use_rerank=True, language=None, k=2, model=None, job_id=None
    )
    reranks = [c for c in calls if c[0] == "rerank"]
    assert len(reranks) == 1
    assert reranks[0][1] == 9


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
    def fake_predict(pairs):
        return [ranking.get(chunk, 0.0) for _, chunk in pairs]

    monkeypatch.setattr(runner.rerank, "_model", lambda: type("M", (), {"predict": staticmethod(fake_predict)})())
    monkeypatch.setattr(runner.rerank, "_predict", fake_predict)


def test_rerank_phase_keeps_candidates_with_their_own_question(monkeypatch):
    retrieved = [("q1", _scored_rows("a", [0.1, 0.2]), None), ("q2", _scored_rows("b", [0.3, 0.4]), None)]
    _rerank_by_content(monkeypatch, {"chunk a 0": 9, "chunk a 1": 1, "chunk b 0": 8, "chunk b 1": 2})

    out = runner._phase_rerank(retrieved, k=1)

    assert [text for text, _, _ in out] == ["q1", "q2"]
    assert [rows[0][0] for _, rows, _ in out] == ["chunk a 0", "chunk b 0"]


def test_rerank_phase_orders_within_each_question(monkeypatch):
    retrieved = [("q", _scored_rows("a", [0.1, 0.2, 0.3]), None)]
    _rerank_by_content(monkeypatch, {"chunk a 0": 1, "chunk a 1": 5, "chunk a 2": 3})

    (_, rows, _), = runner._phase_rerank(retrieved, k=3)

    assert [r[0] for r in rows] == ["chunk a 1", "chunk a 2", "chunk a 0"]


def test_rerank_phase_handles_ragged_and_empty_pools(monkeypatch):
    retrieved = [
        ("q1", _scored_rows("a", [0.1]), None),
        ("q2", [], None),
        ("q3", _scored_rows("c", [0.1, 0.2, 0.3]), None),
    ]
    _rerank_by_content(monkeypatch, {"chunk a 0": 1, "chunk c 0": 1, "chunk c 1": 9, "chunk c 2": 5})

    out = runner._phase_rerank(retrieved, k=2)

    assert [len(rows) for _, rows, _ in out] == [1, 0, 2]
    assert [r[0] for r in out[2][1]] == ["chunk c 1", "chunk c 2"]


def test_unload_targets_the_overridden_generator(monkeypatch):
    calls = _stub_phases(monkeypatch)
    unloaded = []
    monkeypatch.setattr(
        runner.llm, "unload",
        lambda role="embedding", model=None: unloaded.append((role, model)),
    )
    runner.run_phased(
        "run", ["q"], use_rerank=True, language=None, k=2, model="hf.co/some/model:Q4", job_id=None
    )
    assert ("generation", "hf.co/some/model:Q4") in unloaded
    assert calls


def test_embedding_failure_drops_only_its_batch(monkeypatch):
    calls = _stub_phases(monkeypatch)
    monkeypatch.setattr(runner.config.settings.ingestion, "batch_size", 2)

    def flaky(chunk):
        if "boom" in chunk:
            raise RuntimeError("embedder down")
        return [[0.1]] * len(chunk)

    monkeypatch.setattr(runner.llm, "request_embeddings_batch", flaky)

    out = runner._phase_retrieve(["ok1", "ok2", "boom", "ok3"], k=3, use_rerank=False)

    assert [text for text, _, _ in out] == ["ok1", "ok2"]
    assert len([c for c in calls if c[0] == "search"]) == 2


def test_search_failure_skips_one_question(monkeypatch):
    _stub_phases(monkeypatch)

    def flaky(text, vector, category, limit):
        if text == "bad":
            raise RuntimeError("pg down")
        return _rows(text)

    monkeypatch.setattr(runner.db, "hybrid_search", flaky)

    out = runner._phase_retrieve(["good", "bad"], k=3, use_rerank=False)
    assert [text for text, _, _ in out] == ["good"]


def test_cancel_between_phases_skips_rerank_and_generation(monkeypatch):
    calls = _stub_phases(monkeypatch)
    monkeypatch.setattr(runner.job_queue, "is_cancelled", lambda job_id: True)

    answered, cancelled = runner.run_phased(
        "run", ["q1", "q2"], use_rerank=True, language=None, k=2, model=None, job_id=7
    )

    assert (answered, cancelled) == (0, True)
    assert not [c for c in calls if c[0] in ("rerank", "generate")]


def test_phased_snapshot_keeps_the_device_used_during_rerank(monkeypatch):
    _stub_phases(monkeypatch)
    logged = []
    monkeypatch.setattr(
        runner.chat, "answer_from_rows",
        lambda text, rows, **kw: logged.append(kw.get("rerank_device")),
    )
    monkeypatch.setattr(runner.rerank, "device", lambda: "cpu")

    runner.run_phased(
        "run", ["q1", "q2"], use_rerank=True, language=None, k=2, model=None, job_id=None
    )

    assert logged == ["cpu", "cpu"]


def test_rerank_phase_accepts_numpy_scores(monkeypatch):
    import numpy as np

    retrieved = [("q1", _scored_rows("a", [0.1, 0.2]), None), ("q2", _scored_rows("b", [0.3]), None)]
    monkeypatch.setattr(
        runner.rerank, "score_pairs",
        lambda pairs: np.asarray([1.0, 9.0, 5.0], dtype=np.float32),
    )

    out = runner._phase_rerank(retrieved, k=1)

    assert [rows[0][0] for _, rows, _ in out] == ["chunk a 1", "chunk b 0"]


def test_agent_runs_get_the_fallback_policy(monkeypatch):
    seen = []
    monkeypatch.setattr(runner, "_target_texts", lambda set_name, ids: ["q1"])
    monkeypatch.setattr(runner.job_queue, "enqueue", lambda *a, **kw: None)
    monkeypatch.setattr(runner.agent, "run", lambda text, **kw: seen.append(kw["fallback_policy"]))

    runner.run("run", set_name="s", pipeline="agent", fallback_policy="agent_choice")

    assert seen == ["agent_choice"]
