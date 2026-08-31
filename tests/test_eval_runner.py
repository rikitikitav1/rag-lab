from evals import runner


def _rows(marker, n=3):
    return [(f"chunk {marker} {i}", f"{marker}.md", "cat", i, 1, None, 0.1, 0.5) for i in range(n)]


def _stub_phases(monkeypatch, use_rerank_expected=None):
    from use_cases import search_depth

    calls = []
    # the phase resolves the depth once and carries it into every snapshot; asking the
    # planner needs a database, and these tests need none
    monkeypatch.setattr(search_depth, "resolve", lambda *a, **kw: 200)
    monkeypatch.setattr(
        runner.llm, "request_embeddings_batch", lambda texts: [[0.1]] * len(texts)
    )
    monkeypatch.setattr(
        runner.db, "hybrid_search",
        lambda text, vector, category, limit, variant, ef_search=None: (
            calls.append(("search", text, limit, variant, ef_search)) or _rows(text)
        ),
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
    # the card check is a network call; the tests that care about it override these
    monkeypatch.setattr(runner.llm, "warn_if_models_do_not_fit", lambda: [])
    monkeypatch.setattr(runner.llm, "models_off_the_card", lambda: [])
    monkeypatch.setattr(
        runner.chat, "answer_from_rows",
        lambda text, rows, **kw: calls.append(("generate", text, kw.get("phased"))),
    )
    return calls


def test_phases_run_in_order_and_free_vram(monkeypatch):
    calls = _stub_phases(monkeypatch)
    answered, cancelled = runner.run_phased(
        "run", ["q1", "q2"], use_rerank=True, language=None, k=2, model=None, job_id=None, variant="baseline"
    )
    assert (answered, cancelled) == (2, False)
    kinds = [c[0] for c in calls]
    assert kinds == [
        "search", "search",
        "unload", "unload",
        "rerank",
        "unload",
        "generate", "generate",
        # teardown, on every exit: reranker, embedder, generator
        "unload", "unload", "unload",
    ]
    assert [c[1] for c in calls[-3:]] == ["reranker", "embedding", "generation"]


def test_rerank_runs_once_for_the_whole_set(monkeypatch):
    calls = _stub_phases(monkeypatch)
    runner.run_phased(
        "run", ["q1", "q2", "q3"], use_rerank=True, language=None, k=2, model=None, job_id=None, variant="baseline"
    )
    reranks = [c for c in calls if c[0] == "rerank"]
    assert len(reranks) == 1
    assert reranks[0][1] == 9


def test_retrieval_widens_only_when_reranking(monkeypatch):
    calls = _stub_phases(monkeypatch)
    runner.run_phased("run", ["q"], use_rerank=True, language=None, k=3, model=None, job_id=None, variant="baseline")
    wide = [c[2] for c in calls if c[0] == "search"][0]

    calls.clear()
    runner.run_phased("run", ["q"], use_rerank=False, language=None, k=3, model=None, job_id=None, variant="baseline")
    narrow = [c[2] for c in calls if c[0] == "search"][0]

    assert wide == runner.config.settings.rerank.candidates
    assert narrow == 3
    assert not [c for c in calls if c[0] == "rerank"]
    # a run without a rerank phase still frees the embedder before generating, and gives
    # the whole card back when it is done, so the next run starts on an empty one
    assert [c[1] for c in calls if c[0] == "unload"] == [
        "embedding", "reranker", "embedding", "generation"
    ]


def test_generation_marks_logs_as_phased(monkeypatch):
    calls = _stub_phases(monkeypatch)
    runner.run_phased("run", ["q"], use_rerank=False, language=None, k=2, model=None, job_id=None, variant="baseline")
    assert [c[2] for c in calls if c[0] == "generate"] == [True]


def test_cancel_stops_generation_midway(monkeypatch):
    calls = _stub_phases(monkeypatch)
    monkeypatch.setattr(runner.job_queue, "is_cancelled", lambda job_id: True)
    answered, cancelled = runner.run_phased(
        "run", ["q1", "q2"], use_rerank=False, language=None, k=2, model=None, job_id=7, variant="baseline"
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
        "run", ["q"], use_rerank=True, language=None, k=2, model="hf.co/some/model:Q4", job_id=None, variant="baseline"
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

    out, depth = runner._phase_retrieve(["ok1", "ok2", "boom", "ok3"], k=3, use_rerank=False, variant="baseline")

    assert depth == 200
    assert [text for text, _, _ in out] == ["ok1", "ok2"]
    assert len([c for c in calls if c[0] == "search"]) == 2


def test_search_failure_skips_one_question(monkeypatch):
    _stub_phases(monkeypatch)

    def flaky(text, vector, category, limit, variant, ef_search=None):
        if text == "bad":
            raise RuntimeError("pg down")
        return _rows(text)

    monkeypatch.setattr(runner.db, "hybrid_search", flaky)

    out, _depth = runner._phase_retrieve(["good", "bad"], k=3, use_rerank=False, variant="baseline")
    assert [text for text, _, _ in out] == ["good"]


def test_cancel_between_phases_skips_rerank_and_generation(monkeypatch):
    calls = _stub_phases(monkeypatch)
    monkeypatch.setattr(runner.job_queue, "is_cancelled", lambda job_id: True)

    answered, cancelled = runner.run_phased(
        "run", ["q1", "q2"], use_rerank=True, language=None, k=2, model=None, job_id=7, variant="baseline"
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

    # a run measuring the processor says so: the card guard now sees the cross-encoder
    # itself, and a reranker on the cpu is exactly what it refuses
    runner.run_phased(
        "run", ["q1", "q2"], use_rerank=True, language=None, k=2, model=None, job_id=None,
        variant="baseline", allow_cpu=True,
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
    monkeypatch.setattr(runner.db, "corpus_variants", lambda: [{"variant": "baseline"}])
    monkeypatch.setattr(runner.db, "is_empty", lambda *, variant: False)
    monkeypatch.setattr(runner.search_depth, "resolve", lambda *a, **kw: 100)
    monkeypatch.setattr(runner, "_walks_the_index", lambda variant, depth: True)
    monkeypatch.setattr(runner.job_queue, "enqueue", lambda *a, **kw: None)
    monkeypatch.setattr(runner.agent, "run", lambda text, **kw: seen.append(kw["fallback_policy"]))

    runner.run("run", set_name="s", pipeline="agent", fallback_policy="agent_choice")

    assert seen == ["agent_choice"]


def test_a_phased_run_records_the_depth_it_searched_at(monkeypatch):
    # phased is the default for single-shot, and it used to record `ef_search: null`,
    # so the depth of every generation run on this branch was missing from its own record
    _stub_phases(monkeypatch)
    snap = {}
    monkeypatch.setattr(
        runner.chat, "answer_from_rows",
        lambda *a, **kw: snap.update(ef_search=kw.get("ef_search")),
    )
    runner.run_phased("r", ["q1"], use_rerank=False, language=None, k=3,
                      model=None, job_id=None, variant="baseline")
    assert snap["ef_search"] == 200


def test_the_card_is_asked_about_before_the_generator_is_paid_for(monkeypatch):
    # the check used to fire only after the first answer, which is after the generator has
    # been loaded onto the processor. The embedder is enough to see the card is gone
    import pytest

    calls = _stub_phases(monkeypatch)
    monkeypatch.setattr(runner.llm, "models_off_the_card", lambda: ["bge-m3"])
    with pytest.raises(RuntimeError, match="not on the GPU"):
        runner.run_phased(
            "run", ["q1", "q2"], use_rerank=True, language=None, k=2, model=None,
            job_id=None, variant="baseline",
        )
    assert [c[0] for c in calls] == ["search", "search", "unload", "unload", "unload"], (
        "nothing after retrieval should have run, and the card goes back anyway"
    )
    # the refusal used to raise past the unload, so the run that found the card full left
    # its own generator on it and every retry refused for the same reason
    assert [c[1] for c in calls[-3:]] == ["reranker", "embedding", "generation"]


def test_a_phased_run_refuses_a_card_that_dropped_out(monkeypatch):
    # the sequential path refused from the start and the phased path, which is the
    # default for single_shot, did not. ollama kept answering off the card at four
    # times the cost, and nothing in the run said so
    import pytest

    _stub_phases(monkeypatch)
    monkeypatch.setattr(runner.llm, "models_off_the_card", lambda: ["llama3.1:8b"])
    with pytest.raises(RuntimeError, match="not on the GPU"):
        runner.run_phased(
            "run", ["q1", "q2"], use_rerank=False, language=None, k=2, model=None,
            job_id=None, variant="baseline",
        )


def test_a_phased_run_measuring_the_cpu_says_so_and_proceeds(monkeypatch):
    calls = _stub_phases(monkeypatch)
    monkeypatch.setattr(runner.llm, "models_off_the_card", lambda: ["llama3.1:8b"])
    answered, cancelled = runner.run_phased(
        "run", ["q1", "q2"], use_rerank=False, language=None, k=2, model=None,
        job_id=None, variant="baseline", allow_cpu=True,
    )
    assert (answered, cancelled) == (2, False)
    assert [c[0] for c in calls].count("generate") == 2


def test_a_rerank_that_throws_still_gives_the_card_back(monkeypatch):
    # the third uncovered exit: a failure in the rerank phase skipped `rerank.unload()`
    # and the final unload, so the reranker and the generator both stayed resident
    import pytest

    calls = _stub_phases(monkeypatch)

    def _boom(pairs):
        raise RuntimeError("cuda is unhappy")

    monkeypatch.setattr(runner.rerank, "score_pairs", _boom)
    with pytest.raises(RuntimeError, match="cuda is unhappy"):
        runner.run_phased(
            "run", ["q1"], use_rerank=True, language=None, k=2, model=None,
            job_id=None, variant="baseline",
        )
    assert [c[1] for c in calls[-3:]] == ["reranker", "embedding", "generation"]


def test_the_sequential_path_gives_the_card_back_too(monkeypatch):
    # a sweep over `model` on the agent pipeline runs several of these back to back
    calls = []
    monkeypatch.setattr(
        runner.llm, "unload",
        lambda role="embedding", model=None: calls.append(("unload", role)),
    )
    monkeypatch.setattr(runner.rerank, "unload", lambda: calls.append(("unload", "reranker")))
    monkeypatch.setattr(runner, "_answer_one", lambda *a, **kw: calls.append(("answer",)))
    monkeypatch.setattr(runner, "_refuse_a_cpu_run", lambda allow_cpu, use_rerank=False: None)
    answered, cancelled = runner._run_sequential(
        ["q1", "q2"], "run", None, runner.Pipeline.agent, None, None, None, "llama3.1:8b",
        None, None, None, None, None, None, False, "baseline",
    )
    assert (answered, cancelled) == (2, False)
    assert [c[1] for c in calls if c[0] == "unload"] == [
        "reranker", "embedding", "generation"
    ]


def test_a_run_refuses_when_its_depth_stopped_walking_the_index(monkeypatch):
    # the preflight is a snapshot taken before the queue moved: the crossover shifts when a
    # neighbouring variant is indexed, and the planner then sorts while the record says hnsw
    import pytest

    monkeypatch.setattr(runner, "_target_texts", lambda set_name, ids: ["q1"])
    monkeypatch.setattr(runner.db, "corpus_variants", lambda: [{"variant": "baseline"}])
    monkeypatch.setattr(runner.db, "is_empty", lambda *, variant: False)
    monkeypatch.setattr(runner.search_depth, "resolve", lambda *a, **kw: 100)
    monkeypatch.setattr(runner, "_walks_the_index", lambda variant, depth: False)

    with pytest.raises(RuntimeError, match="no longer walks its index"):
        runner.run("run", set_name="s", pipeline="agent")


def test_the_card_guard_sees_the_cross_encoder_before_the_set_is_reranked(monkeypatch):
    # it used to be asked between retrieval and reranking, where the model is still None,
    # so it first fired after the whole set had been reranked on the processor
    import pytest

    warmed = []
    _stub_phases(monkeypatch)
    monkeypatch.setattr(runner.rerank, "warm", lambda: warmed.append(True))
    monkeypatch.setattr(runner.rerank, "off_the_card", lambda: "reranker on cpu")

    with pytest.raises(RuntimeError, match="not on the GPU"):
        runner.run_phased(
            "run", ["q1"], use_rerank=True, language=None, k=2, model=None, job_id=None,
            variant="baseline",
        )
    assert warmed, "the model is loaded before it is asked where it sits"
