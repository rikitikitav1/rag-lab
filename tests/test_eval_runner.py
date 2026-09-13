from pathlib import Path

import pytest
from evals import runner
from use_cases.run_snapshot import ANSWERING


def _spec(**kw):
    kw.setdefault("variant", "baseline")
    return runner.RunSpec(**kw)


def _rows(marker, n=3):
    from db import Hit

    return [
        Hit(f"chunk {marker} {i}", f"{marker}.md", "cat", i, 1, None, 0.1, 0.5, None)
        for i in range(n)
    ]


def _stub_phases(monkeypatch, use_rerank_expected=None):
    from conftest import stub_engines
    from use_cases import search_depth

    # the card gate now asks the generator's own engine, and resolving one needs a database
    stub_engines(monkeypatch, runner)
    calls = []
    # the phase resolves the depth once and carries it; asking the planner needs a database
    monkeypatch.setattr(search_depth, "resolve", lambda *a, **kw: 200)
    monkeypatch.setattr(
        runner.llm, "request_embeddings_batch", lambda texts: [[0.1]] * len(texts)
    )
    monkeypatch.setattr(runner.llm, "embedder_label", lambda role="embedding": "bge-m3@ollama")
    monkeypatch.setattr(
        runner.llm, "embed_labelled", lambda texts: ("bge-m3@ollama", [[0.1]] * len(texts))
    )
    monkeypatch.setattr(
        runner.db, "hybrid_search",
        lambda text, vector, category, limit, variant, ef_search=None, embedded_by=None: (
            calls.append(("search", text, limit, variant, ef_search)) or _rows(text)
        ),
    )
    monkeypatch.setattr(
        runner.rerank, "score_pairs",
        lambda pairs: calls.append(("rerank", len(pairs))) or [1.0] * len(pairs),
    )
    monkeypatch.setattr(
        runner, "_release",
        lambda role="embedding", model=None: calls.append(("unload", role)),
    )
    # the card check is a network call; the tests that care about it override these
    monkeypatch.setattr(runner.card, "model_on_card", lambda spec, name: True)
    monkeypatch.setattr(
        runner.chat, "answer_from_rows",
        lambda text, rows, **kw: calls.append(("generate", text, kw.get("phased"))),
    )
    return calls


def test_phases_run_in_order_and_free_vram(monkeypatch):
    calls = _stub_phases(monkeypatch)
    answered, cancelled = runner.run_phased(["q1", "q2"], "run", _spec(use_rerank=True, k=2))
    assert (answered, cancelled) == (2, False)
    kinds = [c[0] for c in calls]
    assert kinds == [
        "search", "search",
        "unload", "unload",
        # no unload after it: the cross-encoder sleeps in its own server when the card is handed on
        "rerank",
        "generate", "generate",
        # teardown, on every exit: embedder, generator; the reranker lives in its own server
        "unload", "unload",
    ]
    assert [c[1] for c in calls[-2:]] == ["embedding", "generation"]


def test_rerank_runs_once_for_the_whole_set(monkeypatch):
    calls = _stub_phases(monkeypatch)
    runner.run_phased(["q1", "q2", "q3"], "run", _spec(use_rerank=True, k=2))
    reranks = [c for c in calls if c[0] == "rerank"]
    assert len(reranks) == 1
    assert reranks[0][1] == 9


def test_retrieval_widens_only_when_reranking(monkeypatch):
    calls = _stub_phases(monkeypatch)
    runner.run_phased(["q"], "run", _spec(use_rerank=True, k=3))
    wide = [c[2] for c in calls if c[0] == "search"][0]

    calls.clear()
    runner.run_phased(["q"], "run", _spec(use_rerank=False, k=3))
    narrow = [c[2] for c in calls if c[0] == "search"][0]

    assert wide == runner.config.settings.rerank.candidates
    assert narrow == 3
    assert not [c for c in calls if c[0] == "rerank"]
    # a run without a rerank phase still frees the embedder and gives the card back
    assert [c[1] for c in calls if c[0] == "unload"] == [
        "embedding", "embedding", "generation"
    ]


def test_generation_marks_logs_as_phased(monkeypatch):
    calls = _stub_phases(monkeypatch)
    runner.run_phased(["q"], "run", _spec(use_rerank=False, k=2))
    assert [c[2] for c in calls if c[0] == "generate"] == [True]


def test_cancel_stops_generation_midway(monkeypatch):
    calls = _stub_phases(monkeypatch)
    monkeypatch.setattr(runner.job_queue, "is_cancelled", lambda job_id: True)
    answered, cancelled = runner.run_phased(["q1", "q2"], "run", _spec(use_rerank=False, k=2), job_id=7)
    assert (answered, cancelled) == (0, True)
    assert not [c for c in calls if c[0] == "generate"]


def _scored_rows(marker, scores):
    from db import Hit

    return [
        Hit(f"chunk {marker} {i}", f"{marker}.md", "cat", i, 1, None, 0.1, s, None)
        for i, s in enumerate(scores)
    ]


def _rerank_by_content(monkeypatch, ranking):
    def fake_predict(pairs):
        return [ranking.get(chunk, 0.0) for _, chunk in pairs]

    monkeypatch.setattr(runner.rerank, "score_pairs", fake_predict)


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
        runner, "_release",
        lambda role="embedding", model=None: unloaded.append((role, model)),
    )
    runner.run_phased(["q"], "run", _spec(use_rerank=True, k=2, model="hf.co/some/model:Q4"))
    assert ("generation", "hf.co/some/model:Q4") in unloaded
    assert calls


def test_embedding_failure_drops_only_its_batch(monkeypatch):
    calls = _stub_phases(monkeypatch)
    monkeypatch.setattr(runner.config.settings.ingestion, "batch_size", 2)

    def flaky(chunk):
        if "boom" in chunk:
            raise RuntimeError("embedder down")
        return "bge-m3@ollama", [[0.1]] * len(chunk)

    monkeypatch.setattr(runner.llm, "embed_labelled", flaky)

    out, depth = runner._phase_retrieve(["ok1", "ok2", "boom", "ok3"], _spec(use_rerank=False, k=3))

    assert depth == 200
    assert [text for text, _, _ in out] == ["ok1", "ok2"]
    assert len([c for c in calls if c[0] == "search"]) == 2


def test_search_failure_skips_one_question(monkeypatch):
    _stub_phases(monkeypatch)

    def flaky(text, vector, category, limit, variant, ef_search=None, embedded_by=None):
        if text == "bad":
            raise RuntimeError("pg down")
        return _rows(text)

    monkeypatch.setattr(runner.db, "hybrid_search", flaky)

    out, _depth = runner._phase_retrieve(["good", "bad"], _spec(use_rerank=False, k=3))
    assert [text for text, _, _ in out] == ["good"]


def test_cancel_between_phases_skips_rerank_and_generation(monkeypatch):
    calls = _stub_phases(monkeypatch)
    monkeypatch.setattr(runner.job_queue, "is_cancelled", lambda job_id: True)

    answered, cancelled = runner.run_phased(["q1", "q2"], "run", _spec(use_rerank=True, k=2), job_id=7)

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

    # a run measuring the processor says so: the guard now sees the cross-encoder itself
    runner.run_phased(["q1", "q2"], "run", _spec(use_rerank=True, k=2), allow_cpu=True)

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
    # phased is the default for single-shot and it recorded `ef_search: null`
    _stub_phases(monkeypatch)
    snap = {}
    monkeypatch.setattr(
        runner.chat, "answer_from_rows",
        lambda *a, **kw: snap.update(ef_search=kw.get("ef_search")),
    )
    runner.run_phased(["q1"], "r", _spec(use_rerank=False, k=3))
    assert snap["ef_search"] == 200


def test_the_card_is_asked_about_before_the_generator_is_paid_for(monkeypatch):
    # the check fired only after the first answer, with the generator already on the cpu
    import pytest

    calls = _stub_phases(monkeypatch)
    monkeypatch.setattr(runner.card, "model_on_card", lambda spec, name: False)
    with pytest.raises(RuntimeError, match="not on the GPU"):
        runner.run_phased(["q1", "q2"], "run", _spec(use_rerank=True, k=2))
    assert [c[0] for c in calls] == ["search", "search", "unload", "unload"], (
        "nothing after retrieval should have run, and the card goes back anyway"
    )
    # the refusal raised past the unload, so the run left its own generator on the card
    assert [c[1] for c in calls[-2:]] == ["embedding", "generation"]


def test_a_phased_run_refuses_a_card_that_dropped_out(monkeypatch):
    # the sequential path refused from the start and the phased default did not
    import pytest

    _stub_phases(monkeypatch)
    # half on the processor counts as off: the one instrument the preflight reads
    monkeypatch.setattr(runner.card, "model_on_card", lambda spec, name: False)
    with pytest.raises(RuntimeError, match="not on the GPU"):
        runner.run_phased(["q1", "q2"], "run", _spec(use_rerank=False, k=2))


def test_a_phased_run_measuring_the_cpu_says_so_and_proceeds(monkeypatch):
    calls = _stub_phases(monkeypatch)
    monkeypatch.setattr(runner.card, "model_on_card", lambda spec, name: False)
    answered, cancelled = runner.run_phased(["q1", "q2"], "run", _spec(use_rerank=False, k=2), allow_cpu=True)
    assert (answered, cancelled) == (2, False)
    assert [c[0] for c in calls].count("generate") == 2


def test_a_rerank_that_throws_still_gives_the_card_back(monkeypatch):
    # a failure in the rerank phase skipped both unloads
    import pytest

    calls = _stub_phases(monkeypatch)

    def _boom(pairs):
        raise RuntimeError("cuda is unhappy")

    monkeypatch.setattr(runner.rerank, "score_pairs", _boom)
    with pytest.raises(RuntimeError, match="cuda is unhappy"):
        runner.run_phased(["q1"], "run", _spec(use_rerank=True, k=2))
    assert [c[1] for c in calls[-2:]] == ["embedding", "generation"]


def test_the_sequential_path_gives_the_card_back_too(monkeypatch):
    # a sweep over `model` on the agent pipeline runs several of these back to back
    calls = []
    monkeypatch.setattr(
        runner, "_release",
        lambda role="embedding", model=None: calls.append(("unload", role)),
    )
    monkeypatch.setattr(runner, "_answer_one", lambda *a, **kw: calls.append(("answer",)))
    monkeypatch.setattr(runner, "_refuse_a_cpu_run", lambda roles, allow_cpu, model: None)
    answered, cancelled = runner._run_sequential(
        ["q1", "q2"],
        "run",
        runner.RunSpec(
            variant="baseline", pipeline=runner.Pipeline.agent, model="llama3.1:8b"
        ),
        job_id=None,
        allow_cpu=False,
    )
    assert (answered, cancelled) == (2, False)
    assert [c[1] for c in calls if c[0] == "unload"] == ["embedding", "generation"]


def test_a_run_refuses_when_its_depth_stopped_walking_the_index(monkeypatch):
    # the preflight answers from before the queue moved, and the crossover shifts
    import pytest

    monkeypatch.setattr(runner, "_target_texts", lambda set_name, ids: ["q1"])
    monkeypatch.setattr(runner.db, "corpus_variants", lambda: [{"variant": "baseline"}])
    monkeypatch.setattr(runner.db, "is_empty", lambda *, variant: False)
    monkeypatch.setattr(runner.search_depth, "resolve", lambda *a, **kw: 100)
    monkeypatch.setattr(runner, "_walks_the_index", lambda variant, depth: False)

    with pytest.raises(RuntimeError, match="no longer walks its index"):
        runner.run("run", set_name="s", pipeline="agent")


def test_the_answering_knobs_travel_as_one_value_not_as_a_row_of_positions():
    # fourteen positional arguments in one call and sixteen in another order in the next
    import inspect

    # every function of the module but the door itself, so the next one added is covered too
    named = [
        fn
        for name, fn in vars(runner).items()
        if inspect.isfunction(fn) and fn.__module__ == runner.__name__ and name != "run"
    ]
    assert len(named) >= 10, "the module lost its functions, and this guard now checks nothing"
    for fn in named:
        positional = [
            p
            for p in inspect.signature(fn).parameters.values()
            if p.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
        ]
        assert len(positional) <= 3, f"{fn.__name__} takes {len(positional)} by position"


def test_the_run_gate_refuses_a_spill_and_passes_an_asleep_vllm(monkeypatch):
    # the embedder took the card for retrieval, and the asleep generator read as off
    import engines
    from models.registry import EngineKind, Placement, Role
    from use_cases import run_snapshot

    vllm = engines.EngineSpec(3, "vllm", EngineKind.vllm, "VLLM", Placement.gpu)
    gpu = engines.EngineSpec(1, "ollama", EngineKind.ollama, "OLLAMA", Placement.gpu)
    cpu = engines.EngineSpec(5, "ollama-cpu", EngineKind.ollama, "OLLAMA_CPU", Placement.cpu)
    specs = {"generation": vllm, "embedding": gpu}
    monkeypatch.setattr(run_snapshot.llm, "resolve",
                        lambda role: engines.Resolved(f"{role}-model", specs[role]))
    monkeypatch.setattr(runner.card, "model_on_card", lambda spec, name: False)
    runner._refuse_a_cpu_run((Role.generation,), allow_cpu=False, model=None)
    with pytest.raises(RuntimeError, match="embedding=embedding-model"):
        runner._refuse_a_cpu_run(ANSWERING, allow_cpu=False, model=None)
    specs["embedding"] = cpu
    runner._refuse_a_cpu_run(ANSWERING, allow_cpu=False, model=None)


def test_the_run_gate_reads_the_arm_s_own_generator_not_the_role_s(monkeypatch):
    # an arm's `model` may spill while the role's default sits on the card
    import engines
    from models.registry import EngineKind, Placement, Role
    from use_cases import run_snapshot

    gpu = engines.EngineSpec(1, "ollama", EngineKind.ollama, "OLLAMA", Placement.gpu)
    monkeypatch.setattr(run_snapshot.llm, "resolve",
                        lambda role: engines.Resolved("llama3.1:8b", gpu))
    monkeypatch.setattr(run_snapshot.llm.engines, "find_model",
                        lambda name, engine_id=None: engines.Resolved(name, gpu))
    monkeypatch.setattr(runner.card, "model_on_card", lambda spec, name: name == "llama3.1:8b")
    runner._refuse_a_cpu_run((Role.generation,), allow_cpu=False, model=None)
    with pytest.raises(RuntimeError, match="generation=qwen2.5:32b"):
        runner._refuse_a_cpu_run((Role.generation,), allow_cpu=False, model="qwen2.5:32b")


def test_a_stand_fault_ends_the_run_instead_of_one_row(monkeypatch):
    # a lost card or foreign vectors turned into a run `done` with part of its answers
    from engines import card

    import db

    calls = _stub_phases(monkeypatch)

    def foreign(*a, **kw):
        raise db.ForeignVectors("variant holds vectors of another embedder")

    monkeypatch.setattr(runner.db, "hybrid_search", foreign)
    with pytest.raises(db.ForeignVectors):
        runner.run_phased(["q1", "q2"], "run", _spec(use_rerank=False, k=2))
    assert not [c for c in calls if c[0] == "generate"]

    def lost(texts):
        raise card.CardNotHanded("vllm is down; the card stays where it is")

    _stub_phases(monkeypatch)
    monkeypatch.setattr(runner.llm, "embed_labelled", lost)
    with pytest.raises(card.CardNotHanded):
        runner.run_phased(["q1"], "run", _spec(use_rerank=False, k=2))


# a try here reads or writes the stand and never calls a model or a search, so no fault reaches it
_FORGIVES_NO_CALL = {
    ("evals/runner.py", "_release"),
    ("job_handlers/judging.py", "_count_the_attempt"),
    ("job_handlers/judging.py", "_merge_guest_scores"),
    ("job_handlers/judging.py", "_residency"), ("job_handlers/judging.py", "_merge_our_scores"),
    ("agent_tools.py", "remote_tools"), ("orchestrators/graph.py", "versions"),
    ("db.py", "fingerprint_or_none"), ("job_handlers/indexing.py", "index_data"),
}
# read off the source, not listed: a module that starts calling a model later is guarded from then on
def _reaches_a_model(app: Path) -> list[str]:
    calls = ("llm.chat(", "llm.ask(", "llm.embed(", "request_embeddings_batch(", "score_pairs(",
             "embed_labelled(", "embed_with_label(",
             "hybrid_search(", "nearest_distance(", "dispatch(", ".invoke(", "guest_axes.score(")
    return sorted(str(p.relative_to(app)) for p in app.rglob("*.py")
                  if any(c in p.read_text() for c in calls))


def _broad_catches(path: Path):
    import ast

    def names(h):
        kinds = h.type.elts if isinstance(h.type, ast.Tuple) else [h.type]
        return {ast.unparse(k) for k in kinds if k is not None}

    def broad(h):
        return h.type is None or bool(names(h) & {"Exception", "BaseException"})

    def reraises(h):
        return isinstance(h.body[-1], ast.Raise) and h.body[-1].exc is None

    def visit(node, fn):
        for child in ast.iter_child_nodes(node):
            name = getattr(child, "name", fn) if isinstance(
                child, (ast.FunctionDef, ast.AsyncFunctionDef)) else fn
            if isinstance(child, ast.Try):
                for i, h in enumerate(child.handlers):
                    if not broad(h):
                        continue
                    # a `StandFault` handler that logs and returns forgives it as surely as a broad one
                    passed = any(any("StandFault" in n for n in names(e)) and reraises(e)
                                 for e in child.handlers[:i])
                    yield name, passed or reraises(h)
            yield from visit(child, name)

    yield from visit(ast.parse(path.read_text()), "<module>")


def test_the_guard_sees_a_fault_logged_away_and_a_broad_catch_in_a_tuple(tmp_path):
    # a `StandFault` handler must re-raise too, and a tuple with `Exception` is broad
    source = tmp_path / "m.py"
    source.write_text(
        "def swallowed():\n    try:\n        f()\n    except StandFault:\n        log()\n"
        "    except Exception:\n        pass\n\n"
        "def tupled():\n    try:\n        f()\n    except (ValueError, Exception):\n        pass\n\n"
        "def passed():\n    try:\n        f()\n    except StandFault:\n        raise\n"
        "    except Exception:\n        pass\n"
    )
    assert dict(_broad_catches(source)) == {"swallowed": False, "tupled": False, "passed": True}


def test_every_catch_that_forgives_a_row_lets_a_stand_fault_through():
    # a lost card must end the job, not fail row after row
    app = Path(__file__).resolve().parent.parent / "app"
    open_, seen = [], set()
    for rel in _reaches_a_model(app):
        for fn, safe in _broad_catches(app / rel):
            seen.add((rel, fn))
            if not safe and (rel, fn) not in _FORGIVES_NO_CALL:
                open_.append(f"{rel}:{fn}")
    assert open_ == []
    assert _FORGIVES_NO_CALL <= seen


def test_a_phased_run_reads_the_embedder_s_placement_before_it_lets_it_go(monkeypatch):
    # every row is written after the release, and read then the embedder was None
    calls = _stub_phases(monkeypatch)
    seen = []
    monkeypatch.setattr(runner.run_snapshot, "placed",
                        lambda role: calls.append(("placed", role)) or True)
    monkeypatch.setattr(runner.chat, "answer_from_rows",
                        lambda text, rows, **kw: seen.append(kw.get("placed_during")))
    runner.run_phased(["q1"], "run", _spec(use_rerank=True, k=2))
    kinds = [c if c[0] == "placed" else c[0] for c in calls]
    assert kinds.index(("placed", "embedding")) < kinds.index("unload"), "read before the release"
    assert seen == [{"embedding": True, "reranking": True}]


def test_a_resumed_run_asks_only_what_has_no_answer_and_replaces_every_error_row():
    from evals import runner

    rows = [
        (1, "a", {"outcome": "answered"}),
        (2, "b", {"outcome": "error", "failed": "hop cap"}),
        (3, "b", {"outcome": "error"}),
        (4, "d", {"outcome": "refused"}),
    ]
    todo, replaced = runner._split_answered(["a", "b", "c", "d"], rows)
    # a refusal is an answer; a question with no row at all is asked again, like an error
    assert todo == ["b", "c"]
    assert replaced == {"b": {"log_ids": [2, 3], "outcome": "error", "failed": "hop cap"}}


def test_a_run_over_named_questions_refuses_to_start_when_any_is_missing():
    # a floor on 50 rows that quietly became a floor on 48 is read as a floor on 50
    import pytest
    from errors import StandFault
    from evals import runner

    with pytest.raises(runner.MissingQuestions, match=r"1 of 3 question ids are not in the stand: \[2\]") as caught:
        runner._refuse_missing([1, 2, 3], {1, 3})
    assert isinstance(caught.value, StandFault), "eval_run stops a StandFault for good instead of retrying"
    runner._refuse_missing([1, 1, 3], {1, 3})
