import time
from types import SimpleNamespace

import pytest
from use_cases import chat


def _row(src, vector_rank=1, keyword_rank=None, vector_distance=0.1, score=0.5):
    return ("content", src, "cat", 0, vector_rank, keyword_rank, vector_distance, score)



def test_take_sources_dedups():
    rows = [_row("a.md"), _row("a.md"), _row("b.md")]
    assert [s.source for s in chat.take_sources(rows)] == ["a.md", "b.md"]


def test_take_sources_rounds_numbers():
    s = chat.take_sources([_row("a.md", vector_distance=0.123456, score=0.987654)])[0]
    assert s.vector_distance == 0.123
    assert s.score == 0.988


class _Resp:
    text = "answer text"
    prompt_tokens = 10
    completion_tokens = 5


def _stub_generation(monkeypatch):
    monkeypatch.setattr(chat.llm, "ask", lambda **kw: _Resp())
    monkeypatch.setattr(chat.llm, "resolve_name", lambda role: "stub-model")
    monkeypatch.setattr(chat.prompt_repo, "active_template", lambda purpose: "sys")
    logged = {}
    monkeypatch.setattr(chat, "_log_answer", lambda *a, **kw: logged.update(args=a, kwargs=kw))
    return logged


def test_answer_from_rows_skips_retrieval(monkeypatch):
    _stub_generation(monkeypatch)

    def boom(*a, **kw):
        raise AssertionError("retrieval must not run in answer_from_rows")

    monkeypatch.setattr(chat, "_retrieve_rows", boom)
    ans = chat.answer_from_rows("q", [_row("a.md")], k=5, variant="baseline")
    assert ans.success is True
    assert ans.text == "answer text"
    assert [s.source for s in ans.sources] == ["a.md"]


def test_answer_from_rows_empty_is_a_refusal(monkeypatch):
    _stub_generation(monkeypatch)
    monkeypatch.setattr(chat.llm, "ask", lambda **kw: pytest.fail("no generation on empty rows"))
    ans = chat.answer_from_rows("q", [], k=5, variant="baseline")
    assert ans.success is False
    assert ans.text == chat.NO_RESULTS


def test_answer_from_rows_logs_phased_flag(monkeypatch):
    logged = _stub_generation(monkeypatch)
    chat.answer_from_rows("q", [_row("a.md")], k=5, phased=True, variant="baseline")
    phased_arg = logged["args"][7]
    assert phased_arg is True


def test_answer_from_rows_keeps_caller_start(monkeypatch):
    _stub_generation(monkeypatch)
    ans = chat.answer_from_rows("q", [_row("a.md")], k=5, started_at=time.perf_counter() - 3, variant="baseline")
    assert ans.elapsed >= 3


def test_config_snapshot_records_device_only_when_reranking(monkeypatch):
    monkeypatch.setattr(chat, "_rerank_device", lambda: "cuda")

    assert chat._config_snapshot(True, 5, False, 0.55, None, "baseline")["rerank_device"] == "cuda"
    assert chat._config_snapshot(False, 5, False, 0.55, None, "baseline")["rerank_device"] is None


def test_config_snapshot_carries_procedure_fields():
    snap = chat._config_snapshot(False, 7, True, 0.42, None, "baseline")
    assert (snap["k"], snap["phased"], snap["distance_threshold"]) == (7, True, 0.42)


def test_gate_scores_only_the_head_and_pads_the_rest(monkeypatch):
    import rerank

    seen = []
    monkeypatch.setattr(rerank, "score_pairs", lambda pairs: seen.append(pairs) or [0.9, 0.1])
    rows = [_row(f"{i}.md") for i in range(5)]

    scores = chat._gate_scores("q", rows, top=2)

    assert scores == [0.9, 0.1, None, None, None]
    assert seen == [[("q", "content"), ("q", "content")]]


def test_search_chunks_attaches_gate_scores_with_rerank_off(monkeypatch):
    rows = [_row("a.md"), _row("b.md")]
    monkeypatch.setattr(chat, "_retrieve_rows", lambda *a, **kw: (rows, None))
    monkeypatch.setattr(chat, "_gate_scores", lambda query, rows, top: [0.42, None])

    _, sources = chat.search_chunks("q", use_rerank=False, gate_top=1, variant="baseline")

    assert [s.rerank_score for s in sources] == [0.42, None]


def test_search_chunks_leaves_rerank_scores_alone(monkeypatch):
    rows = [_row("a.md")]
    monkeypatch.setattr(chat, "_retrieve_rows", lambda *a, **kw: (rows, [0.77]))
    monkeypatch.setattr(chat, "_gate_scores", lambda *a, **kw: pytest.fail("gate ran anyway"))

    _, sources = chat.search_chunks("q", use_rerank=True, gate_top=5, variant="baseline")

    assert [s.rerank_score for s in sources] == [0.77]


def test_dedup_keeps_the_best_cross_encoder_score():
    rows = [_row("a.md"), _row("b.md"), _row("a.md")]
    sources = chat.take_sources(rows, [0.1, 0.2, 0.9])
    by_path = {s.source: s.rerank_score for s in sources}
    assert by_path == {"a.md": 0.9, "b.md": 0.2}


def test_fts_language_comes_from_config(monkeypatch):
    import config

    import db

    monkeypatch.setattr(
        config.settings, "fts", SimpleNamespace(languages={"ru": "russian"}, fallback="simple")
    )
    assert db._ts_config("что такое хеш-таблица") == "russian"
    assert db._ts_config("...") == "simple"


def test_one_place_decides_whether_a_run_reranks(monkeypatch):
    # the handler passes the switch through untouched and the runner resolves it, so what
    # a run records is what it used
    import config
    from evals import runner
    from job_handlers import evaluation

    # True over a config that is already False: a resolver that ignores config passes
    # the old version of this test and fails this one
    monkeypatch.setattr(config.settings.rerank, "enabled", True)
    seen = {}
    monkeypatch.setattr(evaluation, "require_role_ready", lambda role: None)
    monkeypatch.setattr(evaluation.runner, "run", lambda **kw: seen.update(kw) or 0)

    evaluation.eval_run({"run_name": "r", "set_name": "s"})
    assert seen["use_rerank"] is None

    assert runner.resolve_rerank(None) is True, "unasked means whatever config says"
    assert runner.resolve_rerank(False) is False, "and a run may say otherwise"

    # and the interactive path reads the same key, so there is one default, not two
    asked = []
    monkeypatch.setattr(chat, "_retrieve_rows", lambda *a, **kw: asked.append(a[3]) or ([], None))
    chat.search_chunks("q", variant="baseline")
    assert asked == [True]
