import time
from types import SimpleNamespace

import pytest
from use_cases import chat


def _row(src, vector_rank=1, keyword_rank=None, vector_distance=0.1, score=0.5, content="content"):
    from db import Hit

    return Hit(content, src, "cat", 0, vector_rank, keyword_rank, vector_distance, score, None)


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


def _offline_snapshot(monkeypatch, device=None):
    from use_cases import run_snapshot

    monkeypatch.setattr(run_snapshot, "_rerank_device", lambda: device)
    monkeypatch.setattr(run_snapshot.db, "fingerprint_or_none", lambda *, variant: None)
    monkeypatch.setattr(run_snapshot.llm, "server_context_length", lambda model: 8192)
    monkeypatch.setattr(run_snapshot.llm, "resolve_name", lambda role: "stub")

def test_config_snapshot_records_device_only_when_reranking(monkeypatch):
    _offline_snapshot(monkeypatch, device="cuda")

    assert chat._config_snapshot(True, 5, False, 0.55, None, "baseline")["rerank_device"] == "cuda"
    assert chat._config_snapshot(False, 5, False, 0.55, None, "baseline")["rerank_device"] is None


def test_the_snapshot_records_the_depth_the_search_used(monkeypatch):
    # it resolved the depth itself, asking the planner twice and reaching for a database
    _offline_snapshot(monkeypatch)
    snap = chat._config_snapshot(False, 5, False, 0.55, None, "baseline", 200)
    assert snap["ef_search"] == 200


def test_config_snapshot_carries_procedure_fields(monkeypatch):
    _offline_snapshot(monkeypatch)
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
    monkeypatch.setattr(chat, "_retrieve_rows", lambda *a, **kw: (rows, None, 200))
    monkeypatch.setattr(chat, "_gate_scores", lambda query, rows, top: [0.42, None])

    _, _texts, sources, _ = chat.search_chunks(
        "q", use_rerank=False, gate_top=1, variant="baseline"
    )

    assert [s.rerank_score for s in sources] == [0.42, None]


def test_search_chunks_leaves_rerank_scores_alone(monkeypatch):
    rows = [_row("a.md")]
    monkeypatch.setattr(chat, "_retrieve_rows", lambda *a, **kw: (rows, [0.77], 200))
    monkeypatch.setattr(chat, "_gate_scores", lambda *a, **kw: pytest.fail("gate ran anyway"))

    _, _texts, sources, _ = chat.search_chunks(
        "q", use_rerank=True, gate_top=5, variant="baseline"
    )

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
    # the handler passes the switch through and the runner resolves it
    import config
    from evals import runner
    from job_handlers import evaluation

    # True over a config already False: a resolver ignoring config passes the older test
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
    monkeypatch.setattr(chat, "_retrieve_rows", lambda *a, **kw: asked.append(a[3]) or ([], None, 200))
    chat.search_chunks("q", variant="baseline")
    assert asked == [True]


def test_the_joined_context_is_exactly_the_chunks_it_lists():
    # RAGAS scores positions and the join cannot be undone: a separator may sit in a chunk
    from use_cases import chat

    rows = [_row("a.md", content="first body"), _row("b.md", content="second body")]
    texts = chat.chunk_texts(rows, variant="baseline")

    assert texts == ["[a.md]\nfirst body", "[b.md]\nsecond body"]
    assert chat.format_chunks(rows, variant="baseline") == "\n\n".join(texts)


def test_the_corpus_tool_hands_the_chunks_on_as_well_as_the_text(monkeypatch):
    # the agent's context is the tool messages joined, so the elements are lost at hop one
    import agent_tools
    from use_cases import chat as chat_module

    monkeypatch.setattr(
        chat_module, "search_chunks", lambda *a, **kw: ("joined", ["one", "two"], [], 100)
    )
    result = agent_tools._search_corpus("q", variant="baseline")

    assert result.content == "joined"
    assert result.meta["contexts"] == ["one", "two"]


def test_both_pipelines_record_the_same_keys_and_leave_what_they_lack_empty(monkeypatch):
    # the keys only the agent wrote are the ones the preflight pins
    from use_cases import run_snapshot

    _offline_snapshot(monkeypatch)
    single_shot = chat._config_snapshot(False, 5, True, 0.55, None, "baseline")

    assert set(single_shot) == set(run_snapshot.KEYS)
    assert single_shot["code_version"] == run_snapshot.version.CODE_VERSION
    # what the other pipeline measures is present and empty, not missing
    assert single_shot["max_hops"] is None
    assert single_shot["phased"] is True


def test_the_snapshot_refuses_a_key_it_has_no_place_for(monkeypatch):
    # a writer inventing a field is how the two sides drifted apart in the first place
    from use_cases import run_snapshot

    _offline_snapshot(monkeypatch)
    with pytest.raises(ValueError, match="no place for"):
        run_snapshot.of_run(
            variant="baseline", use_rerank=False, k=5, ef_search=None,
            distance_threshold=0.5, hops_max=4,
        )


def test_the_row_snapshot_says_which_schema_it_is(monkeypatch):
    # every other record carries one, and this branch changed what a snapshot means
    from use_cases import run_snapshot

    _offline_snapshot(monkeypatch)
    snap = chat._config_snapshot(False, 5, True, 0.55, None, "baseline")

    assert snap["schema"] == run_snapshot.SCHEMA
