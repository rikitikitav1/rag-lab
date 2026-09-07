from types import SimpleNamespace

import agent_tools
import llm
from evals import replay


def _row(**over):
    base = dict(
        transcript=[
            {"role": "system", "content": "you are"},
            {"role": "user", "content": "how do I log a request"},
            {"role": "assistant", "content": "",
             "tool_calls": [{"name": "search_corpus", "arguments": '{"query": "logging"}'}]},
            {"role": "tool"},
            {"role": "assistant", "content": "use a middleware"},
        ],
        contexts=["[a.md]\nfirst", "[b.md]\nsecond"],
        chunks=[{"source": "a.md"}, {"source": "b.md"}],
        sources=[{"source": "a.md", "hop": 1}, {"source": "b.md", "hop": 1}],
        metrics={"spans": [{"tool_call_id": "0", "tool": "search_corpus", "hop": 1,
                            "pieces": 2, "sources": ["a.md", "b.md"]}]},
    )
    return SimpleNamespace(**{**base, **over})


def test_only_the_model_turns_come_back_as_turns():
    # a replay must ask the model nothing, or it compares two samplings instead of two graphs
    turns = replay.turns_of(_row().transcript)

    assert len(turns) == 2
    assert isinstance(turns[0], llm.ChatTurn)
    assert turns[0].text is None and turns[1].text == "use a middleware"
    assert turns[0].tool_calls[0].function.name == "search_corpus"
    assert turns[0].tool_calls[0].function.arguments == '{"query": "logging"}'
    assert turns[1].tool_calls == []


def test_the_spans_put_the_flat_contexts_back_where_they_came_from():
    results = replay.results_of(replay.of_row(_row()))

    assert len(results) == 1
    assert isinstance(results[0], agent_tools.ToolResult)
    assert results[0].content == "[a.md]\nfirst\n\n[b.md]\nsecond"
    assert results[0].meta["contexts"] == ["[a.md]\nfirst", "[b.md]\nsecond"]
    assert [s.source for s in results[0].meta["sources"]] == ["a.md", "b.md"]


def test_two_calls_on_one_hop_do_not_bleed_into_each_other():
    # the flat list was the whole reason a replay could not be built before the spans
    row = _row(
        contexts=["one", "two", "three"],
        chunks=[{"source": "a"}, {"source": "b"}, {"source": "c"}],
        sources=[{"source": "a"}, {"source": "b"}, {"source": "c"}],
        metrics={"spans": [
            {"tool_call_id": "0", "tool": "search_corpus", "hop": 1, "pieces": 1,
             "sources": ["a"]},
            {"tool_call_id": "1", "tool": "deepwiki__ask", "hop": 1, "pieces": 2,
             "sources": ["b", "c"]},
        ]},
    )
    first, second = replay.results_of(replay.of_row(row))

    assert first.meta["contexts"] == ["one"]
    assert second.meta["contexts"] == ["two", "three"]
    assert [s.source for s in second.meta["sources"]] == ["b", "c"]


def test_a_row_that_recorded_no_span_replays_nothing_rather_than_guessing():
    row = _row(metrics={})
    assert replay.results_of(replay.of_row(row)) == []


def test_a_run_replays_into_the_same_row_it_recorded(monkeypatch):
    # the round trip is the whole point: record with the real graph, then regroup what it wrote
    import test_agent as live
    from use_cases import agent
    from use_cases.agent import transcript_of

    turns = [
        live._turn(tool_calls=[live._tool_call("a", "search_corpus", '{"query": "logging"}')],
                   message={"role": "assistant"}),
        live._turn(text="use a middleware"),
    ]
    live._agent_harness(monkeypatch, turns, corpus_sources=[SimpleNamespace(source="src/a.md")])
    recorded = agent.run("how do I log a request", max_hops=2)

    row = SimpleNamespace(
        transcript=transcript_of(recorded.messages),
        contexts=list(recorded.contexts),
        chunks=list(recorded.chunks),
        sources=[{"source": s.source} for s in recorded.sources],
        metrics={"spans": list(recorded.spans)},
    )
    again = replay.of_row(row)

    assert [t.text for t in replay.turns_of(again.transcript)] == [None, "use a middleware"]
    results = replay.results_of(again)
    assert len(results) == len(recorded.spans) == 1
    assert [p for r in results for p in r.meta["contexts"]] == recorded.contexts
    assert [s.source for r in results for s in r.meta["sources"]] == [
        s.source for s in recorded.sources
    ]


def test_a_file_seen_twice_does_not_slide_every_slice_after_it():
    # `_unique_sources` collapses across hops, so a counted span stopped addressing the row
    row = _row(
        contexts=["one", "two"],
        chunks=[{"source": "a"}, {"source": "a"}],
        sources=[{"source": "a", "hop": 1}],
        metrics={"spans": [
            {"tool_call_id": "0", "tool": "search_corpus", "hop": 1, "pieces": 1,
             "sources": ["a"]},
            {"tool_call_id": "1", "tool": "search_corpus", "hop": 2, "pieces": 1,
             "sources": ["a"]},
        ]},
    )
    first, second = replay.results_of(replay.of_row(row))

    assert [s.source for s in first.meta["sources"]] == ["a"]
    assert [s.source for s in second.meta["sources"]] == ["a"], (
        "the second call saw the same file, and a cursor would have handed it nothing"
    )


def test_the_idiomatic_arm_speaks_the_dialect_the_replay_reads():
    # the writer said `ai` and `human`, the reader looked for `assistant`, and found nothing
    from orchestrators.react import _as_message
    from use_cases.agent import transcript_of

    class _Msg:
        def __init__(self, type_, content, tool_calls=(), tool_call_id=None):
            self.type, self.content = type_, content
            self.tool_calls, self.tool_call_id = list(tool_calls), tool_call_id

    messages = [
        _Msg("system", "you are"),
        _Msg("human", "q"),
        _Msg("ai", "", [{"name": "search_corpus", "args": {"query": "logging"}}]),
        _Msg("tool", "a chunk", tool_call_id="1"),
        _Msg("ai", "the answer"),
    ]
    transcript = transcript_of([_as_message(m) for m in messages])

    assert [e["role"] for e in transcript] == ["system", "user", "assistant", "tool", "assistant"]
    turns = replay.turns_of(transcript)
    assert [t.text for t in turns] == [None, "the answer"]
    assert turns[0].tool_calls[0].function.name == "search_corpus"
    assert turns[0].tool_calls[0].function.arguments == '{"query": "logging"}'


def test_a_call_that_found_nothing_replays_as_nothing_not_as_an_empty_piece():
    # the row keeps no tool content, and "" counts as context where NO_RESULTS counts as none
    row = _row(
        contexts=["[a.md]\nfirst"],
        chunks=[{"source": "a.md"}],
        sources=[{"source": "a.md", "hop": 2}],
        metrics={"spans": [
            {"tool_call_id": "0", "tool": "search_corpus", "hop": 1, "pieces": 0, "sources": []},
            {"tool_call_id": "1", "tool": "search_corpus", "hop": 2, "pieces": 1,
             "sources": ["a.md"]},
        ]},
    )
    empty, found = replay.results_of(replay.of_row(row))

    assert agent_tools.context_pieces(empty.meta, empty.content) == []
    assert agent_tools.chunk_pieces(empty.meta, empty.content) == []
    assert agent_tools.context_pieces(found.meta, found.content) == ["[a.md]\nfirst"]


def test_a_recorded_run_replays_field_for_field_through_the_graph_it_was_not_recorded_by(
    monkeypatch,
):
    # the split of the phases is checked by equality, not by a threshold on two samplings
    import test_agent as live
    from use_cases import agent
    from use_cases.agent import transcript_of

    # the client hands the calls back on the message, and a fake that drops them hides the replay
    calls = [{"function": {"name": "search_corpus", "arguments": '{"query": "logging"}'}}]
    turns = [
        live._turn(tool_calls=[live._tool_call("a", "search_corpus", '{"query": "logging"}')],
                   message={"role": "assistant", "content": "", "tool_calls": calls}),
        live._turn(text="use a middleware"),
    ]
    live._agent_harness(monkeypatch, turns, corpus_sources=[SimpleNamespace(source="src/a.md")])
    recorded = agent.run("how do I log a request", max_hops=2)

    row = SimpleNamespace(
        question_text="how do I log a request",
        prompts={},
        transcript=transcript_of(recorded.messages),
        contexts=list(recorded.contexts),
        chunks=list(recorded.chunks),
        sources=[{"source": s.source, "hop": getattr(s, "hop", None)} for s in recorded.sources],
        metrics={
            "spans": list(recorded.spans),
            "config": {"max_hops": 2, "k": None, "mcp": [], "fallback_policy": "corpus_first"},
            "fallback_reason": str(recorded.fallback_reason),
            "outcome": str(recorded.outcome),
        },
    )
    result, problems = replay.rerun(row)

    assert problems == []
    assert replay.differences(row, result) == []


def test_a_row_naming_a_prompt_version_that_is_gone_is_refused_not_replayed(monkeypatch):
    # one bad row killed the report for every good one
    import prompt_repo

    monkeypatch.setattr(prompt_repo, "template_of", _raises)
    row = _row(question_text="q", prompts={"agent_system": 99}, metrics={"config": {}, "spans": []})
    result, problems = replay.rerun(row)

    assert result is None
    assert "99" in problems[0]


def _raises(purpose, version):
    raise RuntimeError(f"no prompt {purpose} version {version}")


def test_a_row_whose_gate_dropped_context_replays_into_the_same_drop(monkeypatch):
    # the branch the split was made for: the row now keeps what the gate rewrote
    import test_agent as live
    from use_cases import agent
    from use_cases.agent import transcript_of

    calls = [{"function": {"name": "search_corpus", "arguments": '{"query": "logging"}'}}]
    turns = [
        live._turn(tool_calls=[live._tool_call("a", "search_corpus", '{"query": "logging"}')],
                   message={"role": "assistant", "content": "", "tool_calls": calls}),
        live._turn(text="nothing useful here"),
    ]
    live._agent_harness(monkeypatch, turns, corpus_sources=[live._weak_hit()])
    recorded = agent.run("how do I log a request", max_hops=2,
                         fallback_policy="corpus_first_weak", gate_signal="cross_encoder")

    assert recorded.dropped_sources, "the gate did not fire, the test proves nothing"
    spans = list(recorded.spans)
    assert spans[0].get("dropped"), "the span does not carry what the gate rewrote"

    row = SimpleNamespace(
        question_text="how do I log a request", prompts={},
        transcript=transcript_of(recorded.messages),
        contexts=list(recorded.contexts), chunks=list(recorded.chunks),
        sources=[{"source": s.source, "hop": getattr(s, "hop", None)} for s in recorded.sources],
        metrics={
            "spans": spans,
            "config": {"max_hops": 2, "mcp": [], "fallback_policy": "corpus_first_weak",
                       "gate": {"signal": "cross_encoder", "top": None, "threshold": 0.35,
                                "distance_threshold": None},
                       "drop_weak_context": True},
            "retrieval": {"dropped_sources": list(recorded.dropped_sources)},
            "fallback_reason": str(recorded.fallback_reason),
            "outcome": str(recorded.outcome),
        },
    )
    result, problems = replay.rerun(row)

    assert problems == []
    assert result is not None, "the replay still refuses the branch it can now rebuild"
    assert str(result.fallback_reason) == str(recorded.fallback_reason)
    assert replay.differences(row, result) == []


def test_the_replay_compares_the_toolbox_each_hop_was_handed(monkeypatch):
    # the scripted chat ignores `tools`, so a graph that opened the toolbox early went unseen
    import test_agent as live
    from use_cases import agent
    from use_cases.agent import transcript_of

    live._agent_harness(monkeypatch, [live._turn(text="no tools needed")], corpus_sources=[])
    recorded = agent.run("q", max_hops=2)

    assert recorded.tools_offered == [["search_corpus"]]
    row = SimpleNamespace(
        question_text="q", prompts={}, transcript=transcript_of(recorded.messages),
        contexts=[], chunks=[], sources=[],
        metrics={"spans": [], "config": {"max_hops": 2, "mcp": []},
                 "tools_offered": [["search_corpus", "deepwiki__ask_question"]],
                 "fallback_reason": str(recorded.fallback_reason),
                 "outcome": str(recorded.outcome)},
    )
    _, problems = replay.rerun(row)

    assert any("toolbox" in p for p in problems), problems


def test_the_notice_text_is_compared_where_the_row_carries_it():
    # `announced` said the notice fired and nothing said what it said, so a rewrite went unseen
    row = _row(
        prompts={"agent_fallback": "1.0"},
        metrics={"spans": [], "announced_text": "the corpus has nothing, try search_web"},
    )
    same = SimpleNamespace(
        messages=[], sources=[], chunks=[], contexts=[], spans=[],
        fallback_reason="off_topic", outcome="refused", fallback_announced=True,
        announced_text="the corpus has nothing, try search_web",
    )
    assert "announced_text" not in replay.differences(row, same)

    rewritten = SimpleNamespace(**{**vars(same), "announced_text": "sorry, nothing found"})
    assert "announced_text" in replay.differences(row, rewritten)


def test_a_row_recorded_before_the_notice_was_kept_is_not_failed_for_it():
    # every row of the three arms predates the field: comparing it would fail them all and prove none
    row = _row(prompts={"agent_fallback": "1.0"}, metrics={"spans": []})
    result = SimpleNamespace(
        messages=[], sources=[], chunks=[], contexts=[], spans=[],
        fallback_reason="off_topic", outcome="refused", fallback_announced=True,
        announced_text="whatever today's template says",
    )
    assert "announced_text" not in replay.differences(row, result)
    assert replay._recorded(row)["announced_text"] is replay.UNRECORDED


def test_the_language_probe_restates_the_answer_not_a_line_of_the_context():
    # pass 1 restated a context line, so the judge read a non-answer and scored a fifth of them zero
    from types import SimpleNamespace as NS

    import llm
    from evals import judge_language as jl

    asked = []
    original = llm.ask
    llm.ask = lambda prompt, text, **kw: (asked.append(text), NS(text="restated"))[1]
    try:
        pairs = jl.pairs_for(NS(answer="the middleware logs the method", contexts=["x" * 120]))
    finally:
        llm.ask = original

    assert [lang for lang, _ in pairs] == ["en", "ru"]
    assert asked == ["the middleware logs the method"] * 2, "it restated the context, not the answer"


def test_a_weak_verdict_that_dropped_nothing_is_not_refused(monkeypatch):
    # the gate drops only where its own snapshot says so, and that flag is decided per question
    import prompt_repo

    dropped = _row(metrics={"spans": [], "fallback_reason": "weak", "config": {},
                            "retrieval": {"dropped_sources": [{"source": "a.md"}]}})
    assert "predates recording" in replay.rerun(dropped)[1][0]

    # json null, not an absent key: the sql `[*]` wrapped it into a list and called it a drop
    monkeypatch.setattr(prompt_repo, "template_of", _raises)
    nothing_dropped = _row(
        question_text="q", prompts={"agent_system": 99},
        metrics={"spans": [], "fallback_reason": "weak", "config": {},
                 "retrieval": {"dropped_sources": None}},
    )
    assert "no prompt" in replay.rerun(nothing_dropped)[1][0], "it stopped at the drop check"


def test_the_python_side_splits_every_shape_the_catalogue_names():
    # sql reads the same key in another language, and only a base can prove the two agree
    from evals.run_debts import REPLAY_SHAPES

    refuses = {
        name: bool(shape.get("dropped_sources")) for name, shape in REPLAY_SHAPES.items()
    }
    assert refuses == {
        "json null": False, "empty list": False, "key absent": False, "a real drop": True
    }


def test_the_probe_refuses_its_own_numbers_when_it_is_out_of_regime():
    # pass 1 scored a grounded restatement zero in a fifth of pairs, and nothing said the regime slid
    from evals import judge_language as jl

    good = jl.control([7] * 96 + [0] * 4)
    assert good["control"]["share"] == 0.96 and good["control"]["in_regime"] is True

    # the share pass 1 actually produced, against a history of 96.4% and 95.6%
    bad = jl.control([7] * 62 + [0] * 38)
    assert bad["control"]["share"] == 0.62 and bad["control"]["in_regime"] is False
    assert jl.control([])["control"]["in_regime"] is False


def test_the_bare_arm_still_gets_its_ceiling_re_derived():
    # `react.invoke` stamps `unrecorded` on every row, and a truthy stamp skipped the hop count
    from evals.pools import _exhausted

    bare = {"finished_by": "unrecorded", "hops": 4}
    assert _exhausted(bare, {"max_hops": 4}) is True
    assert _exhausted(bare, {"max_hops": 8}) is False
    assert _exhausted({"finished_by": "answer", "hops": 9}, {"max_hops": 4}) is False


def test_a_span_is_compared_without_the_id_and_without_a_drop_the_row_never_kept():
    # `spans` sat outside the compared fields, so a phantom drop block would have replayed silently
    from evals import replay as r

    old = [{"tool_call_id": "a", "hop": 1, "pieces": 0}]
    now = [{"tool_call_id": "0", "hop": 1, "pieces": 0, "dropped": {"pieces": 0}}]
    assert r._comparable(old, r._kept_the_drop(old)) == r._comparable(now, r._kept_the_drop(old))

    kept = [{"tool_call_id": "a", "hop": 1, "dropped": {"pieces": 2}}]
    moved = [{"tool_call_id": "0", "hop": 1, "dropped": {"pieces": 9}}]
    assert r._comparable(kept, True) != r._comparable(moved, True), "a changed drop must be seen"


def test_the_two_section_rankers_spell_one_metric():
    # `run_metrics.section_mrr` and the comparison report ranked over lists of different lengths
    from evals.retrieval_metrics import section_ids
    from use_cases.retrieval_compare import heading_text

    chunks = [
        {"source": "a.md", "section": "# Logging"},
        {"source": "a.md", "section": "# Logging"},
        {"source": "a.md", "section": "# Routing"},
    ]
    assert section_ids(chunks) == [("a.md", heading_text("# Logging")),
                                   ("a.md", heading_text("# Routing"))]


def test_two_passes_of_a_probe_in_one_day_are_two_files():
    # the second overwrote the first, and a measurement nobody can reread is not a measurement
    import tempfile
    from datetime import date
    from pathlib import Path as Folder

    from evals import measurements

    with tempfile.TemporaryDirectory() as folder:
        original, measurements.FOLDER = measurements.FOLDER, Folder(folder)
        try:
            on = date(2026, 9, 7)
            first = measurements.record("probe", "run", {"n": 1}, on=on)
            second = measurements.record("probe", "run", {"n": 2}, on=on)
        finally:
            measurements.FOLDER = original
    assert first != second and second.endswith("_2.json")


def test_a_row_from_another_arm_is_refused_rather_than_driven_through_the_graph():
    # the bare arm speaks its own dialect, and the debt counted its rows as replayable
    row = _row(metrics={"spans": [], "config": {
        "orchestrator": {"name": "langgraph_idiomatic", "client": "ChatOllama"}
    }})
    result, problems = replay.rerun(row)

    assert result is None and "langgraph_idiomatic" in problems[0]
    assert "langgraph_ported" in replay.REPLAYABLE_ARMS
