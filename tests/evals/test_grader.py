from types import SimpleNamespace

from use_cases import grading


def _verdicts(monkeypatch, said: list):
    from use_cases import agent

    asked, verdicts = [], list(said)

    # the admission probe asks through the same door, and only a passage is the grader's
    def ask(system, user, role=None, model=None, schema=None, logprobs=False):
        if "Passage:" not in user:
            return SimpleNamespace(text="yes")
        asked.append({"role": role, "user": user, "schema": schema})
        return SimpleNamespace(text=verdicts.pop(0) if verdicts else "yes")

    monkeypatch.setattr(agent.llm, "ask", ask)
    return asked


# the real corpus tool records an address per piece, and both the memo and the replay read it
def _harness(monkeypatch, turns, sources):
    import agent_tools
    import test_agent as live
    from use_cases import agent

    live._agent_harness(monkeypatch, turns, corpus_sources=sources)

    def dispatch(name, arguments, extra=None, **runtime):
        if name != agent_tools.CORPUS_TOOL:
            return agent_tools.ToolResult(
                content="remote said so", meta={"sources": [SimpleNamespace(source="remote")]}
            )
        contexts = [f"[{s.source}]\nc" for s in sources]
        return agent_tools.ToolResult(
            content="\n\n".join(contexts),
            meta={"sources": list(sources), "contexts": contexts,
                  "chunks": [{"source": s.source, "section": "s", "chunk_index": n}
                             for n, s in enumerate(sources)]},
        )

    monkeypatch.setattr(agent_tools, "dispatch", dispatch)
    return agent


def _one_hop(monkeypatch, said, sources, **kwargs):
    import test_agent as live

    turns = [
        live._turn(tool_calls=[live._tool_call("a", "search_corpus", '{"query": "q"}')],
                   message={"role": "assistant"}),
        live._turn(text="an answer"),
    ]
    agent = _harness(monkeypatch, turns, sources)
    asked = _verdicts(monkeypatch, said)
    return agent.run("how does MVCC work", max_hops=2, **kwargs), asked


def _hit(source: str, content: str):
    from db import Hit

    return Hit(content, source, "cat", 0, 1, None, 0.2, 0.5, None)


def _two_chunks():
    return [SimpleNamespace(source="src/mvcc.md"), SimpleNamespace(source="src/kafka.md")]


def test_the_grader_drops_the_chunk_it_called_foreign_and_keeps_the_rest(monkeypatch):
    result, asked = _one_hop(monkeypatch, ["yes", "no"], _two_chunks(), grade_chunks=True)

    assert result.contexts == ["[src/mvcc.md]\nc"], "the foreign chunk reached the generator"
    graded = [a for a in result.asks if a["stage"] == "grade"]
    assert [a["text"] for a in graded] == ["yes", "no"]
    assert [a["key"] for a in graded] == ["src/mvcc.md#0", "src/kafka.md#1"]
    assert [a["schema"] for a in asked] == [grading.VERDICT_SCHEMA] * 2
    step = next(s for s in result.trace if s["node"] == "grade")
    assert (step["graded"], step["kept"], step["dropped"]) == (2, 1, 1)


def test_a_verdict_that_cannot_be_read_keeps_the_chunk(monkeypatch):
    # a grader is not a judge: a cut or garbled word must not quietly vote against a chunk
    result, _ = _one_hop(monkeypatch, ["", "perhaps"], _two_chunks(), grade_chunks=True)

    assert len(result.contexts) == 2
    step = next(s for s in result.trace if s["node"] == "grade")
    assert (step["kept"], step["unreadable"]) == (2, 2)


def test_the_grader_stays_out_of_a_run_that_did_not_ask_for_it(monkeypatch):
    result, _ = _one_hop(monkeypatch, ["no", "no"], _two_chunks())

    assert len(result.contexts) == 2, "a run nobody asked to filter came back filtered"
    assert [a for a in result.asks if a["stage"] == "grade"] == []
    assert [s for s in result.trace if s["node"] == "grade"] == []


def test_a_row_the_grader_emptied_says_so_instead_of_borrowing_the_gates_reason(monkeypatch):
    # `empty` counts as `gate_fired` in the comparison report, and the grader is not the gate
    from evals.compare import GATE_REASONS

    result, _ = _one_hop(monkeypatch, ["no", "no"], _two_chunks(), grade_chunks=True)

    assert result.contexts == []
    assert result.fallback_reason == "graded_out"
    assert "graded_out" not in GATE_REASONS


def test_a_grader_that_emptied_the_corpus_does_not_open_the_way_outside(monkeypatch):
    result, _ = _one_hop(monkeypatch, ["no", "no"], _two_chunks(), grade_chunks=True)

    step = next(s for s in result.trace if s["node"] == "fallback")
    assert step["verdict"] == "graded_out"
    assert not step.get("opened") and not step.get("announced"), "a corpus row refuses instead"


def test_a_chunk_seen_on_a_second_hop_is_not_graded_twice(monkeypatch):
    import test_agent as live

    turns = [
        live._turn(tool_calls=[live._tool_call("a", "search_corpus", '{"query": "one"}')],
                   message={"role": "assistant"}),
        live._turn(tool_calls=[live._tool_call("b", "search_corpus", '{"query": "two"}')],
                   message={"role": "assistant"}),
        live._turn(text="an answer"),
    ]
    agent = _harness(monkeypatch, turns, _two_chunks())
    asked = _verdicts(monkeypatch, ["yes", "no"])
    result = agent.run("how does MVCC work", max_hops=3, grade_chunks=True)

    assert len(asked) == 2, "the second hop returned the same chunks and graded them again"
    steps = [s for s in result.trace if s["node"] == "grade"]
    assert [s["asked"] for s in steps] == [2, 0]
    assert [s["kept"] for s in steps] == [1, 1], "the memo keeps deciding, it does not stop deciding"


def test_the_grader_does_not_grade_a_context_the_fallback_will_throw_away_whole(monkeypatch):
    import test_agent as live

    turns = [
        live._turn(tool_calls=[live._tool_call("a", "search_corpus", '{"query": "q"}')],
                   message={"role": "assistant"}),
        live._turn(text="an answer"),
    ]
    agent = _harness(monkeypatch, turns, [live._weak_hit("src/a.md")])
    asked = _verdicts(monkeypatch, ["no"])
    result = agent.run(
        "how does MVCC work", max_hops=2, grade_chunks=True,
        fallback_policy="corpus_first_weak", weak_distance=0.1,
    )

    assert result.fallback_reason == "weak"
    assert asked == [], "verdicts were spent on a context the gate drops whole"
    assert [s for s in result.trace if s["node"] == "grade"] == []


def test_the_verdict_is_read_from_a_schema_answer_and_from_a_bare_word():
    assert grading.read_verdict('{"relevant": "yes"}') == "yes"
    assert grading.read_verdict("No") == "no"
    assert grading.read_verdict(" yes ") == "yes"
    assert grading.read_verdict("") is None
    assert grading.read_verdict("it depends") is None
    assert grading.read_verdict('{"relevant": "maybe"}') is None
    assert grading.read_verdict("{broken") is None


def test_a_cut_verdict_is_recorded_as_cut_and_the_report_names_the_row(monkeypatch):
    import test_agent as live
    from evals import trace

    turns = [
        live._turn(tool_calls=[live._tool_call("a", "search_corpus", '{"query": "q"}')],
                   message={"role": "assistant"}),
        live._turn(text="an answer"),
    ]
    agent = _harness(monkeypatch, turns, _two_chunks())

    def ask(system, user, role=None, model=None, schema=None, logprobs=False):
        if "Passage:" not in user:
            return SimpleNamespace(text="yes")
        return SimpleNamespace(text='{"relevant"', finish_reason="length")

    monkeypatch.setattr(agent.llm, "ask", ask)
    result = agent.run("how does MVCC work", max_hops=2, grade_chunks=True)

    assert all(a.get("cut") for a in result.asks if a["stage"] == "grade")
    row = SimpleNamespace(id=7, metrics={"trace": result.trace, "asks": result.asks,
                                         "outcome": result.outcome})
    got = trace.report([row])
    assert got["grader"]["cut_verdicts"] == 2
    assert got["grader"]["rows_graded_in_name_only"] == [7], "an arm that graded in name only"


def test_a_graded_row_replays_into_the_same_row_it_recorded(monkeypatch):
    # the ruler the grader was accepted on: the replay reads the recorded verdict and calls no model of its own
    import test_agent as live
    from evals import replay
    from use_cases.agent import transcript_of

    # the client hands the calls back on the message, and a fake that drops them hides the replay
    calls = [{"function": {"name": "search_corpus", "arguments": '{"query": "q"}'}}]
    turns = [
        live._turn(tool_calls=[live._tool_call("a", "search_corpus", '{"query": "q"}')],
                   message={"role": "assistant", "content": "", "tool_calls": calls}),
        live._turn(text="an answer"),
    ]
    agent = _harness(monkeypatch, turns, _two_chunks())
    _verdicts(monkeypatch, ["yes", "no"])
    result = agent.run("how does MVCC work", max_hops=2, grade_chunks=True)

    row = SimpleNamespace(
        id=1,
        question_text="how does MVCC work",
        prompts={},
        transcript=transcript_of(result.messages),
        contexts=list(result.contexts),
        chunks=list(result.chunks),
        sources=[{"source": s.source, "hop": getattr(s, "hop", None)} for s in result.sources],
        metrics={
            "spans": list(result.spans),
            "asks": list(result.asks),
            "fallback_reason": str(result.fallback_reason),
            "outcome": str(result.outcome),
            "config": {"grade_chunks": True, "max_hops": 2, "k": None, "mcp": [],
                       "fallback_policy": "corpus_first", "language": "en"},
        },
    )

    again, problems = replay.rerun(row)

    assert problems == [], "a graded row must replay through the door that recorded it"
    assert again is not None and again.contexts == result.contexts
    assert replay.differences(row, again) == []
    # `FIELDS` does not carry the trace, so the node's own record is compared apart from them
    assert again.trace == result.trace, "the replayed row walked the graph differently"


def test_the_direct_path_filters_through_the_same_grader(monkeypatch):
    # one grader for both paths: two of them would make the closing number measure another machine
    from use_cases import chat, grading

    rows = [_hit("src/mvcc.md", "mvcc keeps versions"), _hit("src/kafka.md", "kafka partitions")]
    monkeypatch.setattr(chat, "_hidden_by_cut", lambda source, variant: False)
    monkeypatch.setattr(chat, "_log_answer", lambda *a, **kw: kept.update(kw) or None)
    monkeypatch.setattr(chat.prompt_repo, "active_template", lambda purpose: "tpl")
    monkeypatch.setattr(chat.llm, "resolve_name", lambda role: "stub-model")
    monkeypatch.setattr(grading.prompt_repo, "active_template", lambda purpose: "tpl")
    said = iter(["no", "yes"])

    def ask(system, user, role=None, model=None, schema=None, logprobs=False):
        if "Passage:" in user:
            return SimpleNamespace(text=next(said))
        return SimpleNamespace(text="an answer", prompt_tokens=1, completion_tokens=1, parsed=None)

    monkeypatch.setattr(chat.llm, "ask", ask)
    monkeypatch.setattr(grading.llm, "ask", ask)
    kept: dict = {}

    ans = chat.answer_from_rows(
        "how does MVCC work", rows, variant="clean_1024", grade_chunks=True,
    )

    assert ans.success
    assert kept["contexts"] == ["[src/kafka.md]\nkafka partitions"], "the dropped chunk was read"
    assert kept["graded"]["dropped"] == ["src/mvcc.md#0"]
    assert [a["text"] for a in kept["asks"]] == ["no", "yes"]
    assert kept["graded"]["seconds"] >= 0, "the price of filtering is on the row"


def test_the_direct_path_does_not_grade_unless_the_run_asks(monkeypatch):
    from use_cases import chat

    rows = [_hit("src/mvcc.md", "mvcc")]
    monkeypatch.setattr(chat, "_hidden_by_cut", lambda source, variant: False)
    monkeypatch.setattr(chat, "_log_answer", lambda *a, **kw: kept.update(kw) or None)
    monkeypatch.setattr(chat.prompt_repo, "active_template", lambda purpose: "tpl")
    monkeypatch.setattr(chat.llm, "resolve_name", lambda role: "stub-model")
    monkeypatch.setattr(
        chat.llm, "ask",
        lambda **kw: SimpleNamespace(text="an answer", prompt_tokens=1, completion_tokens=1,
                                     parsed=None),
    )
    kept: dict = {}

    chat.answer_from_rows("how does MVCC work", rows, variant="clean_1024")

    assert kept.get("graded") is None and not kept.get("asks")


def test_a_run_that_grades_without_a_prompt_refuses_instead_of_breaking_every_row(monkeypatch):
    # a smoke wrote three rows with `hops: 0` because the prompt was never seeded
    import pytest
    from errors import StandFault
    from use_cases import grading

    def missing(purpose):
        raise RuntimeError(f"no active prompt for purpose {purpose}")

    monkeypatch.setattr(grading.prompt_repo, "active_template", missing)
    with pytest.raises(StandFault, match="grade.chunk"):
        grading.system_prompt()


def test_the_verdict_carries_the_probability_the_model_gave_it(monkeypatch):
    # a hard label alone draws no curve, and the same call already returns the number
    import test_agent as live
    from use_cases import grading

    turns = [
        live._turn(tool_calls=[live._tool_call("a", "search_corpus", '{"query": "q"}')],
                   message={"role": "assistant"}),
        live._turn(text="an answer"),
    ]
    agent = _harness(monkeypatch, turns, _two_chunks())
    asked = []

    def ask(system, user, role=None, model=None, schema=None, logprobs=False):
        if "Passage:" not in user:
            return SimpleNamespace(text="yes")
        asked.append(logprobs)
        return SimpleNamespace(
            text='{"relevant": "no"}',
            logprobs=[{"token": "no", "top": {"no": 0.77, "yes": 0.23}}],
        )

    monkeypatch.setattr(agent.llm, "ask", ask)
    result = agent.run("how does MVCC work", max_hops=2, grade_chunks=True)

    graded = [a for a in result.asks if a["stage"] == "grade"]
    assert asked == [True, True], "the grader must ask for the probabilities"
    assert [a["p"] for a in graded] == [0.77, 0.77]
    assert grading.confidence(SimpleNamespace(logprobs=None)) is None, "absent is not zero"


def test_two_chunks_of_one_file_do_not_leave_a_dropped_file_on_the_row(monkeypatch):
    # sources are per file and deduplicated, so filtering them by position names the wrong file
    import agent_tools
    import test_agent as live

    turns = [
        live._turn(tool_calls=[live._tool_call("a", "search_corpus", '{"query": "q"}')],
                   message={"role": "assistant"}),
        live._turn(text="an answer"),
    ]
    sources = [SimpleNamespace(source="src/mvcc.md"), SimpleNamespace(source="src/kafka.md")]
    agent = _harness(monkeypatch, turns, sources)

    def dispatch(name, arguments, extra=None, **runtime):
        # three chunks, two of them from one file: the row records two sources for three pieces
        contexts = ["[src/mvcc.md]\nversions", "[src/mvcc.md]\nvacuum", "[src/kafka.md]\npartitions"]
        return agent_tools.ToolResult(
            content="\n\n".join(contexts),
            meta={"sources": list(sources), "contexts": contexts,
                  "chunks": [{"source": "src/mvcc.md", "section": "s", "chunk_index": 0},
                             {"source": "src/mvcc.md", "section": "s", "chunk_index": 1},
                             {"source": "src/kafka.md", "section": "s", "chunk_index": 0}]},
        )

    monkeypatch.setattr(agent_tools, "dispatch", dispatch)
    _verdicts(monkeypatch, ["yes", "yes", "no"])
    result = agent.run("how does MVCC work", max_hops=2, grade_chunks=True)

    assert len(result.contexts) == 2
    assert [s.source for s in result.sources] == ["src/mvcc.md"], "the dropped file stayed on the row"
    step = next(s for s in result.trace if s["node"] == "grade")
    assert step.get("sources_unaligned") is None


def test_the_report_counts_rows_and_verdicts_not_hops_and_memo_hits():
    # a step is a hop: four hops of one row read as four graded rows and five chunks as twenty
    from types import SimpleNamespace as NS

    from evals import trace

    row = NS(id=3, metrics={
        "outcome": "answered",
        "asks": [{"stage": "grade", "key": "a#0", "text": "no", "cut": True}],
        "trace": [
            {"node": "grade", "hop": 1, "graded": 5, "kept": 4, "dropped": 1,
             "unreadable": 1, "asked": 5},
            # the second hop saw the same five chunks: four memo hits and one fresh verdict
            {"node": "grade", "hop": 2, "graded": 5, "kept": 4, "dropped": 1,
             "unreadable": 0, "asked": 1},
        ],
    })
    got = trace.report([row])["grader"]

    assert got["rows"] == 1, "one row, however many hops it spent"
    assert got["verdicts"] == 6, "ten pieces were seen and six verdicts were asked for"
    assert got["unreadable_share"] == round(1 / 6, 4), "the share is read over the verdicts asked"
    assert got["cut_share"] == round(1 / 6, 4)
    assert got["rows_graded_in_name_only"] == []


def test_a_row_whose_every_asked_verdict_was_unreadable_is_named(monkeypatch):
    from types import SimpleNamespace as NS

    from evals import trace

    row = NS(id=8, metrics={"outcome": "answered", "trace": [
        {"node": "grade", "hop": 1, "graded": 5, "kept": 5, "dropped": 0, "unreadable": 2,
         "asked": 2},
    ]})
    got = trace.report([row])["grader"]

    assert got["rows_graded_in_name_only"] == [8], "three memo hits must not hide two dead verdicts"


def test_the_trace_counts_the_grader_of_the_direct_path_too():
    # a filtered arm of `single_shot` writes no trace, and the report read it as an arm that never graded
    from types import SimpleNamespace

    from evals import trace

    direct = SimpleNamespace(id=1, metrics={
        "graded": {"asked": 5, "kept": [0, 1, 2], "unreadable": 0, "dropped": ["a#3", "a#4"]},
        "asks": [{"stage": "grade", "key": "a#0"}],
    })
    silent = SimpleNamespace(id=2, metrics={"graded": {"asked": 2, "kept": [], "unreadable": 2}})
    said = trace.report([direct, silent])["grader"]
    assert said["rows"] == 2 and said["verdicts"] == 7 and said["kept"] == 3
    assert said["rows_graded_in_name_only"] == [2], "every verdict unreadable is grading in name only"
