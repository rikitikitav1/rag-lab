from types import SimpleNamespace

from evals import retrieval_metrics


def _log(sources, marked=("gold.md",), dropped=None, pipeline="agent"):
    return SimpleNamespace(
        question_id=1,
        chunks=None,
        question=SimpleNamespace(
            original_text="q", marked_sources=list(marked), source_question_id=None
        ),
        sources=[{"source": name, "hop": hop} for name, hop in sources],
        metrics={"retrieval": {"dropped_sources": list(dropped or [])}},
        pipeline=pipeline,
    )


def _evaluate(monkeypatch, logs):
    monkeypatch.setattr(retrieval_metrics, "load_logs", lambda run_name: logs)
    return retrieval_metrics.evaluate("run")


def test_a_rank_is_a_rank_inside_its_hop_not_across_the_concatenation(monkeypatch):
    # the gold is second in the second hop and fifth in the list the two hops concatenate to
    log = _log([("a.md", 1), ("b.md", 1), ("c.md", 1), ("d.md", 2), ("gold.md", 2)])
    m = _evaluate(monkeypatch, [log])

    assert (m["hit_at_k"], m["hits"]) == (1.0, 1)
    assert m["mrr"] == 0.2
    assert m["mrr_in_hop"] == 0.5
    assert m["found_at_hop"] == {"2": 1}


def test_a_row_without_hops_reads_as_one_hop(monkeypatch):
    # single_shot and every row written before the stamp: one retrieval, so both numbers agree
    log = SimpleNamespace(
        question_id=1, chunks=None,
        question=SimpleNamespace(
            original_text="q", marked_sources=["gold.md"], source_question_id=None
        ),
        sources=[{"source": "a.md"}, {"source": "gold.md"}],
        metrics={},
        pipeline="single_shot",
    )
    m = _evaluate(monkeypatch, [log])

    assert m["mrr"] == m["mrr_in_hop"] == 0.5
    assert m["found_at_hop"] == {"1": 1}


def test_a_dropped_chunk_counts_and_belongs_to_the_first_hop_only_where_there_is_one(monkeypatch):
    # the gate hides weak chunks from the model, and retrieval still returned them
    single = _log([("a.md", 1)], dropped=["gold.md"], pipeline="single_shot")
    m = _evaluate(monkeypatch, [single])

    assert m["hits"] == 1
    assert m["mrr"] == m["mrr_in_hop"] == 0.5

    # an agent's dropped sources carry no hop, so hop 1 is the claim the module refuses
    m = _evaluate(monkeypatch, [_log([("a.md", 1)], dropped=["gold.md"])])

    assert (m["hits"], m["mrr"]) == (1, 0.5)
    assert (m["mrr_in_hop"], m["hop_unknown"]) == (None, 1)


def test_a_miss_moves_neither_number(monkeypatch):
    m = _evaluate(monkeypatch, [_log([("a.md", 1), ("b.md", 2)])])

    assert (m["hits"], m["mrr"], m["mrr_in_hop"], m["found_at_hop"]) == (0, 0.0, None, {})


def test_an_agent_row_from_before_the_stamp_is_unknown_rather_than_hop_one(monkeypatch):
    # an agent row written before the stamp would report the gold found on hop one
    before = SimpleNamespace(
        question_id=1, chunks=None,
        question=SimpleNamespace(
            original_text="q", marked_sources=["gold.md"], source_question_id=None
        ),
        sources=[{"source": "a.md"}, {"source": "gold.md"}],
        metrics={},
        pipeline="agent",
    )
    m = _evaluate(monkeypatch, [before])

    assert (m["hits"], m["mrr"]) == (1, 0.5)
    assert (m["mrr_in_hop"], m["found_at_hop"], m["hop_unknown"]) == (None, {}, 1)


def test_a_gold_in_no_hop_is_unknown_rather_than_credited_to_the_first():
    # `_rank_inside_its_hop` ended `return 1, None` and the caller credited `found_at_hop[1]`
    from evals import retrieval_metrics as rm

    kept = [{"source": "a.md", "hop": 2}]
    assert rm._rank_inside_its_hop(kept, [], ["nothing/here"]) == (None, None)


def test_precision_counts_the_files_retrieved_not_the_chunks(monkeypatch):
    # `sources` already holds one entry per path, and the standard takes both id lists as sets
    from evals.retrieval_metrics import file_precision

    assert file_precision(["gold.md", "a.md"], ["gold.md"]) == 0.5
    assert file_precision(["gold.md"], ["gold.md"]) == 1.0
    assert file_precision(["a.md", "b.md"], ["gold.md"]) == 0.0
    # a path is gold by containment here and an id by equality there: reduce, then compare
    assert file_precision(["repo/gold.md", "repo/a.md"], ["gold.md"]) == 0.5
    # the same file twice counts once, or a run that repeats a source looks less focused
    assert file_precision(["gold.md", "gold.md", "a.md"], ["gold.md"]) == 0.5
    assert file_precision([], ["gold.md"]) is None


def test_precision_falls_when_more_files_are_reached_not_only_when_wrong_ones_are(monkeypatch):
    # it reads as focus: the gold is found in both, and the number still drops
    from evals.retrieval_metrics import file_precision

    assert file_precision(["gold.md", "a.md"], ["gold.md"]) == 0.5
    assert round(file_precision(["gold.md", "a.md", "b.md"], ["gold.md"]), 4) == 0.3333


def test_a_run_reports_its_focus_beside_its_hits(monkeypatch):
    log = _log([("gold.md", 1), ("a.md", 1)])
    m = _evaluate(monkeypatch, [log])

    assert m["schema"] == 6
    assert (m["hit_at_k"], m["file_precision"], m["n_precision_scored"]) == (1.0, 0.5, 1)


def test_precision_counts_a_gated_file_the_way_the_rank_does(monkeypatch):
    # `hit@k` ranks over kept + dropped, so the focus axis must read the same population
    log = _log([("gold.md", 1)], pipeline="single_shot")
    log.metrics = {"retrieval": {"dropped_sources": ["a.md"]}}
    m = _evaluate(monkeypatch, [log])

    assert m["file_precision"] == 0.5, "a file the gate hid was still retrieved"


def _chunk(source, section, index=0):
    return {"source": source, "section": section, "chunk_index": index}


def test_a_chunk_of_the_right_file_in_the_wrong_section_is_not_a_section_hit():
    # the whole point of the axis: file gold calls this a hit, section gold does not
    from evals.retrieval_metrics import rank_of_gold_section

    chunks = [_chunk("repo/gold.md", "Gold file > Some other heading")]
    assert rank_of_gold_section(chunks, ["gold.md"], "The wanted heading") is None

    chunks = [_chunk("repo/gold.md", "Gold file > The wanted heading")]
    assert rank_of_gold_section(chunks, ["gold.md"], "The wanted heading") == 1


def test_the_section_rank_is_a_rank_and_reads_the_leaf_of_the_path():
    from evals.retrieval_metrics import rank_of_gold_section

    chunks = [
        _chunk("a.md", "A > first"),
        _chunk("repo/gold.md", "Gold > wrong"),
        _chunk("repo/gold.md", "Gold > 12. wanted"),
    ]
    # the leaf identifies the section, and a numbered heading keeps its identity
    assert rank_of_gold_section(chunks, ["gold.md"], "wanted") == 3
    assert rank_of_gold_section(chunks, ["nothing.md"], "wanted") is None


def test_a_row_without_addresses_is_not_scored_on_sections(monkeypatch):
    # `chunks` exists from 06.09 on, and the archive is never readable at this level
    log = _log([("gold.md", 1)])
    log.chunks = None
    m = _evaluate(monkeypatch, [log])

    assert m["schema"] == 6
    assert m["n_section_scored"] == 0
    assert m["section_hit_at_k"] is None and m["section_mrr"] is None


def test_the_section_axes_declare_that_they_see_the_kept_chunks():
    from evals import retrieval_metrics as rm

    assert rm.SCHEMA == 6
