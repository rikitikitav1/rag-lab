from types import SimpleNamespace

from evals import retrieval_metrics


def _log(sources, marked=("gold.md",), dropped=None, pipeline="agent"):
    return SimpleNamespace(
        question=SimpleNamespace(original_text="q", marked_sources=list(marked)),
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
        question=SimpleNamespace(original_text="q", marked_sources=["gold.md"]),
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
        question=SimpleNamespace(original_text="q", marked_sources=["gold.md"]),
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
