from types import SimpleNamespace

from evals import pools


def _log(kind=None, marked=None, answer="the corpus says hello", sources=(), metrics=None,
         faithfulness=None):
    return SimpleNamespace(
        question=SimpleNamespace(kind=kind, marked_sources=marked or []),
        answer=answer,
        sources=[{"source": s} for s in sources],
        metrics=metrics or {},
        faithfulness=faithfulness,
    )


def test_a_declared_kind_wins_over_the_marked_sources():
    assert pools.kind(_log(kind="off_domain", marked=["a.md"])) == "off_domain"
    assert pools.kind(_log(kind="nonsense", marked=["a.md"])) == "in_corpus"


def test_without_a_kind_the_marked_sources_decide():
    assert pools.kind(_log(marked=["a.md"])) == "in_corpus"
    assert pools.kind(_log()) == "out_of_corpus"


def test_an_in_corpus_question_with_nothing_marked_cannot_be_scored_against_the_corpus():
    split = pools.split([_log(kind="in_corpus")])
    assert split["in_corpus"] == []
    assert len(split["out_of_corpus"]) == 1


def test_rejected_questions_keep_their_own_bucket():
    split = pools.split([_log(kind="rejected", marked=["a.md"])])
    assert len(split["rejected"]) == 1
    assert split["in_corpus"] == []


def test_a_recorded_narration_is_trusted_over_the_text():
    log = _log(metrics={"outcome": "narrated_call"})
    log.answer = "No relevant documents found."
    assert pools.outcome(log) == "narrated_call"


def test_an_error_at_the_hop_cap_is_exhaustion_not_a_crash():
    capped = _log(answer="", metrics={"outcome": "error", "hops": 4, "config": {"max_hops": 4}})
    crashed = _log(answer="", metrics={"outcome": "error", "hops": 1, "config": {"max_hops": 4}})
    assert pools.outcome(capped) == "exhausted"
    assert pools.outcome(crashed) == "error"


def test_a_guard_that_fired_stays_an_error_at_the_same_hop_count():
    guarded = _log(
        answer="",
        metrics={"outcome": "error", "hops": 5, "failed": True, "config": {"max_hops": 4}},
    )
    spent = _log(answer="", metrics={"outcome": "error", "hops": 5, "config": {"max_hops": 4}})
    assert pools.outcome(guarded) == "error"
    assert pools.outcome(spent) == "exhausted"


def test_an_answer_without_sources_is_unsupported_not_answered():
    assert pools.outcome(_log(sources=["a.md"])) == "answered"
    assert pools.outcome(_log()) == "unsupported_answer"


def test_only_an_mcp_prefix_counts_as_remote_evidence():
    assert pools.has_remote_evidence(_log(sources=["mcp:deepwiki__ask_question"]))
    assert not pools.has_remote_evidence(_log(sources=["mcp_notes/readme.md"]))
    assert not pools.has_remote_evidence(_log())


def test_the_report_carries_a_bucket_for_every_outcome_the_enum_knows(monkeypatch):
    # the pre-registration listed three buckets while the data held four
    import outcomes
    from evals import generation_metrics

    log = SimpleNamespace(
        question=SimpleNamespace(original_text="q", marked_sources=["a.md"], kind=None),
        metrics={}, answered=True, answer="the corpus says hello",
        faithfulness=8, relevance=9, completeness=7, sources=[{"source": "a.md"}],
    )
    monkeypatch.setattr(generation_metrics, "load_logs", lambda run_name: [log])
    reported = generation_metrics.evaluate("run")["outcomes"]

    assert set(reported) == {o.value for o in outcomes.Outcome}


def test_sources_that_the_answer_did_not_use_are_their_own_bucket():
    # six off-domain questions came back with sources attached and scored 7 to 10
    assert pools.outcome(_log(sources=["a.md"], faithfulness="0")) == "answered_ungrounded"
    assert pools.outcome(_log(sources=["a.md"], faithfulness="7")) == "answered"
    assert pools.outcome(_log(sources=["a.md"])) == "answered", "unjudged stays where it was"
    assert pools.outcome(_log(faithfulness="0")) == "unsupported_answer", "no source is not this"


def test_a_row_is_read_against_the_ceiling_it_recorded_not_the_one_configured_now():
    # `config.get("max_hops") or default` sent a recorded zero to the default
    from types import SimpleNamespace

    from evals import pools

    def row(recorded_max_hops, hops):
        return SimpleNamespace(
            metrics={"outcome": "error", "hops": hops, "config": {"max_hops": recorded_max_hops}},
            question=None, answer="a", sources=[], faithfulness=None,
        )

    assert pools.outcome(row(2, 2)) == "exhausted"
    assert pools.outcome(row(9, 2)) == "error", "two hops of nine is not exhaustion"
    # a row that recorded none is read against the default, which is what the default is for
    assert pools.outcome(row(None, 99)) == "exhausted"
    # the case `or` could not express, unreachable through the door today
    assert pools.outcome(row(0, 0)) == "exhausted"
