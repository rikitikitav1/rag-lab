from types import SimpleNamespace

from evals import pools


def _log(kind=None, marked=None, answer="the corpus says hello", sources=(), metrics=None,
         faithfulness=None):
    return SimpleNamespace(
        question=SimpleNamespace(kind=kind, marked_sources=marked or []),
        question_text="the corpus says what",
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


def test_a_settled_outcome_is_trusted_and_an_unsettled_one_is_still_derived():
    # the judge settles it, because groundedness is unknowable when the answer is written
    settled = _log(metrics={"outcome": "answered", "settled_outcome": "answered_ungrounded"},
                   sources=["a.md"], faithfulness="7")
    assert pools.outcome(settled) == "answered_ungrounded", "the record wins over the derivation"
    assert pools.settled(settled) is True

    # no stamp means a row written before this existed, and it keeps being read exactly as before
    stale = _log(metrics={"outcome": "answered"}, sources=["a.md"], faithfulness="0")
    assert pools.outcome(stale) == "answered_ungrounded", "derived, so no recorded verdict moves"
    assert pools.settled(stale) is False


def test_a_row_without_its_own_ceiling_is_not_judged_by_todays_config(monkeypatch):
    # the ceiling has only ever been 4; pinning it means moving the config never rewrites history
    import config

    old = _log(answer="", metrics={"outcome": "error", "hops": 4})
    assert pools.outcome(old) == "exhausted"
    monkeypatch.setattr(config.settings.agent, "max_hops", 9)
    assert pools.outcome(old) == "exhausted", "history keeps the ceiling it actually ran under"


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
        question_text="the corpus says what", metrics={}, answered=True,
        answer="the corpus says hello",
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


def _question(**over):
    from types import SimpleNamespace

    base = dict(kind=None, marked_sources=[], reference_answer=None, language="en",
                source_question_id=None, set_name="s")
    return SimpleNamespace(**{**base, **over})


def test_the_inventory_counts_what_each_axis_needs_before_a_pass_is_spent():
    # an hour of card went on two pools whose reference answers were zero, and the set knew
    from evals.question_sets import _of

    out = _of([
        _question(marked_sources=["a.md"], reference_answer="ref"),
        _question(marked_sources=["b.md"]),
        _question(kind="off_domain"),
    ])

    assert out["questions"] == 3
    assert out["pools"] == {"in_corpus": 2, "off_domain": 1}
    assert out["with_marked_sources"] == 2
    assert out["with_reference_answer"] == 1


def test_the_pool_rule_has_one_holder_for_a_row_and_for_a_question():
    # the inventory asks it of a question, `split` asks it of a log, and they parted once already
    from types import SimpleNamespace

    from evals import pools

    question = _question(marked_sources=["a.md"])
    assert pools.kind(SimpleNamespace(question=question)) == pools.kind_of_question(question)
    assert pools.kind_of_question(_question(kind="rejected")) == "rejected"


def test_the_stored_refusal_says_exactly_what_the_report_would_say():
    # the judge read a raw key only the agent wrote and the report re-derived from the text
    import outcomes

    names, prefixes = ("search_corpus",), ("web__",)
    cases = [
        "I cannot answer this from the corpus",
        "the corpus has nothing on that",
        outcomes.NO_RESULTS,
        "A middleware records the method and the url of each request.",
        "",
        None,
        "I will call search_corpus with the query logging",
    ]
    for text in cases:
        report_says = outcomes.classify(text or "", True, names, prefixes) == outcomes.Outcome.refused
        assert outcomes.reads_as_refusal(text, names, prefixes) == report_says, text


def test_both_answering_paths_record_the_refusal_fact():
    # the judge abstained on agent rows and judged the same refusal on single_shot ones
    import inspect

    from use_cases import agent, chat

    for module in (agent, chat):
        source = inspect.getsource(module)
        assert '"refusal": outcomes.reads_as_refusal(' in source, module.__name__
