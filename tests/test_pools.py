from types import SimpleNamespace

from evals import pools


def _log(kind=None, marked=None, answer="the corpus says hello", sources=(), metrics=None):
    return SimpleNamespace(
        question=SimpleNamespace(kind=kind, marked_sources=marked or []),
        answer=answer,
        sources=[{"source": s} for s in sources],
        metrics=metrics or {},
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


def test_an_answer_without_sources_is_unsupported_not_answered():
    assert pools.outcome(_log(sources=["a.md"])) == "answered"
    assert pools.outcome(_log()) == "unsupported_answer"


def test_only_an_mcp_prefix_counts_as_remote_evidence():
    assert pools.has_remote_evidence(_log(sources=["mcp:deepwiki__ask_question"]))
    assert not pools.has_remote_evidence(_log(sources=["mcp_notes/readme.md"]))
    assert not pools.has_remote_evidence(_log())
