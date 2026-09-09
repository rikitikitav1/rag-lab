import pytest


@pytest.fixture(scope="module")
def costs():
    from evals import language_cost

    return language_cost


def _row(question_id, faith, answer="text", log_id=1):
    from types import SimpleNamespace

    return SimpleNamespace(id=log_id, question_id=question_id, answer=answer,
                           faithfulness=str(faith), relevance=None, completeness=None)


def test_a_question_twice_in_one_arm_refuses_rather_than_pairing_arbitrarily(costs, monkeypatch):
    # a dict would have kept whichever row came last, and the pair would depend on the loader
    monkeypatch.setattr(costs, "in_corpus_and_answered", lambda ql: True)
    with pytest.raises(ValueError):
        costs._pairs([_row(1, 5), _row(1, 6)], [_row(1, 7)])


def test_the_quoted_cut_is_read_off_the_arm_before_the_change(costs, monkeypatch):
    # the published cut was selected by the outcome; this one is visible without seeing the result
    monkeypatch.setattr(costs, "answered_in_target", lambda ql: ql.answer != "off")
    before, after = _row(1, 5, answer="off"), _row(1, 8, answer="off")

    assert costs._in_group(costs.DECLARED_FROM_BEFORE, before, after) is True
    # the answer never changed language, so the cut the outcome selects does not hold this pair
    assert costs._in_group(costs.SELECTED_BY_OUTCOME, before, after) is False


def test_an_axis_no_pair_carries_says_so_instead_of_averaging_nothing(costs, monkeypatch):
    monkeypatch.setattr(costs, "in_corpus_and_answered", lambda ql: True)
    pairs = costs._pairs([_row(1, 5)], [_row(1, 8)])
    got = costs._over(pairs)

    assert got["faithfulness"]["mean_delta"] == 3.0 and got["faithfulness"]["better"] == 1
    assert "unreadable" in got["relevance"]


def test_an_arm_against_itself_is_the_floor_and_says_so(costs):
    # `measure` would have reported one residency and a delta of zero as if it compared two things
    with pytest.raises(ValueError):
        costs.measure("one_run", "one_run")
