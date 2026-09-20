import pytest
from evals import grade_curve


def _verdict(address, said, p):
    return {"key": address, "text": f'{{"relevant": "{said}"}}', "p": p}


def _measurement(rows, form="per_chunk", top=2):
    return {"form": form, "top": top, "rows": rows}


def _frozen(rows):
    return {"rows": rows}


def _candidate(address, rank, score):
    return {"address": address, "rank": rank, "rerank_score": score}


def test_the_score_of_a_verdict_puts_a_confident_no_below_a_hesitant_yes():
    assert grade_curve.score_of(_verdict("a#0", "yes", 0.6)) == 0.6
    assert grade_curve.score_of(_verdict("a#0", "no", 0.99)) < 0.5


def test_a_cut_drops_a_yes_the_model_was_not_sure_about():
    row = {"addresses": ["a#0", "a#1"], "verdicts": [_verdict("a#0", "yes", 0.99),
                                                     _verdict("a#1", "yes", 0.6)]}
    assert grade_curve.kept_by_grader(row, None, "per_chunk") == {0, 1}
    assert grade_curve.kept_by_grader(row, 0.7, "per_chunk") == {0}


def test_an_unreadable_verdict_keeps_the_chunk_the_way_the_live_grader_keeps_it():
    row = {"addresses": ["a#0"], "verdicts": [{"key": "a#0", "text": "not json", "p": None}]}
    assert grade_curve.kept_by_grader(row, 0.95, "per_chunk") == {0}


def test_the_whole_text_form_keeps_or_drops_the_whole_context():
    row = {"addresses": ["a#0", "a#1"], "verdicts": [_verdict("whole", "no", 0.99)]}
    assert grade_curve.kept_by_grader(row, None, "whole_text") == set()


def test_only_the_arm_counts_and_the_shares_are_the_rows_own():
    # a chunk outside the top of the arm is not the arm's to keep or to drop
    frozen = _frozen([{"id": 1, "candidates": [_candidate("gold#0", 1, 0.9),
                                               _candidate("far#0", 2, 0.1),
                                               _candidate("far#1", 3, 0.2)]}])
    rows = [{
        "question_id": 1,
        "classes": [grade_curve.gold_classes.GOLD, grade_curve.gold_classes.STRANGER,
                    grade_curve.gold_classes.STRANGER],
        "addresses": ["gold#0", "far#0", "far#1"],
        "verdicts": [_verdict("gold#0", "yes", 0.99), _verdict("far#0", "no", 0.99),
                     _verdict("far#1", "yes", 0.99)],
    }]
    at_none = grade_curve.curve(_measurement(rows), frozen)["grader"][0]
    assert at_none["gold_any"]["point"] == 1.0
    assert at_none["strangers_dropped"]["point"] == 1.0


def test_a_row_without_the_gold_section_in_the_arm_is_not_in_the_denominator():
    frozen = _frozen([{"id": 1, "candidates": [_candidate("far#0", 1, 0.1)]}])
    rows = [{"question_id": 1, "classes": [grade_curve.gold_classes.STRANGER],
             "addresses": ["far#0"], "verdicts": [_verdict("far#0", "no", 0.99)]}]
    at_none = grade_curve.curve(_measurement(rows), frozen)["grader"][0]
    assert at_none["gold_any"]["n"] == 0
    assert at_none["strangers_dropped"]["n"] == 1


def test_no_cut_keeps_what_the_node_in_service_keeps_even_when_the_word_came_out_unlikely():
    # a yes said at 0.36 is a yes: the as-said point is the serving node, not the 0.5 cut
    row = {"addresses": ["a#0"], "verdicts": [_verdict("a#0", "yes", 0.36)]}
    assert grade_curve.kept_by_grader(row, None, "per_chunk") == {0}
    assert grade_curve.kept_by_grader(row, 0.5, "per_chunk") == set()


def test_the_score_is_renormalised_when_the_record_carries_both_words():
    # the grammar masks everything else, so the remainder of a `no` is not the mass of `yes`
    verdict = {**_verdict("a#0", "no", 0.6), "top": {"yes": 0.3, "no": 0.6}}
    assert grade_curve.score_of(verdict) == pytest.approx(0.3 / 0.9)


def test_the_curve_reads_a_verdict_with_the_stands_own_reader():
    # a copy of the reader crashed on a scalar and kept a bare `no` the serving node drops
    bare = {"key": "a#0", "text": "No", "p": 0.8}
    assert grade_curve.kept_by_grader({"addresses": ["a#0"], "verdicts": [bare]}, None,
                                      "per_chunk") == set()
    scalar = {"key": "a#0", "text": '"yes"', "p": 0.8}
    assert grade_curve.kept_by_grader({"addresses": ["a#0"], "verdicts": [scalar]}, None,
                                      "per_chunk") == {0}
