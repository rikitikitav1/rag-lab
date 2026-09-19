import json

import pytest
from evals import grade_candidates


def _rows(n: int) -> list:
    return [{"id": i, "text": f"q{i}", "candidates": []} for i in range(n)]


def test_the_same_rows_come_back_for_the_same_seed():
    # a sample drawn per run would compare two draws instead of two arms
    first = grade_candidates.drawn(_rows(100), 10, 0, None)
    assert [r["id"] for r in first] == [r["id"] for r in grade_candidates.drawn(_rows(100), 10, 0, None)]
    assert [r["id"] for r in first] != [r["id"] for r in grade_candidates.drawn(_rows(100), 10, 7, None)]
    assert first == sorted(first, key=lambda r: r["id"]), "the file's order is kept, the draw is of rows"


def test_a_sample_larger_than_the_file_is_the_file():
    assert len(grade_candidates.drawn(_rows(5), 50, 0, None)) == 5


def test_a_limit_is_a_slice_and_says_so():
    assert [r["id"] for r in grade_candidates.drawn(_rows(9), None, 0, 3)] == [0, 1, 2]


def test_the_population_is_the_union_of_the_two_top_fives():
    row = {"candidates": [
        {"address": f"f#{i}", "rank": i + 1, "rerank_score": 1 / (i + 1)} for i in range(10)
    ]}
    # fusion and the cross-encoder agree here, so the union is five, not ten
    assert len(grade_candidates.population(row, 5)) == 5

    row["candidates"][9]["rerank_score"] = 99
    got = grade_candidates.population(row, 5)
    assert len(got) == 6 and "f#9" in [c["address"] for c in got]


def test_an_unknown_call_form_is_refused_before_the_card_is_taken(tmp_path):
    from errors import StandFault

    path = tmp_path / "frozen.json"
    path.write_text(json.dumps({"stamp": {"variant": "v"}, "rows": []}))
    with pytest.raises(StandFault, match="call form"):
        grade_candidates.run(str(path), form="by_vibes")


def test_the_pass_carries_the_curve_of_both_arms_so_the_number_has_a_job_behind_it():
    frozen = {"rows": [{"id": 1, "candidates": [
        {"address": "gold#0", "rank": 1, "rerank_score": 0.9},
        {"address": "far#0", "rank": 2, "rerank_score": 0.1},
    ]}]}
    payload = {"form": "per_chunk", "top": 2, "rows": [{
        "question_id": 1,
        "classes": ["gold_section", "stranger"],
        "addresses": ["gold#0", "far#0"],
        "verdicts": [{"key": "gold#0", "text": '{"relevant": "yes"}', "p": 0.99},
                     {"key": "far#0", "text": '{"relevant": "no"}', "p": 0.99}],
    }]}
    got = grade_candidates.curves(payload, frozen)
    assert set(got) == {"A", "B"}
    assert got["A"]["grader"][0]["gold_any"]["point"] == 1.0
    assert got["A"]["grader"][0]["strangers_dropped"]["point"] == 1.0


def test_a_named_prompt_version_is_read_instead_of_whatever_is_active(monkeypatch):
    # activating a prompt to measure it makes the serving node use an unjudged one
    from use_cases import grading

    monkeypatch.setattr("prompt_repo.template_of", lambda purpose, version: f"v{version}")
    monkeypatch.setattr("prompt_repo.active_template", lambda purpose: "active")
    assert grading.system_prompt(3) == "v3"
    assert grading.system_prompt() == "active"


def test_a_verdict_row_carries_every_field_the_ask_recorded():
    # `top` was added to the ask and never reached the file, because the row copied three names
    ask = {"stage": "grade", "key": "a#0", "text": '{"relevant": "no"}', "p": 0.6,
           "top": {"yes": 0.4, "no": 0.6}}
    assert grade_candidates.verdict_row(ask) == {
        "key": "a#0", "text": '{"relevant": "no"}', "p": 0.6, "top": {"yes": 0.4, "no": 0.6},
    }
