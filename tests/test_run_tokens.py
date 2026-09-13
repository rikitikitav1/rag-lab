from evals import run_tokens


def _entry(engine, model, prompt, completion, calls):
    return {"engine": engine, "model": model, "prompt": prompt, "completion": completion, "calls": calls}


def _axis(prompt, completion="absent"):
    stamp = {"score": 8, "judge_prompt_tokens": prompt}
    return stamp if completion == "absent" else {**stamp, "judge_completion_tokens": completion}


def test_spent_sums_the_jobs_and_the_price_comes_from_the_rows_each_on_its_own():
    # a retried answer paid twice and its row carries one call: the two numbers are not the same thing
    jobs = [
        ({"generation": [_entry("gonka", "mm", 3000, 1600, 2)], "embedding": [_entry("ollama", "bge-m3", 16, 0, 1)]}, "done"),
        ({"judging": [_entry("vllm", "qwen", 3880, 233, 3)]}, "done"),
        ({}, "done"),
        (None, "done"),
        (None, "running"),
    ]
    rows = [
        (1425, 804, {"faithfulness": _axis(1200, 61), "relevance": _axis(1300, 69), "completeness": _axis(1380, 103)}),
        (None, None, {}),
    ]
    got = run_tokens.summarize(jobs, rows)
    assert got["spent"]["generation"] == [_entry("gonka", "mm", 3000, 1600, 2)]
    assert got["spent_by_engine"] == {
        "gonka": {"prompt": 3000, "completion": 1600, "calls": 2},
        "ollama": {"prompt": 16, "completion": 0, "calls": 1},
        "vllm": {"prompt": 3880, "completion": 233, "calls": 3},
    }
    assert got["jobs"] == {"counted": 3, "before_the_count": 1, "not_finished": 1}
    generation = got["per_question"]["generation"]
    assert (generation["rows_counted"], generation["rows_missing"], generation["prompt_per_row"]) == (1, 1, 1425.0)
    judging = got["per_question"]["judging"]
    assert (judging["rows_counted"], judging["prompt"], judging["completion"]) == (1, 3880, 233)


def test_a_verdict_without_its_output_count_is_a_named_gap_and_not_a_zero():
    rows = [
        (10, 5, {"faithfulness": _axis(100), "relevance": _axis(100, 7)}),
        (10, 5, {"faithfulness": _axis(100, 9)}),
    ]
    judging = run_tokens.summarize([], rows)["per_question"]["judging"]
    assert (judging["rows_counted"], judging["rows_missing"], judging["completion"]) == (1, 1, 9)


def test_a_guest_row_is_priced_by_its_own_stamp():
    tokens = {"judging": [_entry("vllm", "qwen", 900, 40, 2)], "embedding": [_entry("ollama", "bge-m3", 30, 0, 1)]}
    rows = [
        (10, 5, {"ragas_faithfulness": {"score": 0.5, "tokens": tokens}}),
        (10, 5, {"ragas_faithfulness": {"score": 0.4}}),
        (10, 5, {}),
    ]
    ragas = run_tokens.summarize([], rows)["per_question"]["ragas"]
    assert (ragas["rows_counted"], ragas["rows_missing"], ragas["prompt"], ragas["completion"]) == (1, 1, 930, 40)
    assert run_tokens.summarize([], [])["spent"] is None
