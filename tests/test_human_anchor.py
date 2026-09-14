import pytest


@pytest.fixture(scope="module")
def anchor():
    from evals import human_anchor

    return human_anchor


def test_a_judge_that_called_the_pair_equal_where_a_human_did_not_has_missed(anchor):
    # an instrument that cannot separate is failing, not abstaining: that is the whole measurement
    assert anchor._verdict(0.0, "A") == "did_not_separate"
    assert anchor._verdict(0.0, "B") == "did_not_separate"
    assert anchor._verdict(0.0, "=") == "agreed"
    assert anchor._verdict(0.3, "=") == "missed"


def test_agreement_reads_the_sign_and_not_the_size(anchor):
    # the human gives a side, never a magnitude, so only the direction can be compared
    assert anchor._verdict(0.3, "A") == "agreed"
    assert anchor._verdict(0.01, "A") == "agreed"
    assert anchor._verdict(-0.3, "A") == "disagreed"
    assert anchor._verdict(-0.3, "B") == "agreed"


def test_an_unfilled_pair_is_not_read_as_an_answer(anchor, tmp_path):
    sheet = tmp_path / "human_anchor_20260908.md"
    sheet.write_text(
        "## Пара 1\nЛучше подкреплён контекстом (впиши A, B или `=`): **A**\n"
        "## Пара 2\nЛучше подкреплён контекстом (впиши A, B или `=`): **____**\n"
        "## Пара 3\nЛучше подкреплён контекстом (впиши A, B или `=`): **=**\n"
        "## Пара 4\nЛучше подкреплён контекстом (впиши A, B или `=`): **что-то не то**\n",
        encoding="utf-8",
    )
    got = anchor._picked(sheet)
    assert got == {1: "A", 2: None, 3: "=", 4: None}


def test_filling_the_sheet_does_not_break_the_lock_that_ties_it_to_its_key(anchor):
    # the lock must catch a rebuilt sheet, and must not catch the owner writing his answers into it
    blank = "## Пара 1\nтекст вопроса\nЛучше подкреплён контекстом (впиши A, B или `=`): **____**\n"
    filled = blank.replace("**____**", "**A**")
    other = blank.replace("текст вопроса", "другой вопрос")

    assert anchor._fingerprint(blank) == anchor._fingerprint(filled)
    assert anchor._fingerprint(blank) != anchor._fingerprint(other)


def test_an_answer_travels_by_the_pair_of_rows_and_not_by_its_number(anchor, tmp_path, monkeypatch):
    # rebuilt lists renumber and may flip sides; an answer carried by number would land on the wrong pair
    import json

    monkeypatch.setattr(anchor, "SHEETS", tmp_path)
    sheet = tmp_path / "human_anchor_20260908.md"
    sheet.write_text(
        "## Пара 1\nЛучше подкреплён контекстом (впиши A, B или `=`): **A**\n"
        "## Пара 2\nЛучше подкреплён контекстом (впиши A, B или `=`): **=**\n"
        "## Пара 3\nЛучше подкреплён контекстом (впиши A, B или `=`): **____**\n",
        encoding="utf-8",
    )
    (tmp_path / "human_anchor_key_20260908.json").write_text(json.dumps({
        "sheets": {"sitting": str(sheet)},
        "pairs": {"sitting": [
            {"n": 1, "A": {"log_id": 11}, "B": {"log_id": 22}},
            {"n": 2, "A": {"log_id": 33}, "B": {"log_id": 44}},
            {"n": 3, "A": {"log_id": 55}, "B": {"log_id": 66}},
        ]},
    }), encoding="utf-8")

    # a pruned answer lives in the done file, and a rebuild that forgot it would ask again
    (tmp_path / "human_anchor_done_20260908.json").write_text(
        json.dumps({"3": {"answer": "B", "A": 55, "B": 66}}), encoding="utf-8")

    kept = anchor._already("20260908")
    assert kept[frozenset((55, 66))] == 66, "a pruned answer is carried too"
    assert kept[frozenset((11, 22))] == 11, "the winner is a row, so a flipped side still resolves"
    assert kept[frozenset((33, 44))] == "="
    assert len(kept) == 3


def test_the_repeats_sheet_never_arrives_carrying_the_answer_it_checks(anchor):
    # it is the same pair of rows flipped, so a carried answer would agree with itself perfectly
    from types import SimpleNamespace

    def row(log_id, answer):
        return SimpleNamespace(id=log_id, answer=answer, question_text="q", contexts=["c"])

    left, right = row(1, "ответ по-русски"), row(2, "an answer in english")
    kept = {frozenset((1, 2)): 1}
    main = anchor._sheet([(left, right, 1)], "основной", "нота", kept)
    repeats = anchor._sheet([(right, left, 1)], "повторы", "нота", None)

    assert "**A**" in main, "the main sheet keeps what he already answered"
    assert "**____**" in repeats and "**A**" not in repeats and "**B**" not in repeats


def test_a_pruned_pair_leaves_the_sheet_but_not_the_count(anchor, tmp_path, monkeypatch):
    # the sheet shrinks so the next sitting is only what is left; the answer moves, it does not vanish
    import json

    monkeypatch.setattr(anchor, "SHEETS", tmp_path)
    sheet = tmp_path / "human_anchor_20260908.md"
    body = "шапка\n"
    for n, mark in ((1, "A"), (2, "____"), (3, "B")):
        body += f"\n## Пара {n}\nтекст\nЛучше подкреплён контекстом (впиши A, B или `=`): **{mark}**\n"
    sheet.write_text(body, encoding="utf-8")
    (tmp_path / "human_anchor_key_20260908.json").write_text(json.dumps({
        "population": "тестовая",
        "sheets": {"sitting": str(sheet)},
        "pairs": {"sitting": [
            {"n": 1, "A": {"log_id": 11}, "B": {"log_id": 22}, "ours": 0.5, "guest": 0.5},
            {"n": 2, "A": {"log_id": 33}, "B": {"log_id": 44}, "ours": 0.5, "guest": 0.5},
            {"n": 3, "A": {"log_id": 55}, "B": {"log_id": 66}, "ours": 0.5, "guest": 0.5},
        ]},
    }), encoding="utf-8")

    assert anchor.prune("20260908")["left"] == 1
    left = sheet.read_text(encoding="utf-8")
    assert "## Пара 2" in left and "## Пара 1" not in left and "## Пара 3" not in left
    assert anchor.read("20260908")["pairs_answered"] == 2, "pruned answers still count"


def test_the_order_of_the_list_does_not_depend_on_the_answers(anchor):
    # declared before he opens it, so stopping early is not a choice made after seeing the rows
    from types import SimpleNamespace

    def row(log_id, question, ours, guest):
        return SimpleNamespace(
            id=log_id, question_id=question, faithfulness=str(ours), contexts=["c"],
            metrics={"ragas_faithfulness": {"score": guest}},
        )

    # q1 the judges oppose, q2 they agree, q3 ours cannot separate at all
    rows = [
        row(1, "q1", 10, 0.1), row(2, "q1", 2, 0.9),
        row(3, "q2", 10, 0.9), row(4, "q2", 2, 0.1),
        row(5, "q3", 10, 0.5), row(6, "q3", 10, 0.1),
    ]
    order = anchor._ordered(rows)
    assert [(a.question_id) for a, _ in order] == ["q1", "q2", "q3"]


def test_the_covariates_a_pair_carries_are_the_ones_the_reading_buckets_by(anchor, monkeypatch):
    # `build` wrote them and `mark` wrote them again, and only one of the two copies got fixed
    from types import SimpleNamespace

    monkeypatch.setattr(anchor, "_language", lambda text: "ru" if text == "по-русски" else "en")
    monkeypatch.setattr(anchor, "_hit_gold", lambda ql: False)
    left = SimpleNamespace(id=1, run_name="a", answer="по-русски", question_text="q")
    right = SimpleNamespace(id=2, run_name="b", answer="in english", question_text="q")

    got = anchor._covariates(left, right)
    assert got["cross_language"] is True
    assert set(got) >= {"cross_language", "template_leak", "neither_hit_gold", "A", "B"}


def test_a_missing_key_is_refused_by_name_and_not_by_a_traceback(anchor, tmp_path, monkeypatch):
    # the library is imported by a job now, and `SystemExit` from a library kills the worker
    monkeypatch.setattr(anchor, "SHEETS", tmp_path)
    with pytest.raises(anchor.Refused):
        anchor.read("20260908")
    with pytest.raises(anchor.Refused):
        anchor.read("nonsense")


def test_the_letter_on_the_sheet_and_the_row_it_points_at_are_one_mapping(anchor):
    # three copies of this lived in the module, and one of them was written inverted
    sides = {"A": 11, "B": 22}
    for letter in ("A", "B", "="):
        assert anchor._letter_picked(sides, anchor._row_picked(sides, letter)) == letter
    assert anchor._row_picked(sides, "B") == 22, "B is the right side, and it was once the left"


def test_a_pruned_answer_lands_on_its_own_pair_and_not_on_the_pair_that_took_its_number(
    anchor, tmp_path, monkeypatch
):
    # `read` joined the done file by pair number while every other reader joined by the rows
    import json

    monkeypatch.setattr(anchor, "SHEETS", tmp_path)
    (tmp_path / "human_anchor_20260908.md").write_text(
        "## Пара 1\nЛучше подкреплён контекстом (впиши A, B или `=`): **____**\n",
        encoding="utf-8",
    )
    # pair 1 today is a different pair of rows than the pair 1 that was answered and pruned
    (tmp_path / "human_anchor_key_20260908.json").write_text(json.dumps({
        "population": "p",
        "pairs": {"sitting": [
            {"n": 1, "A": {"log_id": 30}, "B": {"log_id": 40}, "ours": 0.5, "guest": 0.5},
        ]},
    }), encoding="utf-8")
    (tmp_path / "human_anchor_done_20260908.json").write_text(
        json.dumps({"1": {"answer": "A", "A": 10, "B": 20}}), encoding="utf-8"
    )

    got = anchor.read("20260908")
    assert got["pairs_answered"] == 0, "an answer about rows 10 and 20 is not an answer about 30/40"
    assert got["ours"]["n"] == 0 and got["ours"]["agreed"] == 0


def test_a_rebuild_refuses_to_walk_over_a_filled_repeats_sheet(anchor, tmp_path, monkeypatch):
    # the main sheet carries answers forward and this one must not, so a rebuild can only lose them
    monkeypatch.setattr(anchor, "SHEETS", tmp_path)
    monkeypatch.setattr(anchor, "_rows", lambda: [])
    (tmp_path / "human_anchor_repeats_20260908.md").write_text(
        "## Пара 1\nЛучше подкреплён контекстом (впиши A, B или `=`): **B**\n", encoding="utf-8"
    )
    with pytest.raises(anchor.Refused):
        anchor.build("20260908")


def test_a_rebuild_refuses_when_an_answered_pair_falls_off_the_list(anchor, tmp_path, monkeypatch):
    # the sheet promises a rebuild keeps answers, and it kept only the pairs that stayed in the top
    import json

    monkeypatch.setattr(anchor, "SHEETS", tmp_path)
    (tmp_path / "human_anchor_20260908.md").write_text(
        "## Пара 1\nЛучше подкреплён контекстом (впиши A, B или `=`): **A**\n", encoding="utf-8"
    )
    (tmp_path / "human_anchor_key_20260908.json").write_text(json.dumps({
        "population": "p",
        "pairs": {"sitting": [{"n": 1, "A": {"log_id": 10}, "B": {"log_id": 20}}]},
    }), encoding="utf-8")
    # the new order holds neither row, so the answer about 10 against 20 has nowhere to land
    monkeypatch.setattr(anchor, "_rows", lambda: [])

    with pytest.raises(anchor.Refused):
        anchor.build("20260908")


def test_a_cyrillic_letter_that_looks_like_a_is_read_as_a(anchor, tmp_path):
    # the sheet is Russian and so is the layout: pair 6 of the repeats came back as U+0410
    sheet = tmp_path / "human_anchor_20260908.md"
    sheet.write_text(
        "## Пара 1\nЛучше подкреплён контекстом (впиши A, B или `=`): **А**\n"
        "## Пара 2\nЛучше подкреплён контекстом (впиши A, B или `=`): **В**\n"
        "## Пара 3\nЛучше подкреплён контекстом (впиши A, B или `=`): **Ы**\n",
        encoding="utf-8",
    )
    assert anchor._picked(sheet) == {1: "A", 2: "B", 3: None}


def test_a_new_judge_is_read_off_the_copies_of_the_rows_the_sheet_names():
    # the sheet names rows of two arc 3 runs; a rejudge writes its verdicts onto copies with new ids
    from evals import human_anchor

    pairs = [{"n": 1, "A": {"log_id": 10, "run": "left"}, "B": {"log_id": 20, "run": "right"}},
             {"n": 2, "A": {"log_id": 11, "run": "left"}, "B": {"log_id": 21, "run": "right"}}]
    asked = {10: 100, 20: 100, 11: 101, 21: 101}
    copied = {("left", 100): "9", ("right", 100): "4", ("left", 101): "5"}
    deltas = human_anchor._deltas_from(pairs, asked, copied)
    assert deltas[1] == 0.5
    assert deltas[2] is None, "a side the copies did not judge is not read as a tie"
