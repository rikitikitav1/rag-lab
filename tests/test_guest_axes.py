from types import SimpleNamespace

import engines
from conftest import stub_engine
from evals import guest_axes


def _row(**over):
    base = dict(
        id=1, answered=True, answer="a", contexts=["c"], question_text="q", metrics={},
        question=SimpleNamespace(reference_answer="ref"),
    )
    return SimpleNamespace(**{**base, **over})


def _session_of(ql):
    class _Session:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def execute(self, statement, params=None):
            return None

        def get(self, model, ident, **kw):
            return ql

        def commit(self):
            pass

    return _Session


def test_what_an_axis_needs_is_spelled_once_for_the_row_and_once_for_the_query():
    # two spellings of materiality drifted apart once already, for our own three axes
    from job_handlers import judging

    for axis, guest in guest_axes.AXES.items():
        for name in guest.needs:
            assert name in guest_axes.HAS, f"{axis} needs {name}, the row cannot check it"
            assert name in judging.GUEST_MATERIAL, f"{axis} needs {name}, the query cannot ask it"


def test_an_abstention_is_a_verdict_and_ends_the_debt():
    # `nan` is what the standard answers when it has nothing to judge, and it answered
    metrics = {"ragas_faithfulness": {"score": None, "abstained": True}}
    assert "ragas_faithfulness" not in guest_axes.owed(_row(), metrics)
    assert "ragas_faithfulness" in guest_axes.owed(_row(), {})


def test_a_row_without_the_material_is_never_owed_the_axis():
    bare = _row(question=SimpleNamespace(reference_answer=None))
    owed = guest_axes.owed(bare, {})
    assert "ragas_faithfulness" in owed
    assert "ragas_context_precision" not in owed and "ragas_context_recall" not in owed
    # relevancy asks whether the answer fits the question, so it is the one guest that needs no context
    assert guest_axes.owed(_row(contexts=None), {}) == ("ragas_answer_relevancy",)
    assert guest_axes.owed(_row(answer=None, contexts=None), {}) == ()


def test_a_guest_that_throws_records_the_try_and_moves_on(monkeypatch):
    from job_handlers import judging

    def boom(axis, ql):
        raise RuntimeError("ragas is not installed here")

    monkeypatch.setattr(guest_axes, "score", boom)
    ql = _row(metrics={"relevance": {"reason": "ok"}})
    monkeypatch.setattr(judging, "Session", _session_of(ql))
    assert judging._score_guests(1, {}) is False
    assert ql.metrics["relevance"] == {"reason": "ok"}
    assert ql.metrics["ragas_faithfulness"]["attempts"] == 1
    assert "RuntimeError" in ql.metrics["ragas_faithfulness"]["error"]


def test_a_lost_card_on_a_guest_ends_the_pass_and_writes_no_error_into_the_row(monkeypatch):
    import pytest
    from engines.card import CardNotHanded
    from job_handlers import judging

    def lost(axis, ql):
        raise CardNotHanded("ollama did not let go of the card in 60s")

    monkeypatch.setattr(guest_axes, "score", lost)
    ql = _row(metrics={"relevance": {"reason": "ok"}})
    monkeypatch.setattr(judging, "Session", _session_of(ql))
    with pytest.raises(CardNotHanded):
        judging._score_guests(1, {})
    assert "ragas_faithfulness" not in ql.metrics


def test_a_guest_stops_being_tried_after_the_same_cap_our_axes_have(monkeypatch):
    from job_handlers import judging

    tried = []
    monkeypatch.setattr(guest_axes, "score", lambda axis, ql: tried.append(axis) or {})
    ql = _row(metrics={axis: {"attempts": judging._MAX_JUDGE_ATTEMPTS} for axis in guest_axes.NAMES})
    monkeypatch.setattr(judging, "Session", _session_of(ql))
    judging._score_guests(1, {})
    assert tried == []


def test_the_judging_pass_never_asks_for_a_guest():
    # by request only: the guests are a pass of their own, and no run waits on them
    from job_handlers import judging
    from job_handlers.base import HANDLERS

    assert not _named(judging.still_to_judge()) & set(guest_axes.NAMES)
    assert "judge_guest_axes" in HANDLERS
    assert judging.judge_guest_axes is HANDLERS["judge_guest_axes"]


def test_the_guest_pass_asks_only_for_rows_that_owe_a_guest():
    from job_handlers import judging
    from sqlalchemy import or_

    assert set(guest_axes.NAMES) <= _named(or_(*judging.guest_clauses()))


def test_a_guest_pass_that_cannot_score_says_so_instead_of_walking(monkeypatch):
    import pytest
    from job_handlers import judging

    monkeypatch.setattr(judging, "guests_available", lambda: False)
    with pytest.raises(ValueError, match="ragas"):
        judging.judge_guest_axes({"run_name": "r"})


def test_a_guest_number_says_at_which_width_and_on_what_card_it_was_taken(monkeypatch):
    from job_handlers import judging

    monkeypatch.setattr(guest_axes, "score", lambda axis, ql: {"score": 1.0, "abstained": False})
    monkeypatch.setattr(judging, "judge_on_card", lambda: False)
    ql = _row()
    monkeypatch.setattr(judging, "Session", _session_of(ql))
    judging._score_guests(1, {"seed": 0, "width": 4})
    written = ql.metrics["ragas_faithfulness"]
    assert written["width"] == 4 and written["seed"] == 0 and written["score"] == 1.0
    assert written["on_card_at_this_row"] is False, "the row's own reading, beside the pass's"


def test_a_guest_number_names_the_process_that_took_it(monkeypatch):
    # `ragas` is in neither image: until it is, every guest number comes from a host process
    from evals import guest_llm

    monkeypatch.setattr(guest_llm.llm, "resolve_name", lambda role: "q:7b")
    stamp = guest_llm.stamp()
    assert stamp["runtime"] in ("host", "container")
    assert stamp["ragas"] and stamp["model"] == "q:7b"


def test_the_card_is_read_from_the_one_holder_that_already_answers_it(monkeypatch):
    # a pass 9.6x as long as ours can lose the judge to a neighbour halfway
    from job_handlers import judging

    monkeypatch.setattr(judging.llm, "resolve", lambda role: engines.Resolved("q:7b", stub_engine()))
    monkeypatch.setattr(
        judging.card.ollama, "residency", lambda spec=None: [{"model": "q:7b", "size_mb": 100, "vram_mb": 100}]
    )
    assert judging.judge_on_card() is True

    monkeypatch.setattr(
        judging.card.ollama, "residency", lambda spec=None: [{"model": "q:7b", "size_mb": 100, "vram_mb": 40}]
    )
    assert judging.judge_on_card() is False

    monkeypatch.setattr(judging.card.ollama, "residency", lambda spec=None: [])
    assert judging.judge_on_card() is None


def test_a_guest_pass_sweeps_itself_and_queues_nothing(monkeypatch):
    # the container has the handler and not `ragas`: a queued sweep lands there and dies
    import inspect

    from job_handlers import judging

    source = inspect.getsource(judging.judge_guest_axes)
    assert "job_queue.enqueue" not in source
    assert "_MAX_SWEEPS" in source
    assert not hasattr(judging, "_sweep_guests_again")


def test_the_door_refuses_where_the_library_is_missing(monkeypatch):
    # a queued job that cannot score would spend three sweeps proving it
    from fastapi.testclient import TestClient
    from job_handlers import judging
    from server import app

    monkeypatch.setattr(judging, "guests_available", lambda: False)
    with TestClient(app) as client:
        answer = client.post("/v1/eval/guest-axes", json={"run_name": "r"})
    assert answer.status_code == 409
    assert "ragas" in answer.json()["detail"]


def test_a_copy_of_a_run_carries_no_guest_score():
    from use_cases import rejudge

    sql = str(rejudge.copy_statement("a", "b").compile(compile_kwargs={"literal_binds": True}))
    for axis in guest_axes.NAMES:
        assert f"'{axis}'" in sql, f"a copy would inherit {axis} from the run it copies"


# a json path arrives as a tuple, and the axis name is the head of it
def _named(clause) -> set:
    found = set()
    for value in clause.compile().params.values():
        found.update(value if isinstance(value, tuple) else [value])
    return {v for v in found if isinstance(v, str)}


class _FakeSession:
    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False


def test_every_writer_takes_the_lock_and_no_reader_holds_it_through_a_model_call():
    # the guest held the row through a minute of calls, and our own judge starved on a 5s timeout
    import ast
    import inspect
    from pathlib import Path

    from job_handlers import judging

    source = (Path(inspect.getfile(judging))).read_text()
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.FunctionDef):
            continue
        body = ast.get_source_segment(source, node) or ""
        writes = "ql.metrics = " in body
        locked = "with_for_update=True" in body
        assert writes == locked, f"{node.name}: writes={writes} locks={locked}"
        if locked:
            assert "guest_axes.score" not in body and "judge.axis" not in body, (
                f"{node.name} holds the row through a model call"
            )


def test_the_door_refuses_a_run_that_owes_nothing_and_one_too_large(monkeypatch):
    import limits
    from fastapi.testclient import TestClient
    from job_handlers import judging
    from server import app

    monkeypatch.setattr(judging, "guests_available", lambda: True)
    monkeypatch.setattr(judging, "guest_rows_of", lambda run: 0)
    with TestClient(app) as client:
        assert client.post("/v1/eval/guest-axes", json={"run_name": "typo"}).status_code == 404

    monkeypatch.setattr(judging, "guest_rows_of", lambda run: limits.MAX_GUEST_ROWS + 1)
    with TestClient(app) as client:
        answer = client.post("/v1/eval/guest-axes", json={"run_name": "huge"})
    assert answer.status_code == 400 and "cap" in answer.json()["detail"]


def test_every_material_predicate_is_total_so_its_negation_counts_the_rest():
    # the cheap half; `scripts/materiality_totality.py` proves it on rows, where NULL lives
    from job_handlers import judging

    for name, clause in judging.GUEST_MATERIAL.items():
        rendered = str(clause)
        assert "coalesce" in rendered.lower(), f"{name}: a NULL column reaches the comparison"
        assert "jsonb_array_length" not in rendered, f"{name}: raises on a jsonb scalar"
    for name in ("answer", "question_text"):
        assert not guest_axes.HAS[name](_row(**{name: ""}))
    assert not guest_axes.HAS["contexts"](_row(contexts=[]))
    assert not guest_axes.HAS["contexts"](_row(contexts=None))


def test_every_axis_declares_the_question_it_is_handed():
    # all three build `user_input` from it, and none of them named it
    for axis, guest in guest_axes.AXES.items():
        assert "question_text" in guest.needs, axis
    assert not guest_axes.HAS["question_text"](_row(question_text=None))


def test_a_json_null_context_is_not_material_rather_than_an_error():
    # 375 rows hold jsonb `null` there, and the pass survived only on the planner's filter order
    from job_handlers import judging

    assert "jsonb_typeof" in str(judging.GUEST_MATERIAL["contexts"])
    assert not guest_axes.HAS["contexts"](_row(contexts=None))


def test_the_debt_is_one_statement_over_one_population_and_says_whether_it_adds_up():
    # counts stood over four populations, cost 22 round trips, and nothing checked the sum
    from evals import run_debts

    columns = run_debts._columns()
    assert "population" in columns and "ours" in columns
    for axis in guest_axes.AXES:
        assert {f"owed__{axis}", f"answered__{axis}", f"unreachable__{axis}"} <= set(columns)
    seen = {}

    def _totals(session, run_name, cols):
        seen["n"] = len(cols)
        return dict.fromkeys(cols, 0)

    import evals.run_debts as rd

    original, rd._totals = rd._totals, _totals
    try:
        out = rd.of("whatever")
    finally:
        rd._totals = original
    assert seen["n"] == len(columns), "the debt asks the database more than once"
    assert out["guests"]["ragas_faithfulness"]["reconciles"] is True
    # the fourth bucket: an axis given up on after three failures fell out of the other three
    assert "given_up_on" in out["guests"]["ragas_faithfulness"]


def test_the_judge_door_refuses_a_run_that_owes_it_nothing(monkeypatch):
    # `/rejudge` judges a copy; a run judged in place had no door and the queue was filled by hand
    from evals import run_debts
    from fastapi.testclient import TestClient
    from server import app

    monkeypatch.setattr(run_debts, "of", lambda run: {"answered_rows": 0, "ours_still_to_judge": 0})
    with TestClient(app) as client:
        assert client.post("/v1/eval/judge", json={"run_name": "ghost"}).status_code == 404

    monkeypatch.setattr(
        run_debts, "of", lambda run: {"answered_rows": 5, "ours_still_to_judge": 0}
    )
    with TestClient(app) as client:
        answer = client.post("/v1/eval/judge", json={"run_name": "done"})
    assert answer.status_code == 404 and "owes our judge nothing" in answer.json()["detail"]


def test_a_run_owes_a_third_debt_and_it_is_whether_the_replay_can_drive_it():
    # `control_a` recorded no transcript at all, and only reading the file said so
    import evals.run_debts as rd

    columns = rd._columns()
    assert {"replay__replayable", "replay__no_turns", "replay__weak_unrecorded"} <= set(columns)

    def _totals(session, run_name, cols):
        return dict.fromkeys(cols, 0) | {
            "population": 10, "replay__replayable": 6,
            "replay__no_turns": 3, "replay__weak_unrecorded": 1,
        }

    original, rd._totals = rd._totals, _totals
    try:
        out = rd.of("whatever")
    finally:
        rd._totals = original

    assert out["replay"] == {
        "replayable": 6, "no_turns": 3, "weak_drop_unrecorded": 1, "another_arm": 0,
        "reconciles": True,
    }


def test_the_replay_debt_says_so_when_its_parts_do_not_cover_the_population():
    # the same live guard the guest counts carry: a predicate answering NULL drops rows silently
    import evals.run_debts as rd

    def _totals(session, run_name, cols):
        return dict.fromkeys(cols, 0) | {"population": 10, "replay__replayable": 6}

    original, rd._totals = rd._totals, _totals
    try:
        out = rd.of("whatever")
    finally:
        rd._totals = original

    assert out["replay"]["reconciles"] is False


def test_a_cancelled_judging_job_stops_instead_of_running_to_the_end(monkeypatch):
    # the console marked the row cancelled and the pass judged on: 7 hours of card were at stake
    import job_queue
    from job_handlers import judging

    judged, swept = [], []
    monkeypatch.setattr(job_queue, "is_cancelled", lambda id: len(judged) >= 2)
    monkeypatch.setattr(judging, "_judge_log", lambda log_id, **kw: judged.append(log_id) or True)
    monkeypatch.setattr(judging, "_bench_from", lambda o: SimpleNamespace(
        model=None, template=lambda p: "t"))
    monkeypatch.setattr(judging, "require_role_ready", lambda role, **kw: None)
    monkeypatch.setattr(judging, "require_card", lambda role, model=None, asked_by=None: None)
    monkeypatch.setattr(judging, "_refuse_a_second_judge", lambda run_name, model: None)
    monkeypatch.setattr(judging, "_target_log_ids", lambda s, o: [1, 2, 3, 4, 5])
    monkeypatch.setattr(judging.experiment, "revive_for_run", lambda r: None)
    monkeypatch.setattr(judging, "_sweep_again_if_rows_are_still_owed",
                        lambda o, r: swept.append(r))
    monkeypatch.setattr(judging.experiment, "try_aggregate_for_run", lambda r: None)
    monkeypatch.setattr(judging.Session, "__enter__", lambda self: None, raising=False)

    judging.judge_answers({"run_name": "r", "_job_id": 77})

    assert judged == [1, 2], "it kept judging after the cancellation"
    assert swept == [], "a sweep after a cancellation queues the work the cancellation stopped"


def test_the_guest_pass_draws_a_seeded_subsample_and_redraws_the_same_rows():
    # guests calibrate on a subsample, they cost 35x ours a row
    from job_handlers.judging import _drawn

    ids = list(range(100, 200))
    first = _drawn(ids, 20, 7)
    assert len(first) == 20 and first == sorted(first)
    assert first == _drawn(list(reversed(ids)), 20, 7), "a sweep would redraw other rows"
    assert _drawn(ids, 20, 8) != first, "the seed does nothing"
    assert _drawn(ids, None, 0) == ids and _drawn(ids, 500, 0) == ids


def test_the_guest_budget_is_drawn_once_and_a_sweep_stays_inside_it():
    # `sample=50` scored up to 150: every sweep called the drawing function again
    import job_handlers.judging as j

    drawn = j._drawn(list(range(1, 301)), 100, 0)
    assert len(drawn) == 100
    budget = set(drawn)
    # what the second sweep sees: every row still owing, unsampled, filtered to the budget
    still_owed = list(range(1, 301))
    second = [i for i in still_owed if i in budget]
    assert set(second) <= budget and len(second) == 100


def test_our_judge_never_took_a_subsample():
    # a replace over a pattern that appeared twice put the guests' budget into the judge's rows
    import inspect

    import job_handlers.judging as j

    assert "_drawn" not in inspect.getsource(j._target_log_ids)
    assert inspect.getsource(j).count("def _drawn") == 1


def test_the_guard_does_not_read_false_on_an_axis_given_up_on():
    # three failures close an axis, and it was in none of owed, answered or unreachable
    import evals.run_debts as rd

    def _totals(session, run_name, cols):
        return dict.fromkeys(cols, 0) | {
            "population": 10, "owed__ragas_faithfulness": 6,
            "answered__ragas_faithfulness": 1, "unreachable__ragas_faithfulness": 0,
            "capped__ragas_faithfulness": 3,
        }

    original, rd._totals = rd._totals, _totals
    try:
        out = rd.of("whatever")
    finally:
        rd._totals = original

    axis = out["guests"]["ragas_faithfulness"]
    assert axis["given_up_on"] == 3 and axis["reconciles"] is True


def test_the_guest_pass_scores_with_the_session_closed():
    # the lock went weeks ago and the session stayed, holding a connection through minutes of calls
    import inspect

    from job_handlers import judging

    body = inspect.getsource(judging._score_guests)
    with_block, after = body.split("guest_axes.carried(ql)", 1)
    assert "guest_axes.score" in after and "guest_axes.score" not in with_block


def test_what_a_guest_reads_off_a_row_survives_the_session():
    from types import SimpleNamespace

    from evals import guest_axes

    row = SimpleNamespace(
        question_text="q", answer="a", contexts=["c"],
        question=SimpleNamespace(reference_answer="ref"),
    )
    carried = guest_axes.carried(row)
    assert (carried.question_text, carried.answer, carried.contexts) == ("q", "a", ["c"])
    assert carried.question.reference_answer == "ref"
    assert guest_axes.carried(SimpleNamespace(
        question_text=None, answer=None, contexts=None, question=None
    )).question.reference_answer is None
