from types import SimpleNamespace

import engines
import passes
import pytest
from engines import card
from errors import Final
from models.registry import EngineKind, Placement, Role

OLLAMA = engines.EngineSpec(1, "ollama", EngineKind.ollama, "OLLAMA", Placement.gpu)


def _seated_in_turn(monkeypatch, *names):
    left = iter(names)
    monkeypatch.setattr(passes.llm, "resolve_for", lambda role, model=None: engines.Resolved(next(left), OLLAMA))


def test_a_role_reseated_mid_pass_stops_it_for_good(monkeypatch):
    # the role is read on every call, and a gemma cancelled mid-row left a row scored by two models
    _seated_in_turn(monkeypatch, "qwen2.5:7b", "qwen2.5:7b", "gemma2:9b")
    monkeypatch.setattr(passes, "_read_spills", lambda checked, allow_spill: {})
    walk = passes.Pass(None, (passes.Seat(Role.ragas),))
    assert walk.before_row()
    with pytest.raises(passes.SeatChanged, match="gemma2:9b") as stopped:
        walk.before_row()
    assert isinstance(stopped.value, Final)


def test_a_seat_nobody_could_read_is_not_a_move(monkeypatch):
    def unread(role, model=None):
        raise ConnectionError("database away")

    monkeypatch.setattr(passes.llm, "resolve_for", unread)
    walk = passes.Pass(None, (passes.Seat(Role.judging),))
    assert walk.before_row() and walk.before_row()


def test_a_cancelled_pass_reads_no_seat_and_says_stop(monkeypatch):
    monkeypatch.setattr(passes, "_resolve", lambda seat: engines.Resolved("m", OLLAMA))
    monkeypatch.setattr(passes.job_queue, "is_cancelled", lambda job_id: True)
    monkeypatch.setattr(passes, "_read_spills", lambda checked, allow_spill: pytest.fail("a cancelled pass read the card"))
    walk = passes.Pass(7, (passes.Seat(Role.judging),))
    assert walk.cancelled() and not walk.before_row()


def test_a_model_the_first_call_loads_is_read_for_a_spill_from_the_second_row(monkeypatch):
    # ollama loads on the first call, so before it the lazy model is not there to read
    read = []
    monkeypatch.setattr(passes, "_resolve", lambda seat: engines.Resolved("m", OLLAMA))
    monkeypatch.setattr(passes, "_read_spills", lambda checked, allow_spill: read.append([s.role for s, _ in checked]) or {})
    walk = passes.Pass(None, (passes.Seat(Role.generation), passes.Seat(Role.embedding, spill_from=1),
                              passes.Seat(Role.paraphrasing, spill_from=None)))
    walk.before_row()
    walk.before_row()
    assert read == [[Role.generation], [Role.generation, Role.embedding]]


def test_a_model_half_on_the_processor_stops_the_pass_unless_the_cpu_is_meant(monkeypatch):
    # gemma2:9b scored an axis at 15% on the processor, and the row only wrote `false` beside the score
    monkeypatch.setattr(passes.llm, "resolve_for", lambda role, model=None: engines.Resolved("gemma2:9b", OLLAMA))
    monkeypatch.setattr(passes.card, "model_on_card", lambda spec, name: False)
    with pytest.raises(card.CardNotHanded, match="not whole on the card: ragas=gemma2:9b"):
        passes.refuse_spill([passes.Seat(Role.ragas)], allow_spill=False)
    passes.refuse_spill([passes.Seat(Role.ragas)], allow_spill=True)


def test_a_pass_that_owed_rows_and_did_none_stops_for_good(monkeypatch):
    # a judge and a guest whose every row failed ended `done`, as a run with no answer once did
    monkeypatch.setattr(passes, "_resolve", lambda seat: None)
    walk = passes.Pass(None, (passes.Seat(Role.judging),))
    with pytest.raises(passes.NothingDone, match="0 of 3") as stopped:
        walk.close(owed=3, done=0)
    assert isinstance(stopped.value, Final)
    walk.close(owed=3, done=1)
    walk.close(owed=0, done=0)
    monkeypatch.setattr(passes.job_queue, "is_cancelled", lambda job_id: True)
    passes.Pass(7, (passes.Seat(Role.judging),)).close(owed=3, done=0)


def test_the_residency_is_read_once_and_only_when_first_asked(monkeypatch):
    monkeypatch.setattr(passes, "_resolve", lambda seat: None)
    reads = []
    walk = passes.Pass(None, (passes.Seat(Role.judging),), residency=lambda: reads.append(1) or SimpleNamespace(on_card=True))
    assert reads == []
    walk.get()
    walk.get()
    assert reads == [1]


def test_a_judging_pass_stops_when_the_judge_is_reseated_mid_pass(monkeypatch):
    import job_queue
    from job_handlers import judging

    vllm = engines.EngineSpec(3, "vllm", EngineKind.vllm, "VLLM", Placement.gpu)
    seats = iter([engines.Resolved("qwen", vllm), engines.Resolved("qwen", vllm), engines.Resolved("qwen2.5:7b", OLLAMA)])
    judged = []
    monkeypatch.setattr(passes, "_resolve", lambda seat: next(seats))
    monkeypatch.setattr(passes, "_read_spills", lambda checked, allow_spill: {})
    monkeypatch.setattr(job_queue, "is_cancelled", lambda id: False)
    monkeypatch.setattr(judging, "_judge_log", lambda log_id, **kw: judged.append(log_id) or True)
    monkeypatch.setattr(judging, "_bench_from", lambda o: SimpleNamespace(model=None, template=lambda p: "t"))
    monkeypatch.setattr(judging, "require_role_ready", lambda role, **kw: None)
    monkeypatch.setattr(judging, "require_card", lambda role, model=None: None)
    monkeypatch.setattr(judging, "_refuse_a_second_judge", lambda run_name, model: None)
    monkeypatch.setattr(judging, "_target_log_ids", lambda s, o: [1, 2, 3])
    monkeypatch.setattr(judging.experiment, "revive_for_run", lambda r: None)
    monkeypatch.setattr(judging.Session, "__enter__", lambda self: None, raising=False)
    with pytest.raises(passes.SeatChanged):
        judging.judge_answers({"run_name": "r", "_job_id": 77})
    assert judged == [1]


def test_a_row_stamps_the_card_reading_its_pass_went_on_by(monkeypatch):
    # the guest read the card a second time for its stamp, and the two readings could differ
    reads = iter([True, False])
    monkeypatch.setattr(passes, "_resolve", lambda seat: engines.Resolved("qwen2.5:7b", OLLAMA))
    monkeypatch.setattr(passes.card, "model_on_card", lambda spec, name: next(reads))
    walk = passes.Pass(None, (passes.Seat(Role.ragas),), allow_spill=True)
    walk.before_row()
    assert walk.on_card(Role.ragas) is True
    walk.before_row()
    assert walk.on_card(Role.ragas) is False


def test_a_budget_changed_mid_pass_stops_it_as_a_reseat_does(monkeypatch):
    # the seat compared engine and model, and a new max_tokens answered the next row under the old stamp
    budgets = iter([{}, {}, {"max_tokens": 4096}])
    monkeypatch.setattr(passes.llm, "resolve_for",
                        lambda role, model=None: engines.Resolved("qwen2.5:7b", OLLAMA, options=next(budgets)))
    monkeypatch.setattr(passes, "_read_spills", lambda checked, allow_spill: {})
    walk = passes.Pass(None, (passes.Seat(Role.ragas),))
    assert walk.before_row()
    with pytest.raises(passes.SeatChanged, match="max_tokens"):
        walk.before_row()


def test_the_row_answers_with_what_its_calls_read_once_they_are_in(monkeypatch):
    # the start of the row read a card another engine held, and the call saw the embedder on the cpu
    monkeypatch.setattr(passes, "_resolve", lambda seat: engines.Resolved("bge-m3", OLLAMA))
    monkeypatch.setattr(passes.card, "model_on_card", lambda spec, name: None)
    walk = passes.Pass(None, (passes.Seat(Role.embedding),), allow_spill=True)
    assert walk.before_row() and walk.on_card(Role.embedding) is None
    walk.after_row({"embedding": False})
    assert walk.on_card(Role.embedding) is False


@pytest.mark.parametrize("sweep", [None, 1])
def test_a_sweep_that_judges_none_of_its_leftovers_ends_them_rather_than_failing_the_experiment(monkeypatch, sweep):
    # one row the judge cut at its budget every time failed the sweep, and the experiment with it
    import job_queue
    from job_handlers import judging

    after = []
    monkeypatch.setattr(passes, "_resolve", lambda seat: engines.Resolved("qwen", OLLAMA))
    monkeypatch.setattr(passes, "_read_spills", lambda checked, allow_spill: {})
    monkeypatch.setattr(job_queue, "is_cancelled", lambda id: False)
    monkeypatch.setattr(judging, "_judge_log", lambda log_id, **kw: False)
    monkeypatch.setattr(judging, "_bench_from", lambda o: SimpleNamespace(model=None, template=lambda p: "t"))
    monkeypatch.setattr(judging, "require_role_ready", lambda role, **kw: None)
    monkeypatch.setattr(judging, "require_card", lambda role, model=None: None)
    monkeypatch.setattr(judging, "_refuse_a_second_judge", lambda run_name, model: None)
    monkeypatch.setattr(judging, "_target_log_ids", lambda s, o: [42359])
    monkeypatch.setattr(judging.experiment, "revive_for_run", lambda r: None)
    monkeypatch.setattr(judging.Session, "__enter__", lambda self: None, raising=False)
    monkeypatch.setattr(judging, "_sweep_again_if_rows_are_still_owed", lambda o, r: after.append("sweep"))
    monkeypatch.setattr(judging.experiment, "try_aggregate_for_run", lambda r: after.append("aggregate"))
    options = {"run_name": "r", "_job_id": 77, **({"sweep": sweep} if sweep else {})}
    if sweep is None:
        with pytest.raises(passes.NothingDone, match="0 of 1"):
            judging.judge_answers(options)
        assert after == [], "a first pass that judged nothing is a broken judge"
    else:
        judging.judge_answers(options)
        assert after == ["sweep", "aggregate"]
