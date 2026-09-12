# the residency rule lives in sql, so a stubbed `_last_residency` cannot say whether it is right
import json

import pytest
from real_db import pytestmark  # noqa: F401
from sqlalchemy import text
from sqlalchemy.orm import sessionmaker

T0 = "2026-09-09T20:00:00+00:00"
T1 = "2026-09-09T21:00:00+00:00"
# microseconds are dropped by `isoformat` when they are zero, so the two spellings meet here
T2 = "2026-09-09T21:00:00.500000+00:00"
T3 = "2026-09-09T22:00:00+00:00"


@pytest.fixture
def judging(db, monkeypatch):
    from job_handlers import judging

    with db.connect() as c:
        c.execute(text("TRUNCATE question_logs"))
        c.execute(text("DELETE FROM model_roles"))
        c.execute(text("DELETE FROM models"))
        c.execute(text("DELETE FROM engines"))
        for name, kind, placement in (
            ("ollama", "ollama", "gpu"), ("vllm", "vllm", "gpu"),
            ("ollama-cpu", "ollama", "cpu"), ("vllm-cpu", "vllm", "cpu"),
        ):
            c.execute(text(
                "INSERT INTO engines (name, kind, env_prefix, placement)"
                " VALUES (:n, :k, :p, :pl)"
            ), {"n": name, "k": kind, "p": name.upper().replace("-", "_"), "pl": placement})
    monkeypatch.setattr(judging, "Session", sessionmaker(db))
    return judging


def _judged(db, engine, residency, at, started=None, on_card=True):
    stamp = {"residency_id": residency, "engine_name": engine, "judged_at": at, "on_card": on_card}
    if started:
        stamp["engine_added"] = {"started_at": started}
    with db.connect() as c:
        c.execute(
            text("INSERT INTO question_logs (answered, metrics) VALUES (true, CAST(:m AS jsonb))"),
            {"m": json.dumps({"faithfulness": stamp})},
        )


def test_a_run_is_not_finished_by_a_judge_on_another_engine(db, judging, monkeypatch):
    # three verdicts of a `vllm-cpu` run were once given by the judge on the card
    import engines
    from job_handlers.base import Final
    from models.registry import EngineKind, Placement

    with db.connect() as c:
        for run, engine in (("cpu_run", "vllm-cpu"), ("cpu_run", None), ("fresh_run", None)):
            stamp = {"relevance": {"engine_name": engine}} if engine else {}
            c.execute(text("INSERT INTO question_logs (run_name, answered, metrics)"
                           " VALUES (:r, true, CAST(:m AS jsonb))"), {"r": run, "m": json.dumps(stamp)})
    judge = {"name": "vllm"}
    monkeypatch.setattr(judging.llm, "resolve_for", lambda role, model=None: engines.Resolved(
        "Qwen/Q", engines.EngineSpec(3, judge["name"], EngineKind.vllm, "VLLM", Placement.gpu)))
    with pytest.raises(Final, match="judged on vllm-cpu and the judge now sits on vllm"):
        judging._refuse_a_second_judge("cpu_run", None)
    judging._refuse_a_second_judge("fresh_run", None)
    judge["name"] = "vllm-cpu"
    judging._refuse_a_second_judge("cpu_run", None)


def test_a_residency_is_lent_only_by_the_engine_that_holds_it(db, judging):
    # `arc5_floor_d_ollama` carries 2533, which is the number of a vLLM pass
    _judged(db, "ollama", 2526, T1)
    _judged(db, "vllm", 2533, T3)

    assert judging._last_residency("ollama") == (2526, T1)
    assert judging._last_residency("vllm") == (2533, T3)
    assert judging._last_residency("ollama-cpu") is None, "no row of its own, nothing to inherit"


def test_a_verdict_from_another_gpu_engine_says_the_card_changed_hands(db, judging):
    # a shell unloaded ollama for vLLM and back; the queue saw only judging, which keeps the judge
    _judged(db, "vllm", 2520, T0)
    _judged(db, "ollama", 2526, T1)
    _judged(db, "ollama-cpu", 2544, T2)
    assert judging._card_changed_hands(T1, "ollama") is False, "before it, or off the card"

    _judged(db, "vllm", 2533, T3)
    assert judging._card_changed_hands(T1, "ollama") is True
    assert judging._card_changed_hands(T3, "ollama") is False, "nothing judged after the last"
    assert judging._card_changed_hands(None, "ollama") is True, "too old to vouch for anything"


def test_a_vllm_residency_is_lent_only_under_the_same_process_start(db, judging):
    s1, s2 = "2026-09-09T19:00:00+00:00", "2026-09-09T21:30:00+00:00"
    # as the real rows are: vLLM has no `/api/ps`, so `on_card` is null and must not decide here
    _judged(db, "vllm", 2530, T1, started=s1, on_card=None)
    _judged(db, "vllm", 2531, T3, started=s2, on_card=None)
    _judged(db, "vllm-cpu", 2542, T3, started=s1, on_card=None)

    assert judging._last_residency("vllm", s1) == (2530, T1)
    assert judging._last_residency("vllm", s2) == (2531, T3)
    assert judging._last_residency("vllm", "2026-09-10T00:00:00+00:00") is None, "a new process"
    assert judging._last_residency("vllm-cpu", s2) is None, "the same start on another engine"


def test_a_pass_that_found_the_judge_on_the_cpu_lends_nothing_to_the_card(db, judging):
    # ollama lost the card, pass 2551 judged on the cpu, and the next one on the card took 2551
    _judged(db, "ollama", 2533, T1)
    _judged(db, "ollama", 2551, T3, on_card=False)

    assert judging._last_residency("ollama") == (2533, T1)
    _judged(db, "ollama", 2552, T3, on_card=None)
    assert judging._last_residency("ollama") == (2533, T1), "not read is not the same as whole"


def test_the_engines_that_can_hold_the_card_are_found_not_assumed(judging, monkeypatch, db):
    import engines
    from engines import lookup
    from models.registry import EngineKind

    monkeypatch.setattr(lookup, "Session", sessionmaker(db))
    assert [e.name for e in engines.card_engines()] == ["ollama", "vllm"]
    assert [e.name for e in engines.card_engines(EngineKind.vllm)] == ["vllm"]


def test_the_bootstrap_fills_the_quant_of_vllm_rows_from_their_weights(judging, monkeypatch, db):
    import bootstrap

    with db.connect() as c:
        for name, engine in (("Qwen/Q-AWQ", "vllm"), ("qwen2.5:7b", "ollama")):
            c.execute(text(
                "INSERT INTO models (name, engine_id, status)"
                " SELECT :n, id, 'ready' FROM engines WHERE name = :e"
            ), {"n": name, "e": engine})
    from engines import lookup
    from job_handlers import model_ops

    # the boot fills a row through the pull's own record, which opens its own session
    for module in (bootstrap, model_ops, lookup):
        monkeypatch.setattr(module, "Session", sessionmaker(db))
    monkeypatch.setattr(bootstrap.vllm, "weights_check", lambda repo: [])
    monkeypatch.setattr(
        bootstrap.vllm, "artifact_of",
        lambda repo: {"quant": "AWQ" if repo == "Qwen/Q-AWQ" else "ASKED", "size_bytes": 5},
    )
    bootstrap._fill_vllm_rows()
    with db.connect() as c:
        seen = dict(c.execute(text("SELECT name, quant FROM models")).all())
    assert seen == {"Qwen/Q-AWQ": "AWQ", "qwen2.5:7b": None}, "ollama rows are ollama's"


def test_the_last_residency_is_the_last_judged_not_the_last_created(db, judging):
    # run A created, run B created, B judged, A judged: the next pass must see A, not B
    _judged(db, "ollama", 2601, T3)
    _judged(db, "ollama", 2602, T1)
    assert judging._last_residency("ollama") == (2601, T3)


def test_the_queue_runs_the_generation_first_and_judges_in_one_batch(db, monkeypatch):
    # jobs take the card in their own turn: the run first, the judges back to back, only the chat queues
    import job_queue

    monkeypatch.setattr(job_queue, "Session", sessionmaker(db))
    rows = [
        ("judge_answers", {}, "-3 minutes", "-3 minutes"),
        ("eval_run", {}, "-2 minutes", "-2 minutes"),
        ("judge_answers", {"late": 1}, "-1 minutes", "-1 minutes"),
        # asked by the chat, which has nothing to wait with: it overtakes everything that waits
        ("hand_card", {"engine_id": 1, "asked_by": "chat"}, "0 minutes", "0 minutes"),
        # queued long ago and deferred since: its turn has come whatever its priority
        ("judge_language", {}, "0 minutes", "-40 minutes"),
    ]
    with db.connect() as c:
        c.execute(text("TRUNCATE jobs"))
        for kind, options, since, created in rows:
            c.execute(text(
                "INSERT INTO jobs (type, options, apply_since, created_at)"
                " VALUES (:t, CAST(:o AS jsonb), now() + CAST(:s AS interval),"
                " now() + CAST(:c AS interval))"
            ), {"t": kind, "o": json.dumps(options), "s": since, "c": created})
    order = []
    while (claimed := job_queue.claim_next(["default"])) is not None:
        order.append((claimed.type, claimed.options.get("asked_by")))
    # the chat's handover waited behind a queued run
    assert order == [
        ("hand_card", "chat"),
        ("judge_language", None),
        ("eval_run", None),
        ("judge_answers", None),
        ("judge_answers", None),
    ]


def test_a_second_ollama_on_the_cpu_does_not_unseat_the_seeded_one(judging, monkeypatch, db):
    # with `ollama-cpu` registered the bootstrap skipped reconciling the seeded ollama
    import engines
    from engines import lookup

    monkeypatch.setattr(lookup, "Session", sessionmaker(db))
    assert engines.seeded_ollama().name == "ollama"


def test_a_clean_stand_seats_the_judge_on_the_model_vllm_serves(judging, monkeypatch, db):
    import bootstrap
    import config
    from engines import lookup

    session = sessionmaker(db)
    monkeypatch.setattr(bootstrap, "Session", session)
    monkeypatch.setattr(lookup, "Session", session)
    monkeypatch.setattr(bootstrap.model_acceptance, "refuse_unfit_model",
                        lambda role, name, engine_id: None)
    cfg = type("Role", (), {"model": "Qwen/Q-AWQ", "engine": "vllm", "options": {}})()
    monkeypatch.setattr(config.settings.llm, "roles", {"judging": cfg})

    # compose and the config name the model twice; a server serving another seats nothing
    monkeypatch.setattr(bootstrap.vllm, "served", lambda spec: ["Qwen/Other"])
    bootstrap._ensure_roles(None)
    with db.connect() as c:
        assert c.execute(text("SELECT count(*) FROM model_roles")).scalar() == 0

    monkeypatch.setattr(bootstrap.vllm, "served", lambda spec: ["Qwen/Q-AWQ"])
    bootstrap._ensure_roles(None)
    with db.connect() as c:
        seated = c.execute(text(
            "SELECT m.name, e.name, m.status FROM model_roles r JOIN models m ON m.id = r.model_id"
            " JOIN engines e ON e.id = m.engine_id WHERE r.role = 'judging'"
        )).one()
    assert tuple(seated) == ("Qwen/Q-AWQ", "vllm", "ready")


def test_a_boot_seats_a_role_whose_engine_is_stopped_or_unprobed(judging, monkeypatch, db):
    # the default seats the reranker on `vllm-rerank`, stopped unless its profile is up
    import bootstrap
    import config
    from engines import lookup

    session = sessionmaker(db)
    monkeypatch.setattr(bootstrap, "Session", session)
    monkeypatch.setattr(lookup, "Session", session)
    monkeypatch.setattr(bootstrap.vllm, "served", lambda spec: ["Qwen/Q-AWQ"])
    cfg = type("Role", (), {"model": "Qwen/Q-AWQ", "engine": "vllm", "options": {}})()
    monkeypatch.setattr(config.settings.llm, "roles", {"judging": cfg})
    for fault in (bootstrap.model_acceptance.EngineDown("vllm does not answer"),
                  bootstrap.model_acceptance.NeedsProbe("vllm is asleep")):
        def refuse(role, name, engine_id, fault=fault):
            raise fault

        monkeypatch.setattr(bootstrap.model_acceptance, "refuse_unfit_model", refuse)
        with db.connect() as c:
            c.execute(text("DELETE FROM model_roles"))
            c.commit()
        bootstrap._ensure_roles(None)
        with db.connect() as c:
            assert c.execute(text("SELECT count(*) FROM model_roles")).scalar() == 1, fault


def test_a_seat_asked_before_a_later_choice_does_not_undo_it(judging, monkeypatch, db):
    # a seat queued behind a run must not overwrite a role chosen after it
    from models.registry import Role
    from use_cases import model_acceptance

    monkeypatch.setattr(model_acceptance, "Session", sessionmaker(db))
    with db.connect() as c:
        ids = dict(c.execute(text(
            "INSERT INTO models (name, engine_id, status) SELECT n, id, 'ready' FROM engines,"
            " (VALUES ('Qwen/Q'), ('llama')) v(n) WHERE engines.name = 'vllm' RETURNING name, id"
        )).all())
        c.execute(text("INSERT INTO model_roles (role, model_id) VALUES ('generation', :m)"),
                  {"m": ids["llama"]})
        c.commit()
        vllm_id = c.execute(text("SELECT id FROM engines WHERE name = 'vllm'")).scalar()

    with pytest.raises(ValueError, match="reassigned"):
        model_acceptance.seat(Role.generation, vllm_id, "Qwen/Q", over=None)
    model_acceptance.seat(Role.generation, vllm_id, "Qwen/Q", over=ids["llama"])
    model_acceptance.seat(Role.generation, vllm_id, "Qwen/Q", over=ids["llama"])
    with db.connect() as c:
        held = c.execute(text("SELECT model_id FROM model_roles WHERE role = 'generation'")).scalar()
    assert held == ids["Qwen/Q"], "the second of two identical asks finds it done"


def test_weights_another_row_of_the_same_kind_names_are_not_deleted(judging, monkeypatch, db):
    # ollama and ollama-cpu share a volume, and the vLLMs share one HF cache
    from job_handlers import model_ops

    monkeypatch.setattr(model_ops, "Session", sessionmaker(db))
    with db.connect() as c:
        for name, engine in (("bge-m3", "ollama"), ("bge-m3:latest", "ollama-cpu"),
                             ("gemma2:9b", "ollama"), ("gemma2:9b", "vllm"),
                             ("Qwen/Q", "vllm"), ("Qwen/Q", "vllm-cpu")):
            c.execute(text("INSERT INTO models (name, engine_id, status)"
                           " SELECT :n, id, 'ready' FROM engines WHERE name = :e"),
                      {"n": name, "e": engine})
        c.commit()
        ids = dict(c.execute(text("SELECT name, id FROM engines")).all())

    with pytest.raises(ValueError, match="shares its weights with bge-m3:latest on ollama-cpu"):
        model_ops.refuse_if_the_weights_are_shared("bge-m3", ids["ollama"])
    with pytest.raises(ValueError, match="on vllm-cpu"):
        model_ops.refuse_if_the_weights_are_shared("Qwen/Q", ids["vllm"])
    model_ops.refuse_if_the_weights_are_shared("gemma2:9b", ids["ollama"])
    # a vLLM repo and an ollama tag are two stores, however alike the names
    with db.connect() as c:
        c.execute(text("DELETE FROM models WHERE engine_id = :e"), {"e": ids["vllm-cpu"]})
        c.commit()
    model_ops.refuse_if_the_weights_are_shared("Qwen/Q", ids["vllm"])


def test_the_bootstrap_pulls_vllm_weights_that_are_absent_or_broken(judging, monkeypatch, db):
    import bootstrap

    with db.connect() as c:
        # one repository on two vLLMs is hashed once, not once per row
        for name, engine in (("Qwen/Intact", "vllm"), ("Qwen/Broken", "vllm"),
                             ("Qwen/Intact", "vllm-cpu")):
            c.execute(text(
                "INSERT INTO models (name, engine_id, status, quant)"
                " SELECT :n, id, 'available', 'AWQ' FROM engines WHERE name = :e"
            ), {"n": name, "e": engine})
    queued = []
    from engines import lookup
    from job_handlers import model_ops

    # the boot fills a row through the pull's own record, which opens its own session
    for module in (bootstrap, model_ops, lookup):
        monkeypatch.setattr(module, "Session", sessionmaker(db))
    checked = []

    def check(repo):
        checked.append(repo)
        return [] if repo == "Qwen/Intact" else ["model.safetensors"]

    monkeypatch.setattr(bootstrap.vllm, "weights_check", check)
    monkeypatch.setattr(bootstrap.vllm, "artifact_of", lambda repo: {"size_bytes": 1})
    monkeypatch.setattr(bootstrap.job_queue, "pending_of_type", lambda t, **o: False)
    monkeypatch.setattr(bootstrap.job_queue, "enqueue", lambda t, o, **kw: queued.append((t, o)))
    bootstrap._fill_vllm_rows()
    with db.connect() as c:
        seen = set(c.execute(text("SELECT name, status FROM models")).all())
    assert seen == {("Qwen/Intact", "ready"), ("Qwen/Broken", "loading")}
    assert [(t, o["name"]) for t, o in queued] == [("pull_llm_model", "Qwen/Broken")]
    assert sorted(checked) == ["Qwen/Broken", "Qwen/Intact"], "each repository hashed once"
    with db.connect() as c:
        size = c.execute(text("SELECT size_bytes FROM models WHERE name = 'Qwen/Intact'")).scalar()
    assert size == 1, "a quant typed by hand no longer leaves the size empty"


def test_live_answers_gather_in_one_waiting_job_and_never_join_a_running_one(db, monkeypatch):
    # the judge on vLLM would wake once per chat question; the batch wakes it once per five minutes
    import job_queue

    monkeypatch.setattr(job_queue, "Session", sessionmaker(db))
    with db.connect() as c:
        c.execute(text("TRUNCATE jobs"))
    first = job_queue.judge_live(101)
    assert job_queue.judge_live(102) == first, "a second answer joins the waiting batch"
    with db.connect() as c:
        options, waits = c.execute(text(
            "SELECT options, apply_since > now() + interval '4 minutes' FROM jobs WHERE id = :i"
        ), {"i": first}).one()
        assert options == {"log_ids": [101, 102], "live": True} and waits
        c.execute(text("UPDATE jobs SET status = 'running' WHERE id = :i"), {"i": first})
    third = job_queue.judge_live(103)
    assert third != first, "a running job read its rows already, so the answer starts a new batch"
    with db.connect() as c:
        kept = c.execute(text("SELECT options FROM jobs WHERE id = :i"), {"i": first}).scalar()
    assert kept["log_ids"] == [101, 102]
