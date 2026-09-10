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


def test_a_residency_is_lent_only_by_the_engine_that_holds_it(db, judging):
    # 10.09: `arc5_floor_d_ollama` carries 2533, which is the number of a vLLM pass
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
    # 10.09: ollama lost the card, pass 2551 judged on the cpu, and the next one on the card took 2551
    _judged(db, "ollama", 2533, T1)
    _judged(db, "ollama", 2551, T3, on_card=False)

    assert judging._last_residency("ollama") == (2533, T1)
    _judged(db, "ollama", 2552, T3, on_card=None)
    assert judging._last_residency("ollama") == (2533, T1), "not read is not the same as whole"
