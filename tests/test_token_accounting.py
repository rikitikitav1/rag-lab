from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import engines
import job_queue
import llm
import pytest
from models.registry import EngineKind, Placement

LOCAL = engines.EngineSpec(1, "ollama", EngineKind.ollama, "OLLAMA", Placement.gpu)
CLOUD = engines.EngineSpec(8, "gonka", EngineKind.openai_compatible, "GONKA", Placement.remote)


def _entry(prompt, completion, calls, **more):
    return {"engine": "ollama", "model": "m", "prompt": prompt, "completion": completion, "calls": calls, **more}


def test_a_call_counts_into_every_open_scope_and_only_a_carried_pool_thread_counts():
    # the judge's pool threads start with an empty context and would count into nothing
    with llm.accounting() as job:
        with llm.accounting() as row:
            llm._count("judging", LOCAL, "m", 10, 3)
        with ThreadPoolExecutor(2) as pool:
            list(pool.map(llm.carried(lambda i: llm._count("judging", LOCAL, "m", 1, 1)), range(4)))
            list(pool.map(lambda i: llm._count("judging", LOCAL, "m", 100, 100), range(2)))
    assert row.record() == {"judging": [_entry(10, 3, 1, max_prompt=10)]}
    assert job.record() == {"judging": [_entry(14, 7, 5, max_prompt=10)]}
    assert llm.Tally().record() is None, "a job that called nothing writes nothing"


def _embedder(monkeypatch, usage):
    reply = SimpleNamespace(data=[SimpleNamespace(embedding=[0.1])], usage=usage)
    client = SimpleNamespace(embeddings=SimpleNamespace(create=lambda **kw: reply))
    monkeypatch.setattr(llm.engines, "client_for", lambda spec: client)


def test_an_embedding_counts_its_tokens_and_a_local_one_without_them_is_a_named_gap(monkeypatch):
    _embedder(monkeypatch, SimpleNamespace(prompt_tokens=12))
    with llm.accounting() as job:
        llm._embeddings(engines.Resolved("m", LOCAL), ["a", "b"], "embedding")
    assert job.record() == {"embedding": [_entry(12, 0, 1, max_prompt=12)]}
    _embedder(monkeypatch, None)
    with llm.accounting() as job:
        llm._embeddings(engines.Resolved("m", LOCAL), ["a"], "embedding")
    assert job.record() == {"embedding": [_entry(0, 0, 1, uncounted=1, max_prompt=0)]}
    # on a cloud the tokens are the quota
    with pytest.raises(llm.NoUsage):
        llm._embeddings(engines.Resolved("m", CLOUD), ["a"], "embedding")


def test_attempts_add_up_per_engine_and_model():
    first = {"judging": [_entry(10, 3, 1)]}
    second = {"judging": [_entry(5, 2, 1), {**_entry(7, 1, 1), "engine": "vllm"}], "embedding": [_entry(4, 0, 2)]}
    assert job_queue.merged_tokens(first, second) == {
        "judging": [_entry(15, 5, 2), {**_entry(7, 1, 1), "engine": "vllm"}],
        "embedding": [_entry(4, 0, 2)],
    }
    assert job_queue.merged_tokens(None, first) == first


def test_the_worker_writes_what_a_failed_job_spent(monkeypatch):
    import worker

    written, failed = [], []

    def handler(options):
        llm._count("generation", CLOUD, "m", 30, 9)
        raise worker.Final("broker said no")

    monkeypatch.setitem(worker.HANDLERS, "spender", handler)
    monkeypatch.setattr(worker.job_queue, "claim_next", lambda queues: job_queue.ClaimedJob(id=5, type="spender", options={}))
    monkeypatch.setattr(worker.job_specs, "check", lambda *a, **kw: None)
    monkeypatch.setattr(worker.job_queue, "fail", lambda id, error, elapsed=None: failed.append(id))
    monkeypatch.setattr(worker.job_queue, "add_tokens", lambda id, record: written.append((id, record)))
    assert worker.run_once(["default"])
    assert failed == [5]
    assert written == [(5, {"generation": [{"engine": "gonka", "model": "m", "prompt": 30, "completion": 9, "calls": 1,
                                           "max_prompt": 30}]})]


def test_a_cloud_call_carries_its_run_as_the_cache_key_and_a_local_one_is_left_alone(monkeypatch):
    # a broker answered a repeated body from its cache, and a second pass would have measured the cache
    import contextlib

    seen = []
    reply = SimpleNamespace(choices=[], usage=None)

    def create(**kw):
        seen.append(kw.get("user"))
        return reply

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    monkeypatch.setattr(llm.engines, "client_for", lambda spec: client)
    monkeypatch.setattr(llm, "_card_for", lambda spec, name: contextlib.nullcontext())
    with llm.cache_keyed("arm_a"):
        llm._complete(CLOUD, "m", [], {})
        llm._complete(LOCAL, "m", [], {})
        with ThreadPoolExecutor(1) as pool:
            list(pool.map(llm.carried(lambda i: llm._complete(CLOUD, "m", [], {})), [0]))
    llm._complete(CLOUD, "m", [], {})
    assert seen == ["arm_a", None, "arm_a", None]
    assert (llm.cache_key_of(CLOUD), llm.cache_key_of(LOCAL)) == ("user=run_name", None)


def test_the_worker_keys_every_call_of_a_job_by_its_run(monkeypatch):
    import worker

    keyed = []
    monkeypatch.setitem(worker.HANDLERS, "keyed", lambda options: keyed.append(llm._cache_key.get()))
    monkeypatch.setattr(worker.job_queue, "claim_next",
                        lambda queues: job_queue.ClaimedJob(id=6, type="keyed", options={"run_name": "arm_b"}))
    monkeypatch.setattr(worker.job_specs, "check", lambda *a, **kw: None)
    monkeypatch.setattr(worker.job_queue, "complete", lambda id, elapsed=None: None)
    monkeypatch.setattr(worker.job_queue, "add_tokens", lambda id, record: None)
    assert worker.run_once(["default"])
    assert keyed == ["arm_b"] and llm._cache_key.get() is None


def test_the_run_snapshot_says_a_cloud_role_was_keyed_by_the_run(monkeypatch):
    # a reader of an old run must know whether its second pass could have come from the cache
    from models.registry import Role
    from use_cases import run_snapshot

    monkeypatch.setattr(run_snapshot.llm, "resolve", lambda role: engines.Resolved("bge-m3", LOCAL))
    monkeypatch.setattr(run_snapshot.llm, "sampler", lambda role, spec: engines.Sampler({}, {}))
    monkeypatch.setattr(run_snapshot.card, "model_on_card", lambda spec, name: None)
    *_, cache_keys, _ = run_snapshot._by_role(engines.Resolved("m", CLOUD))
    assert cache_keys == {Role.generation: "user=run_name"}
    assert "cache_keys" in run_snapshot.KEYS and run_snapshot.SCHEMA == 11


def test_a_finished_job_has_its_count_before_it_reads_done(monkeypatch):
    # the count was written after the status, and a reader polling for done found it empty
    import worker

    events = []

    def handler(options):
        llm._count("judging", LOCAL, "m", 5, 1)

    monkeypatch.setitem(worker.HANDLERS, "counted", handler)
    monkeypatch.setattr(worker.job_queue, "claim_next", lambda queues: job_queue.ClaimedJob(id=7, type="counted", options={}))
    monkeypatch.setattr(worker.job_specs, "check", lambda *a, **kw: None)
    monkeypatch.setattr(worker.job_queue, "complete", lambda id, elapsed=None: events.append("done"))
    monkeypatch.setattr(worker.job_queue, "add_tokens", lambda id, record: events.append("tokens"))
    assert worker.run_once(["default"])
    assert events == ["tokens", "done"]


def test_a_count_keeps_the_longest_input_and_the_calls_the_output_limit_cut():
    # the guest's reasoning ran past 1024 tokens on 15 of 113 calls, and only the log said so
    with llm.accounting() as job:
        llm._count("ragas", LOCAL, "m", 6403, 1024, "length")
        llm._count("ragas", LOCAL, "m", 900, 200, "stop")
    assert job.record() == {"ragas": [{**_entry(7303, 1224, 2), "model": "m", "max_prompt": 6403, "cut_by_length": 1}]}
    first = {"ragas": [{**_entry(900, 200, 1), "max_prompt": 900}]}
    second = {"ragas": [{**_entry(6403, 1024, 1), "max_prompt": 6403, "cut_by_length": 1}]}
    merged = job_queue.merged_tokens(first, second)["ragas"][0]
    assert (merged["max_prompt"], merged["cut_by_length"], merged["calls"]) == (6403, 1, 2)
