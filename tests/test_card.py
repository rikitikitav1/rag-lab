import engines
import llm
import pytest
from engines import card
from models.registry import EngineKind, Placement

OLLAMA = engines.EngineSpec(1, "ollama", EngineKind.ollama, "OLLAMA", Placement.gpu)
VLLM = engines.EngineSpec(3, "vllm", EngineKind.vllm, "VLLM", Placement.gpu)
CPU = engines.EngineSpec(5, "ollama-cpu", EngineKind.ollama, "OLLAMA_CPU", Placement.cpu)


@pytest.fixture
def stand(monkeypatch):
    # what the servers would say: who is awake, what ollama holds, how many wakes fail first
    s = {"awake": False, "ollama": [], "spilled": [], "wake_fails": 0, "sticky": False, "calls": []}

    def is_sleeping(spec):
        return not s["awake"]

    def card_state(spec):
        return s.get("state") or ("awake" if s["awake"] else "asleep")

    def sleep(spec):
        s["calls"].append("sleep")
        s["awake"] = False

    def wake_up(spec):
        s["calls"].append("wake")
        if s["wake_fails"]:
            s["wake_fails"] -= 1
            raise engines.WakeFailed("CUDA OOM")
        s["awake"] = True

    def unload(name, spec):
        s["calls"].append(f"unload {name}")
        if not s["sticky"]:
            s["ollama"].remove(name)

    def residency(spec):
        on = [{"model": n, "vram_mb": 5000, "size_mb": 5000} for n in s["ollama"]]
        return on + [{"model": n, "vram_mb": 0, "size_mb": 5000} for n in s["spilled"]]

    def load(name, spec):
        s["calls"].append(f"load {name}")
        s["ollama"].append(name)

    monkeypatch.setattr(card, "card_engines", lambda kind=None: [OLLAMA, VLLM])
    monkeypatch.setattr(card.vllm, "is_sleeping", is_sleeping)
    monkeypatch.setattr(card.vllm, "card_state", card_state)
    monkeypatch.setattr(card.vllm, "sleep", sleep)
    monkeypatch.setattr(card.vllm, "wake_up", wake_up)
    monkeypatch.setattr(card.vllm, "served", lambda spec: ["Qwen/Q"])
    monkeypatch.setattr(card.ollama, "unload", unload)
    monkeypatch.setattr(card.ollama, "residency", residency)
    monkeypatch.setattr(card.ollama, "load_into_memory", load)
    monkeypatch.setattr(card, "POLL_SECONDS", 0)
    monkeypatch.setattr(card, "WAIT_CEILING", 0.05)
    return s


def test_the_holder_is_read_from_the_servers(stand):
    assert card.on_card() == []
    # 11.09: ollama lost CUDA and served the judge from the cpu, which holds nothing on the card
    stand["spilled"] = ["qwen2.5:7b"]
    assert card.on_card() == []
    stand["ollama"] = ["llama3.1:8b"]
    assert [h.engine.name for h in card.on_card()] == ["ollama"]
    stand["awake"] = True
    assert [h.engine.name for h in card.on_card()] == ["ollama", "vllm"]


def test_a_card_nobody_holds_is_ollamas_but_not_vllms(stand):
    # ollama loads a role on its first call; vLLM asleep has nothing on the card to answer with
    assert card.holds_for(OLLAMA) is True
    assert card.holds_for(VLLM) is False
    stand["awake"] = True
    assert card.holds_for(VLLM) is True
    assert card.holds_for(OLLAMA) is False, "an awake judge leaves ollama no room"


def test_the_card_goes_to_vllm_after_ollama_let_go_even_if_the_first_wake_fails(stand):
    # 11.09: a wake 3 s after the unload met 5946 MiB still held and failed; a later one worked
    stand["ollama"] = ["llama3.1:8b", "bge-m3:latest"]
    stand["wake_fails"] = 1
    card.hand_to(VLLM)
    assert stand["ollama"] == [] and stand["awake"] is True
    assert stand["calls"] == ["unload llama3.1:8b", "unload bge-m3:latest", "wake", "wake"]


def test_the_card_goes_back_to_ollama_and_loads_the_role(stand):
    stand["awake"] = True
    card.hand_to(OLLAMA, "llama3.1:8b")
    assert stand["calls"] == ["sleep", "load llama3.1:8b"]
    assert [h.engine.name for h in card.on_card()] == ["ollama"]


def test_a_card_that_will_not_free_is_a_failure_with_a_reason(stand):
    stand["ollama"] = ["llama3.1:8b"]
    stand["sticky"] = True
    with pytest.raises(card.CardNotHanded, match="ollama did not let go"):
        card.hand_to(VLLM)
    assert "wake" not in stand["calls"], "never wake onto a card another engine still holds"


def test_a_wake_that_never_succeeds_is_a_failure_not_an_endless_retry(stand):
    stand["wake_fails"] = 10**6
    with pytest.raises(card.CardNotHanded, match="vllm did not wake"):
        card.hand_to(VLLM)


def test_a_job_asks_for_the_card_once_and_waits(monkeypatch):
    import job_queue
    from job_handlers import base

    holds, pending, asked = [False], [False], []
    monkeypatch.setattr(
        llm, "resolve_for",
        lambda role, model=None: engines.Resolved("Qwen/Q", {"judging": VLLM, "cpu": CPU}[role]),
    )
    monkeypatch.setattr(card, "holds_for", lambda spec: holds[0])
    monkeypatch.setattr(job_queue, "pending_of_type", lambda t, **o: pending[0])
    monkeypatch.setattr(job_queue, "enqueue", lambda t, o, **kw: asked.append((t, o)) or 1)

    with pytest.raises(base.Deferred) as waited:
        base.require_card("judging")
    assert asked == [("hand_card", {"engine_id": 3, "model": "Qwen/Q", "asked_by": None})]
    assert waited.value.delay_seconds <= 10, "the waiter must not sleep long past the handover"

    pending[0] = True
    with pytest.raises(base.Deferred):
        base.require_card("judging")
    assert len(asked) == 1, "a retry must not queue a second handover"

    holds[0] = True
    base.require_card("judging")
    # an engine off the card never waits for it
    holds[0] = False
    base.require_card("cpu")
    assert len(asked) == 1


def test_an_engine_off_the_card_loads_without_taking_it_from_anyone(stand):
    stand["awake"] = True
    card.hand_to(CPU, "qwen2.5:7b-instruct-fp16")
    assert stand["calls"] == ["load qwen2.5:7b-instruct-fp16"], "the judge on the card stays awake"
    vllm_cpu = engines.EngineSpec(4, "vllm-cpu", EngineKind.vllm, "VLLM_CPU", Placement.cpu)
    card.hand_to(vllm_cpu)
    assert "wake" not in stand["calls"], "a vLLM on the cpu has no sleep to wake from"


def test_a_job_that_needs_the_card_cannot_be_sent_to_another_lane():
    import job_queue
    import job_specs

    for card_job in ("hand_card", "judge_answers", "eval_run"):
        assert job_specs.lane(card_job) == "default"
        with pytest.raises(ValueError, match="lives in the default lane"):
            job_queue._lane(card_job, "io")
    assert job_queue._lane("pull_llm_model", "io") == "io", "a download keeps its own lane"
    assert job_queue._lane("hand_card", None) == "default"


# one road to the card: a load from the API met an awake judge with CUDA OOM (auditor, 11.09)
ALLOWED_TO_TOUCH_THE_CARD = {
    "engines/card.py", "engines/vllm.py", "engines/ollama.py",
    "job_handlers/card.py",
    # inside jobs of the default lane, which runs one job at a time
    "evals/runner.py", "job_handlers/dataprep.py",
    # before the stack takes any work: vLLM goes to sleep before ollama loads a role
    "bootstrap.py",
}


def test_nothing_outside_the_default_lane_sleeps_wakes_loads_or_unloads():
    import re
    from pathlib import Path

    app = Path(__file__).resolve().parent.parent / "app"
    call = re.compile(r"\b(vllm|ollama|card)\.(sleep|wake_up|load_into_memory|unload|hand_to)\(")
    touching = {
        str(f.relative_to(app)) for f in app.rglob("*.py") if call.search(f.read_text())
    }
    assert touching <= ALLOWED_TO_TOUCH_THE_CARD, touching - ALLOWED_TO_TOUCH_THE_CARD
    assert not any(f.startswith("api/") for f in touching), "the API process never touches the card"


def test_the_chat_waits_for_the_card_through_the_queue_and_says_how_long(client, monkeypatch):
    from types import SimpleNamespace

    import job_queue
    from api.v1 import chat as door
    from use_cases import card_wait

    holds, judging, asked = [False], [True], []
    monkeypatch.setattr(card_wait.llm, "resolve", lambda role: engines.Resolved(
        "bge-m3", VLLM if role == "judging" else OLLAMA))
    monkeypatch.setattr(card_wait.card, "holds_for", lambda spec: holds[0])
    monkeypatch.setattr(card_wait.card, "on_card", lambda: [card.Holding(VLLM, ("Qwen/Q",))])
    monkeypatch.setattr(job_queue, "pending_of_type",
                        lambda t, **o: judging[0] if t == "judge_answers" else bool(asked))
    monkeypatch.setattr(job_queue, "enqueue", lambda t, o, **kw: asked.append((t, o)) or 1)
    monkeypatch.setattr(door.chat, "retrieve",
                        lambda *a, **kw: SimpleNamespace(sources=[], elapsed=0.1))

    busy = client.post("/v1/chat/fast_question", json={"text": "what is redis"})
    assert busy.status_code == 503 and "held by the judge on vllm" in busy.json()["detail"]
    assert busy.headers["Retry-After"] == "60", "a judging pass still running takes about a minute"
    assert asked == [("hand_card", {"engine_id": 1, "asked_by": "chat"})]

    judging[0] = False
    idle = client.post("/v1/chat/fast_question", json={"text": "what is redis"})
    assert idle.headers["Retry-After"] == "5" and len(asked) == 1, "one handover, not one per ask"

    asked_full = client.post("/v1/chat/question", json={"text": "what is redis"})
    assert asked_full.status_code == 503, "the answering door waits for the card as well"

    holds[0] = True
    assert client.post("/v1/chat/fast_question", json={"text": "what is redis"}).status_code == 200


def test_a_silent_vllm_is_not_a_free_card_but_a_stopped_one_is(stand):
    # a timeout may hide a judge awake on the card; a refused connection is a process that is gone
    stand["state"] = "unknown"
    assert card.holds_for(OLLAMA) is False, "never load beside a server that did not answer"
    stand["state"] = "down"
    assert card.holds_for(OLLAMA) is True, "a stopped vLLM must not block ollama forever"


def test_every_role_that_is_ready_asks_for_the_card(monkeypatch):
    # 11.09: the language probe and the guest axes reached a sleeping judge without asking
    from job_handlers import base
    from models.registry import Role, Status

    asked = []
    monkeypatch.setattr(base, "require_card", lambda role, model=None, asked_by=None:
                        asked.append((role, asked_by)))

    class _Session:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def scalar(self, _stmt):
            return type("M", (), {"status": Status.ready})()

    monkeypatch.setattr(base, "Session", _Session)
    base.require_role_ready(Role.judging)
    base.require_role_ready(Role.paraphrasing)
    assert asked == [("judging", "judge_answers"), ("paraphrasing", None)], (
        "a judge's handover waits with the judging, a run's does not"
    )


def test_every_handler_that_names_a_role_passes_a_gate_that_asks_for_the_card():
    import inspect
    import re

    import job_handlers

    gates = ("require_role_ready", "require_embedder_ready", "require_card")
    names_a_role = re.compile(
        r"\bRole\.(generation|embedding|judging|paraphrasing)|resolve(_for)?\("
    )
    for job_type, handler in job_handlers.HANDLERS.items():
        source = inspect.getsource(handler)
        if names_a_role.search(source):
            assert any(g in source for g in gates), f"{job_type} reaches a role without the card"


# the handlers that answer with a model, and the role each must ask for before its first call
ROLE_HANDLERS = [
    ("judging", "judge_language", {"run_name": "r"}, "require_role_ready", "judging"),
    ("judging", "judge_guest_axes", {"run_name": "r"}, "require_role_ready", "judging"),
    ("judging", "judge_answers", {"run_name": "r"}, "require_role_ready", "judging"),
    ("dataprep", "paraphrase_questions", {}, "require_role_ready", "paraphrasing"),
    ("dataprep", "build_veto_set", {}, "require_role_ready", "paraphrasing"),
    ("indexing", "index_data", {}, "require_embedder_ready", None),
    ("indexing", "embed_questions", {}, "require_embedder_ready", None),
    ("evaluation", "eval_run", {"run_name": "r"}, "require_role_ready", "generation"),
]


class _Stopped(Exception):
    pass


@pytest.mark.parametrize("module,handler,options,gate,role", ROLE_HANDLERS)
def test_a_handler_asks_for_its_role_before_it_calls_a_model(monkeypatch, module, handler,
                                                              options, gate, role):
    import importlib

    mod = importlib.import_module(f"job_handlers.{module}")
    asked = []

    def stop(*args, **kw):
        asked.append(getattr(args[0], "value", args[0]) if args else None)
        raise _Stopped

    monkeypatch.setattr(mod, gate, stop)
    if hasattr(mod, "guests_available"):
        monkeypatch.setattr(mod, "guests_available", lambda: True)
    with pytest.raises(_Stopped):
        getattr(mod, handler)(options)
    assert asked == [role], f"{handler} must ask for {role} first"


def test_a_role_says_where_its_model_answered_from_by_its_own_engine(stand):
    # 11.09: ollama lost CUDA and the generator answered from the cpu with nothing in the record
    assert card.model_on_card(OLLAMA, "llama3.1:8b") is None, "not loaded is not read, not a no"
    stand["ollama"] = ["llama3.1:8b"]
    assert card.model_on_card(OLLAMA, "llama3.1:8b") is True
    stand["ollama"], stand["spilled"] = [], ["llama3.1:8b"]
    assert card.model_on_card(OLLAMA, "llama3.1:8b") is False, "all of it on the cpu"
    assert card.model_on_card(VLLM, "Qwen/Q") is False, "asleep answers nothing from the card"
    stand["awake"] = True
    assert card.model_on_card(VLLM, "Qwen/Q") is True
    stand["state"] = "unknown"
    assert card.model_on_card(VLLM, "Qwen/Q") is None
    assert card.model_on_card(CPU, "qwen2.5:7b-instruct-fp16") is False, "declared off the card"


def test_the_run_snapshot_stamps_each_answering_role_placement(monkeypatch):
    from models.registry import Role
    from use_cases import run_snapshot

    embed = engines.Resolved("bge-m3", OLLAMA)
    monkeypatch.setattr(run_snapshot.llm, "resolve", lambda role: embed)
    monkeypatch.setattr(run_snapshot.llm, "sampler",
                        lambda role, spec: engines.Sampler({}, {}))
    monkeypatch.setattr(run_snapshot.card, "model_on_card",
                        lambda spec, name: {"llama3.1:8b": False, "bge-m3": True}[name])
    named, _, placed = run_snapshot._by_role(engines.Resolved("llama3.1:8b", OLLAMA))
    assert placed == {Role.generation: False, Role.embedding: True}
    assert "on_card" in run_snapshot.KEYS and run_snapshot.SCHEMA == 8


def test_no_process_of_ours_holds_the_cross_encoder_on_the_card():
    # 11.09: torch in the API held card the handover could not reach, and on OOM it went to the cpu
    import inspect

    import rerank
    from job_handlers import card as job

    source = inspect.getsource(rerank)
    assert "torch" not in source and "sentence_transformers" not in source
    assert "rerank" not in inspect.getsource(job), "a handover has no torch to ask aside"


def test_a_second_role_takes_the_card_per_call_and_the_judge_goes_to_sleep(stand, monkeypatch):
    # 11.09: the language probe woke the judge, then asked ollama for a restatement on a full card
    from job_handlers import card as handler

    stand["awake"] = True
    handler.take_for_call(OLLAMA, "llama3.1:8b")
    assert stand["calls"] == ["sleep"], "ollama loads on the call itself, the judge only sleeps"

    handler.take_for_call(OLLAMA, "llama3.1:8b")
    assert stand["calls"] == ["sleep"], "the card already held is not handed again"

    stand["ollama"] = ["llama3.1:8b"]
    handler.take_for_call(VLLM, "Qwen/Q")
    assert stand["calls"][1:] == ["unload llama3.1:8b", "wake"]
    handler.take_for_call(VLLM, "Qwen/Q")
    assert stand["calls"][1:] == ["unload llama3.1:8b", "wake"], "an awake vLLM is not woken again"

    def unread(spec):
        raise AssertionError("an engine off the card reads nobody's card")

    monkeypatch.setattr(handler.card, "holds_for", unread)
    handler.take_for_call(CPU, "bge-m3")
    assert stand["calls"][-1] == "wake", "an engine off the card takes nothing from the one on it"


def test_every_call_to_a_model_passes_the_card_hook_with_its_own_engine(monkeypatch):
    from types import SimpleNamespace

    seen = []
    monkeypatch.setattr(llm, "resolve_for", lambda role, model=None: engines.Resolved(
        {"generation": "llama3.1:8b", "embedding": "bge-m3"}[role],
        {"generation": OLLAMA, "embedding": VLLM}[role]))
    monkeypatch.setattr(llm, "resolve", lambda role: llm.resolve_for(role))
    usage = SimpleNamespace(prompt_tokens=1, completion_tokens=1)
    reply = SimpleNamespace(usage=usage, choices=[SimpleNamespace(
        message=SimpleNamespace(content="ok", tool_calls=None), finish_reason="stop")])
    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kw: reply)),
        embeddings=SimpleNamespace(create=lambda **kw: SimpleNamespace(
            data=[SimpleNamespace(embedding=[0.0])])),
    )
    monkeypatch.setattr(llm.engines, "client_for", lambda spec: client)
    monkeypatch.setattr(llm, "_before_call", lambda spec, name: seen.append((spec.name, name)))

    llm.ask("s", "u")
    llm.chat([{"role": "user", "content": "u"}])
    llm.embed("q")
    assert seen == [("ollama", "llama3.1:8b"), ("ollama", "llama3.1:8b"), ("vllm", "bge-m3")]


def test_the_worker_takes_the_card_before_calls_and_the_api_never_does(monkeypatch):
    import server  # noqa: F401
    import worker

    assert llm._before_call is None, "the API answers 503 or 409, it never hands the card itself"

    class _Up(Exception):
        pass

    monkeypatch.setattr(worker, "reclaim", lambda queues: None)
    monkeypatch.setattr(worker, "QUEUES", ["default"])

    def up(queues):
        raise _Up

    monkeypatch.setattr(worker, "_loop", up)
    monkeypatch.setattr(llm, "_before_call", None)
    with pytest.raises(_Up):
        worker.main()
    assert llm._before_call is worker.job_handlers.card.take_for_call


def test_a_run_takes_the_card_once_for_generation_and_not_for_each_role(monkeypatch):
    # two preambles on two engines handed the card back and forth, and the run never started
    from job_handlers import evaluation

    carded, stopped = [], RuntimeError("stop")
    monkeypatch.setattr(evaluation, "require_role_ready",
                        lambda role, take_card=True: carded.append((role.value, take_card)))

    def once(role, model=None, asked_by=None):
        carded.append(role)
        raise stopped

    monkeypatch.setattr(evaluation, "require_card", once)
    with pytest.raises(RuntimeError):
        evaluation.eval_run({"run_name": "r"})
    assert carded == [("generation", False), ("embedding", False), "generation"]


def test_the_busy_card_names_its_real_holder_and_a_split_layout_is_a_409(monkeypatch):
    from use_cases import card_wait

    roles = {"judging": VLLM, "embedding": OLLAMA, "generation": OLLAMA}
    monkeypatch.setattr(card_wait.llm, "resolve", lambda role: engines.Resolved("m", roles[role]))
    monkeypatch.setattr(card_wait.card, "holds_for", lambda spec: False)
    monkeypatch.setattr(card_wait.job_queue, "pending_of_type", lambda t, **o: True)
    other = engines.EngineSpec(6, "vllm-2", EngineKind.vllm, "VLLM_2", Placement.gpu)
    monkeypatch.setattr(card_wait.card, "on_card", lambda: [card.Holding(other, ("x",))])
    with pytest.raises(card_wait.CardBusy) as held:
        card_wait.wait_for_the_card("embedding")
    assert held.value.status == 503 and "held by vllm-2;" in held.value.detail, "not the judge"

    # the embedder and the generator on two card engines: the chat would hand the card per call
    roles["embedding"] = VLLM
    with pytest.raises(card_wait.CardBusy) as split:
        card_wait.wait_for_the_card("embedding", "generation")
    assert split.value.status == 409 and split.value.retry_after is None
    assert "ollama, vllm" in split.value.detail
    # one role alone is served whatever the other sits on
    monkeypatch.setattr(card_wait.card, "holds_for", lambda spec: True)
    card_wait.wait_for_the_card("embedding")
    # a role off the card never makes a split
    roles["embedding"] = CPU
    card_wait.wait_for_the_card("embedding", "generation")


def test_indexing_clears_the_embedder_s_engine_of_everything_else(monkeypatch):
    # 11.09: bge-m3 in batches of 64 beside a resident llama3.1:8b dropped ollama's runner
    from job_handlers import card as handler

    resident = [{"model": "llama3.1:8b"}, {"model": "bge-m3:latest"}]
    gone = []
    spec = [OLLAMA]
    monkeypatch.setattr(handler.llm, "resolve", lambda role: engines.Resolved("bge-m3", spec[0]))
    monkeypatch.setattr(handler.ollama, "residency", lambda s: resident)
    monkeypatch.setattr(handler.ollama, "unload", lambda name, s: gone.append(name))
    handler.clear_the_engine_for("embedding")
    assert gone == ["llama3.1:8b"]
    spec[0] = CPU
    handler.clear_the_engine_for("embedding")
    assert gone == ["llama3.1:8b"], "the processor has no card to run out of"


def test_both_embedding_jobs_clear_the_engine_before_the_first_batch():
    import inspect

    from job_handlers import indexing

    for job in (indexing.index_data, indexing.embed_questions):
        lines = [line.strip() for line in inspect.getsource(job).splitlines()]
        gate = lines.index("require_embedder_ready()")
        assert lines[gate + 1] == 'clear_the_engine_for("embedding")', job.__name__
