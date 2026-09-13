import engines
import llm
import pytest
from engines import card
from models.registry import EngineKind, Placement
from stand_specs import OLLAMA, VLLM
from stand_specs import OLLAMA_CPU as CPU


@pytest.fixture(autouse=True)
def _no_call_left_in_flight(monkeypatch):
    # a test that fails mid-call must not leave the next one waiting for a call that never ends
    from job_handlers import card as handler

    monkeypatch.setattr(handler, "_in_flight", {})
    monkeypatch.setattr(handler, "_waiting_for", None)


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
    monkeypatch.setattr("engines.ollama.unload", unload)
    monkeypatch.setattr("engines.ollama.residency", residency)
    def card_reading(spec=None):
        if s.get("ollama_state"):
            return s["ollama_state"], []
        seen = residency(spec)
        return ("holds" if any(m["vram_mb"] > 0 for m in seen) else "free"), seen

    monkeypatch.setattr("engines.ollama.card_reading", card_reading)
    monkeypatch.setattr("engines.ollama.load_into_memory", load)
    monkeypatch.setattr(card, "POLL_SECONDS", 0)
    monkeypatch.setattr(card, "WAIT_CEILING", 0.05)
    return s


def test_the_holder_is_read_from_the_servers(stand):
    assert card.on_card() == []
    # ollama lost CUDA and served the judge from the cpu, which holds nothing on the card
    stand["spilled"] = ["qwen2.5:7b"]
    assert card.on_card() == []
    stand["ollama"] = ["llama3.1:8b"]
    assert [h.engine.name for h in card.on_card()] == ["ollama"]
    stand["awake"] = True
    assert [h.engine.name for h in card.on_card()] == ["ollama", "vllm"]


def test_a_silent_ollama_holds_the_card_as_a_silent_vllm_does(stand):
    # silence was a free card for ollama alone, and a wake would meet what it held
    stand["ollama_state"] = "unknown"
    assert [h.engine.name for h in card.on_card()] == ["ollama"]
    assert card.holds_for(VLLM) is False, "nothing wakes beside a server that did not answer"


def test_a_stopped_ollama_holds_nothing_and_the_judge_takes_the_card_at_once(stand):
    # with ollama stopped a handover to the judge waited 60 s and failed
    stand["ollama_state"] = "down"
    assert card.on_card() == []
    assert card.holds_for(VLLM) is False and card.holds_for(OLLAMA) is True
    card.hand_to(VLLM)
    assert stand["awake"] is True and stand["calls"] == ["wake"]


def test_a_card_nobody_holds_is_ollamas_but_not_vllms(stand):
    # ollama loads a role on its first call; vLLM asleep has nothing on the card to answer with
    assert card.holds_for(OLLAMA) is True
    assert card.holds_for(VLLM) is False
    stand["awake"] = True
    assert card.holds_for(VLLM) is True
    assert card.holds_for(OLLAMA) is False, "an awake judge leaves ollama no room"


def test_the_card_goes_to_vllm_after_ollama_let_go_even_if_the_first_wake_fails(stand):
    # a wake 3 s after the unload met 5946 MiB still held and failed; a later one worked
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


def test_a_job_takes_the_card_in_its_own_turn_and_queues_nothing(stand, monkeypatch):
    # a handover queued apart ping-ponged the card with whoever came between
    import job_queue
    from job_handlers import base
    from job_handlers import card as handler

    monkeypatch.setattr(handler, "_probe_the_woken_generator", lambda spec: None)
    monkeypatch.setattr(
        llm, "resolve_for",
        lambda role, model=None: engines.Resolved("Qwen/Q", {"judging": VLLM, "cpu": CPU}[role]),
    )
    monkeypatch.setattr(job_queue, "enqueue", lambda *a, **kw: pytest.fail("a job queued a handover"))
    stand["ollama"] = ["llama3.1:8b"]
    base.require_card("judging")
    assert stand["calls"] == ["unload llama3.1:8b", "wake"], "taken before the first row"
    base.require_card("judging")
    assert stand["calls"] == ["unload llama3.1:8b", "wake"], "a card already held is not taken again"
    base.require_card("cpu")
    assert stand["calls"][-1] == "wake", "an engine off the card never takes it"


def _calls_in(source: str, hit) -> bool:
    import ast

    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        if isinstance(fn, ast.Attribute):
            # the last name before the dot: `card_holder.hand_to` and `engines.vllm.sleep` alike
            owner = getattr(fn.value, "id", None) or getattr(fn.value, "attr", None)
            name = fn.attr
        else:
            owner, name = None, getattr(fn, "id", None)
        if hit(owner, name):
            return True
    return False


# by the call in the tree, not the text: a comment never matches, a bare imported name still does
def _files_calling(hit) -> set[str]:
    from pathlib import Path

    app = Path(__file__).resolve().parent.parent / "app"
    return {str(f.relative_to(app)) for f in app.rglob("*.py") if _calls_in(f.read_text(), hit)}


HANDS_THE_CARD = {"hand_to", "sleep_every_vllm", "clear_for", "wake_up", "let_go", "take_once",
                  "load_off_card", "make_room"}


def _hands_the_card(owner, name) -> bool:
    return name in HANDS_THE_CARD or (name == "sleep" and (owner or "").startswith("vllm"))


def test_a_handover_waits_for_the_calls_on_the_engine_it_would_put_to_sleep(stand, monkeypatch):
    # two threads of one pass on two card engines: the embedder must not sleep the judge mid-request
    import threading

    from job_handlers import card as handler

    monkeypatch.setattr(handler, "_probe_the_woken_generator", lambda spec: None)
    stand["awake"] = True
    judge_call = handler.take_for_call(VLLM, "Qwen/Q")

    def call(spec, name, ends):
        ends.append(handler.take_for_call(spec, name))
        stand["calls"].append(f"started on {spec.name}")

    embed_ends, judge_ends = [], []
    embed = threading.Thread(target=call, args=(OLLAMA, "bge-m3", embed_ends))
    embed.start()
    embed.join(0.3)
    assert embed.is_alive() and "sleep" not in stand["calls"], "the judge's request is still out"

    # a new judge call does not overtake the handover that already waits
    judge = threading.Thread(target=call, args=(VLLM, "Qwen/Q", judge_ends))
    judge.start()
    judge.join(0.3)
    assert judge.is_alive()

    stand["calls"].append("judge answered")
    judge_call()
    embed.join(5)
    assert not embed.is_alive() and judge.is_alive(), "the embedder's call is out now"
    embed_ends[0]()
    judge.join(5)
    assert not judge.is_alive()
    calls = stand["calls"]
    assert calls.index("judge answered") < calls.index("sleep") < calls.index("started on ollama")
    assert calls.index("started on ollama") < calls.index("wake") < calls.index("started on vllm")
    judge_ends[0]()


def test_the_card_guards_read_calls_not_text():
    assert _calls_in("from engines.vllm import wake_up\nwake_up(spec)", _hands_the_card)
    assert _calls_in("import engines.vllm as v\nv.wake_up(spec)", _hands_the_card)
    assert _calls_in("engines.vllm.sleep(spec)", _hands_the_card)
    assert not _calls_in("# vllm.wake_up(spec)\nsaid = 'hand_to('\ntime.sleep(1)", _hands_the_card)


def test_no_module_but_the_owner_changes_who_holds_the_card():
    # five places changed the holder once; one road is the whole point
    owners = {"engines/card.py", "engines/drivers.py", "job_handlers/card.py", "bootstrap.py"}
    changing = _files_calling(_hands_the_card)
    assert changing <= owners, changing


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


# one road to the card: a load from the API met an awake judge with CUDA OOM
ALLOWED_TO_TOUCH_THE_CARD = {
    "engines/card.py", "engines/drivers.py", "engines/vllm.py", "engines/ollama.py",
    "job_handlers/card.py",
    # before the stack takes any work: vLLM goes to sleep before ollama loads a role
    "bootstrap.py",
}

TOUCHES_THE_CARD = {"sleep", "wake_up", "load_into_memory", "unload", "hand_to"}


def _touches_the_card(owner, name) -> bool:
    if name not in TOUCHES_THE_CARD:
        return False
    # a bare `sleep` is `time.sleep`, and a bare `unload` could be anyone's
    if owner is None:
        return name not in ("sleep", "unload")
    return owner.startswith(("vllm", "ollama", "card"))


def test_nothing_outside_the_default_lane_sleeps_wakes_loads_or_unloads():
    touching = _files_calling(_touches_the_card)
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
    monkeypatch.setattr(job_queue, "pending_of_type", lambda t, **o: bool(asked))
    monkeypatch.setattr(job_queue, "running_of_type", lambda t: judging[0])
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
    # the language probe and the guest axes reached a sleeping judge without asking
    from job_handlers import base
    from models.registry import Role, Status

    asked = []
    monkeypatch.setattr(base, "require_card", lambda role, model=None: asked.append(role))

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
    assert asked == ["judging", "paraphrasing"]


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
    # ollama lost CUDA and the generator answered from the cpu with nothing in the record
    assert card.model_on_card(OLLAMA, "llama3.1:8b") is None, "not loaded is not read, not a no"
    stand["ollama"] = ["llama3.1:8b"]
    assert card.model_on_card(OLLAMA, "llama3.1:8b") is True
    stand["ollama"], stand["spilled"] = [], ["llama3.1:8b"]
    assert card.model_on_card(OLLAMA, "llama3.1:8b") is False, "all of it on the cpu"
    assert card.model_on_card(VLLM, "Qwen/Q") is False, "asleep answers nothing from the card"
    stand["awake"] = True
    assert card.model_on_card(VLLM, "Qwen/Q") is True
    assert card.model_on_card(VLLM, "Qwen/Other") is False, "awake, but with another model"
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
    named, _, placed, cache_keys, parsers = run_snapshot._by_role(engines.Resolved("llama3.1:8b", OLLAMA))
    assert placed == {Role.generation: False, Role.embedding: True}
    assert parsers == {Role.generation: "none@1", Role.embedding: "none@1"}
    assert cache_keys == {}, "a local engine keeps no broker cache"
    assert "on_card" in run_snapshot.KEYS and run_snapshot.SCHEMA == 11


def test_a_run_that_reranks_names_the_reranker_and_keeps_what_was_read_while_roles_worked(monkeypatch):
    # the reranker's move left no trace in the record, and the embedder read as None
    from models.registry import Role
    from use_cases import run_snapshot

    rerank_engine = engines.EngineSpec(7, "vllm-rerank", EngineKind.vllm, "VLLM_RERANK", Placement.gpu)
    picks = {"embedding": engines.Resolved("bge-m3", OLLAMA),
             "reranking": engines.Resolved("BAAI/r", rerank_engine)}
    monkeypatch.setattr(run_snapshot.llm, "resolve", lambda role: picks[role])
    monkeypatch.setattr(run_snapshot.llm, "sampler", lambda role, spec: engines.Sampler({}, {}))
    monkeypatch.setattr(run_snapshot.card, "model_on_card", lambda spec, name: None)
    monkeypatch.setattr(run_snapshot, "_window", lambda picked: 8192)
    monkeypatch.setattr(run_snapshot, "_generator", lambda model: engines.Resolved("llama", OLLAMA))
    monkeypatch.setattr(run_snapshot.db, "fingerprint_or_none", lambda variant: None)
    snap = run_snapshot.of_run(variant="baseline", use_rerank=True, k=5, ef_search=100,
                               distance_threshold=None, placed_during={Role.embedding: True})
    assert snap["engines"][Role.reranking] == "vllm-rerank"
    assert snap["on_card"][Role.embedding] is True, "read while it worked, not after it left"
    plain = run_snapshot.of_run(variant="baseline", use_rerank=False, k=5, ef_search=100,
                                distance_threshold=None)
    assert Role.reranking not in plain["engines"]
    gated = run_snapshot.of_run(variant="baseline", use_rerank=False, k=5, ef_search=100,
                                distance_threshold=None, cross_encoder_used=True)
    assert gated["engines"][Role.reranking] == "vllm-rerank", "the agent's gate called it"
    assert gated["rerank"] is False, "the knob stays what was asked"


def test_no_process_of_ours_holds_the_cross_encoder_on_the_card():
    # torch in the API held card the handover could not reach, and on OOM it went to the cpu
    import inspect

    import rerank
    from job_handlers import card as job

    source = inspect.getsource(rerank)
    assert "torch" not in source and "sentence_transformers" not in source
    assert "rerank" not in inspect.getsource(job), "a handover has no torch to ask aside"


def test_a_second_role_takes_the_card_per_call_and_the_judge_goes_to_sleep(stand, monkeypatch):
    # the language probe woke the judge, then asked ollama for a restatement on a full card
    from job_handlers import card as handler

    monkeypatch.setattr(handler, "_probe_the_woken_generator", lambda spec: None)
    stand["awake"] = True
    # each call ends before the next, as `llm` ends it once the answer is in
    handler.take_for_call(OLLAMA, "llama3.1:8b")()
    assert stand["calls"] == ["sleep"], "ollama loads on the call itself, the judge only sleeps"

    handler.take_for_call(OLLAMA, "llama3.1:8b")()
    assert stand["calls"] == ["sleep"], "the card already held is not handed again"

    stand["ollama"] = ["llama3.1:8b"]
    handler.take_for_call(VLLM, "Qwen/Q")()
    assert stand["calls"][1:] == ["unload llama3.1:8b", "wake"]
    handler.take_for_call(VLLM, "Qwen/Q")()
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

    def once(role, model=None, allow_spill=False):
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
    monkeypatch.setattr(card_wait.job_queue, "running_of_type", lambda t: True)
    other = engines.EngineSpec(6, "vllm-2", EngineKind.vllm, "VLLM_2", Placement.gpu)
    monkeypatch.setattr(card_wait.card, "on_card", lambda: [card.Holding(other, ("x",))])
    with pytest.raises(card_wait.CardHeld) as held:
        card_wait.wait_for_the_card("embedding")
    assert "held by vllm-2;" in held.value.detail, "not the judge"

    # no judge seated: the holder is still named, not a 500
    def unseated(role):
        if role == "judging":
            raise engines.Unnamed("no model assigned to role judging")
        return engines.Resolved("m", roles[role])

    monkeypatch.setattr(card_wait.llm, "resolve", unseated)
    with pytest.raises(card_wait.CardHeld) as held:
        card_wait.wait_for_the_card("embedding")
    assert "held by vllm-2;" in held.value.detail
    monkeypatch.setattr(card_wait.llm, "resolve", lambda role: engines.Resolved("m", roles[role]))

    # the embedder and the generator on two card engines: the chat would hand the card per call
    roles["embedding"] = VLLM
    with pytest.raises(card_wait.CannotAnswer) as split:
        card_wait.wait_for_the_card("embedding", "generation")
    assert "ollama, vllm" in split.value.detail
    # one role alone is served whatever the other sits on
    monkeypatch.setattr(card_wait.card, "holds_for", lambda spec: True)
    card_wait.wait_for_the_card("embedding")
    # a role off the card never makes a split
    roles["embedding"] = CPU
    card_wait.wait_for_the_card("embedding", "generation")


def test_indexing_clears_the_embedder_s_engine_of_everything_else(monkeypatch):
    # bge-m3 in batches of 64 beside a resident llama3.1:8b dropped ollama's runner
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


def test_a_handover_to_a_silent_server_refuses_before_anything_lets_go(stand):
    # the card was released first, then `wake_up` met a refused connection, card lost
    stand["ollama"] = ["llama3.1:8b"]
    stand["state"] = "down"
    with pytest.raises(card.CardNotHanded, match="vllm is down; the card stays"):
        card.hand_to(VLLM)
    stand["state"] = "unknown"
    with pytest.raises(card.CardNotHanded, match="unknown"):
        card.hand_to(VLLM)
    assert stand["calls"] == [], "nobody let go of the card"

    stand.pop("state")
    stand["awake"] = True
    for state in ("down", "unknown"):
        stand["ollama_state"] = state
        with pytest.raises(card.CardNotHanded, match=f"ollama is {state}; the card stays"):
            card.hand_to(OLLAMA)
    assert stand["calls"] == [], "the judge stays awake when ollama cannot take the card"


def test_a_server_that_dies_mid_wake_is_retried_to_the_ceiling_not_thrown_raw(stand, monkeypatch):
    import requests

    def dead(spec):
        raise requests.ConnectionError("refused")

    monkeypatch.setattr(card.vllm, "wake_up", dead)
    with pytest.raises(card.CardNotHanded, match="did not wake"):
        card.hand_to(VLLM)


def test_whatever_breaks_a_handover_inside_a_call_is_a_stand_fault(stand, monkeypatch):
    import requests
    from errors import StandFault
    from job_handlers import card as handler

    def broken(spec):
        raise requests.ConnectionError("refused")

    monkeypatch.setattr(handler.card, "holds_for", broken)
    with pytest.raises(StandFault, match="did not reach vllm"):
        handler.take_for_call(VLLM, "Qwen/Q")


def test_a_held_card_is_not_handed_again_unless_ollama_lacks_the_model_asked(stand, monkeypatch):
    from job_handlers import card as handler

    handed = []
    monkeypatch.setattr(handler, "_probe_the_woken_generator", lambda spec: None)
    monkeypatch.setattr(handler.card, "hand_to",
                        lambda spec, model=None, allow_spill=False: handed.append((spec.name, model)))
    stand["ollama"] = ["llama3.1:8b"]
    handler.take(OLLAMA, "llama3.1:8b")
    handler.take(OLLAMA)
    assert handed == [], "every call of a role would otherwise ask the servers to hand it over again"
    handler.take(OLLAMA, "gemma2:9b")
    assert handed == [("ollama", "gemma2:9b")], "a load of another model on the holder still loads"
    spill = []
    monkeypatch.setattr(handler.card, "hand_to",
                        lambda spec, model=None, allow_spill=False: spill.append(allow_spill))
    handler.take(OLLAMA, "qwen2.5:7b", allow_spill=True)
    assert spill == [True], "a run's `allow_cpu` reaches the card"
    # a model resident only on the cpu is not loaded for the card
    stand["spilled"] = ["qwen2.5:7b"]
    handler.take(OLLAMA, "qwen2.5:7b")
    assert len(spill) == 2, "a model resident only on the cpu is loaded again for the card"


def test_a_probe_that_fails_after_a_handover_is_not_a_card_that_did_not_arrive(stand, monkeypatch):
    # the handover itself went through
    from job_handlers import card as handler

    monkeypatch.setattr(handler.card, "hand_to", lambda spec, model=None, allow_spill=False: None)

    def hiccup(spec):
        raise RuntimeError("registry unreachable")

    monkeypatch.setattr(handler, "_probe_the_woken_generator", hiccup)
    with pytest.raises(RuntimeError, match="registry unreachable") as raised:
        handler.take(VLLM)
    assert not isinstance(raised.value, card.CardNotHanded)


def test_a_model_that_loads_half_on_the_processor_is_not_a_handed_card(stand, monkeypatch):
    # a model half on the processor answers with other kernels
    def spills(name, spec):
        stand["calls"].append(f"load {name}")
        stand["spilled"].append(name)

    monkeypatch.setattr("engines.ollama.load_into_memory", spills)
    with pytest.raises(card.CardNotHanded, match="not whole on the card"):
        card.hand_to(OLLAMA, "llama3.1:8b")
    # a run with `allow_cpu` asked for the cpu
    card.hand_to(OLLAMA, "llama3.1:8b", allow_spill=True)


def test_a_run_s_allow_cpu_reaches_the_card(monkeypatch):
    from job_handlers import evaluation

    asked = []
    monkeypatch.setattr(evaluation, "require_role_ready", lambda role, take_card=True: None)
    monkeypatch.setattr(evaluation, "require_card",
                        lambda role, model=None, allow_spill=False: asked.append(allow_spill))
    monkeypatch.setattr(evaluation.runner, "run", lambda **kw: 0)
    evaluation.eval_run({"run_name": "r", "set_name": "s", "allow_cpu": True})
    evaluation.eval_run({"run_name": "r", "set_name": "s"})
    assert asked == [True, False]


def test_the_chat_waits_a_minute_only_while_a_judge_is_running(monkeypatch):
    # the live batch waits five minutes after every answer, and read as a busy judge
    from use_cases import card_wait

    monkeypatch.setattr(card_wait.llm, "resolve", lambda role: engines.Resolved("m", OLLAMA))
    monkeypatch.setattr(card_wait.card, "holds_for", lambda spec: False)
    monkeypatch.setattr(card_wait.card, "on_card", lambda: [])
    monkeypatch.setattr(card_wait.job_queue, "pending_of_type", lambda t, **o: True)
    running = [False]
    monkeypatch.setattr(card_wait.job_queue, "running_of_type", lambda t: running[0])
    with pytest.raises(card_wait.CardHeld) as idle:
        card_wait.wait_for_the_card("generation")
    running[0] = True
    with pytest.raises(card_wait.CardHeld) as busy:
        card_wait.wait_for_the_card("generation")
    assert (idle.value.retry_after, busy.value.retry_after) == (5, 60)


def test_an_unseated_role_is_a_409_not_a_500(monkeypatch):
    from use_cases import card_wait

    def unseated(role):
        raise engines.Unnamed("no model assigned to role reranking")

    monkeypatch.setattr(card_wait.llm, "resolve", unseated)
    with pytest.raises(card_wait.CannotAnswer) as refused:
        card_wait.wait_for_the_card("reranking")
    assert "PUT /v1/role" in refused.value.detail


def test_a_judging_pass_asks_for_the_card_once(monkeypatch):
    # the role gate took the card, and then the bench took it again
    from job_handlers import judging

    asked = []

    class _Stop(Exception):
        pass

    monkeypatch.setattr(judging, "require_role_ready",
                        lambda role, take_card=True: asked.append(("role", take_card)))

    def once(role, model=None):
        asked.append(("card", role))
        raise _Stop

    monkeypatch.setattr(judging, "require_card", once)
    monkeypatch.setattr(judging, "_refuse_a_second_judge",
                        lambda run_name, model: asked.append(("second judge?", run_name)))
    with pytest.raises(_Stop):
        judging.judge_answers({"run_name": "r"})
    # a refusal comes before the card, so it never wakes a judge it would turn away
    assert asked == [("role", False), ("second judge?", "r"), ("card", "judging")]
