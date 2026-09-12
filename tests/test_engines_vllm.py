import engines
import pytest
from engines import vllm
from models.registry import EngineKind, Placement
from stand_specs import VLLM as SPEC


class _Reply:
    def __init__(self, status=200, body=None, text=""):
        self.status_code = status
        self.ok = status < 400
        self._body = body
        self.text = text

    def json(self):
        return self._body

    def raise_for_status(self):
        if not self.ok:
            raise RuntimeError(f"http {self.status_code}")


@pytest.fixture
def server(monkeypatch):
    monkeypatch.setenv("VLLM_BASE_URL", "http://vllm:8000")
    vllm._probed.clear()
    seen = {"calls": [], "replies": {}}

    def call(method):
        def go(url, **kw):
            path = url.removeprefix("http://vllm:8000")
            seen["calls"].append((method, path, kw))
            reply = seen["replies"].get((method, path))
            if isinstance(reply, Exception):
                raise reply
            return reply or _Reply(404)
        return go

    monkeypatch.setattr(vllm.requests, "get", call("get"))
    monkeypatch.setattr(vllm.requests, "post", call("post"))
    return seen


START = "process_start_time_seconds 1.78905401397e+09\n"


def test_the_sleep_state_is_read_and_silence_is_not_a_no(server):
    server["replies"][("get", "/is_sleeping")] = _Reply(body={"is_sleeping": True})
    assert vllm.is_sleeping(SPEC) is True
    server["replies"][("get", "/is_sleeping")] = _Reply(body={"is_sleeping": False})
    assert vllm.is_sleeping(SPEC) is False
    # without VLLM_SERVER_DEV_MODE the route is absent: such a server never sleeps, so it is awake
    server["replies"][("get", "/is_sleeping")] = _Reply(404)
    assert vllm.is_sleeping(SPEC) is False
    assert vllm.card_state(SPEC) == "awake" and vllm.has_sleep_routes(SPEC) is False


def test_a_wake_that_the_server_refused_is_an_error_not_a_quiet_retry(server):
    # a wake on a card ollama still held answered 500 and left the server asleep
    server["replies"][("post", "/wake_up")] = _Reply(500)
    with pytest.raises(engines.WakeFailed):
        vllm.wake_up(SPEC)
    server["replies"][("post", "/wake_up")] = _Reply(200)
    vllm.wake_up(SPEC)


def test_sleep_asks_for_the_level_that_keeps_the_weights_in_host_memory(server):
    server["replies"][("post", "/sleep")] = _Reply(200)
    vllm.sleep(SPEC)
    assert server["calls"][-1][2]["params"] == {"level": 1}
    server["replies"][("post", "/sleep")] = _Reply(500)
    with pytest.raises(RuntimeError):
        vllm.sleep(SPEC)


def test_every_call_carries_the_engine_key(server, monkeypatch):
    monkeypatch.setenv("VLLM_API_KEY", "k")
    server["replies"][("get", "/v1/models")] = _Reply(body={"data": [{"id": "Qwen/Q"}]})
    assert vllm.served(SPEC) == ["Qwen/Q"]
    assert server["calls"][-1][2]["headers"] == {"Authorization": "Bearer k"}


def test_the_tool_probe_reads_the_server_and_asks_once_per_process(server):
    server["replies"][("get", "/metrics")] = _Reply(text=START)
    server["replies"][("post", "/v1/chat/completions")] = _Reply(
        body={"choices": [{"message": {"tool_calls": [{"id": "1"}]}}]}
    )
    assert vllm.tool_calls_probed(SPEC, "Qwen/Q") is True
    assert vllm.tool_calls_probed(SPEC, "Qwen/Q") is True
    asked = [c for c in server["calls"] if c[1] == "/v1/chat/completions"]
    assert len(asked) == 1, "the flags cannot change while the process lives"

    # a restart may bring other flags, so a new start is asked again
    server["replies"][("get", "/metrics")] = _Reply(text="process_start_time_seconds 1.8e+09\n")
    server["replies"][("post", "/v1/chat/completions")] = _Reply(400)
    assert vllm.tool_calls_probed(SPEC, "Qwen/Q") is False, "refusing auto means no parser"


def test_a_probe_that_answers_in_text_says_no_and_a_silent_server_says_nothing(server):
    server["replies"][("get", "/metrics")] = _Reply(text=START)
    server["replies"][("post", "/v1/chat/completions")] = _Reply(
        body={"choices": [{"message": {"content": "search_corpus(redis)", "tool_calls": None}}]}
    )
    assert vllm.tool_calls_probed(SPEC, "Qwen/Q") is False
    vllm._probed.clear()
    server["replies"][("post", "/v1/chat/completions")] = _Reply(503)
    assert vllm.tool_calls_probed(SPEC, "Qwen/Q") is None, "unknown must not be cached as no"
    assert vllm._probed == {}
    server["replies"][("get", "/metrics")] = _Reply(404, text="")
    assert vllm.tool_calls_probed(SPEC, "Qwen/Q") is None, "no start, no process to vouch for"


def test_the_bootstrap_puts_only_an_awake_vllm_to_sleep(monkeypatch):
    # vLLM takes the card first when the stack comes up, and ollama roles would land half on the cpu
    import bootstrap
    from engines import card

    state, slept = {}, []
    other = engines.EngineSpec(4, "vllm-2", EngineKind.vllm, "VLLM_2", Placement.gpu)
    monkeypatch.setattr(card, "card_engines", lambda kind=None: [SPEC, other])
    monkeypatch.setattr(card.vllm, "card_state", lambda spec: state[spec.name])
    monkeypatch.setattr(card.vllm, "sleep", lambda spec: slept.append(spec.name))
    monkeypatch.setattr(bootstrap.job_queue, "running_in_lane", lambda lane: False)

    state.update({"vllm": "awake", "vllm-2": "asleep"})
    bootstrap._put_vllm_to_sleep()
    assert slept == ["vllm"]

    # a stopped service under a profile holds no card; a silent one may, so it is put to sleep too
    state.update({"vllm": "down", "vllm-2": "unknown"})
    bootstrap._put_vllm_to_sleep()
    assert slept == ["vllm", "vllm-2"]

    def refuses(spec):
        raise RuntimeError("did not go to sleep")

    monkeypatch.setattr(card.vllm, "sleep", refuses)
    state["vllm"] = "awake"
    with pytest.raises(RuntimeError):
        bootstrap._put_vllm_to_sleep()


def test_the_bootstrap_puts_vllm_to_sleep_before_anything_touches_ollama(monkeypatch):
    import bootstrap

    order = []
    for step in ("_put_vllm_to_sleep", "_ensure_models", "_ensure_roles", "_reconcile_with_ollama",
                 "_fill_vllm_rows",
                 "_ensure_index", "_ensure_vector_indexes", "_repair_served_vector_index",
                 "_ensure_question_embeddings"):
        monkeypatch.setattr(bootstrap, step, lambda *a, _s=step, **kw: order.append(_s))
    monkeypatch.setattr(bootstrap, "_seeded", lambda: object())
    monkeypatch.setattr(bootstrap.engines, "registered", lambda: [])
    bootstrap.bootstrap_models()
    assert order[0] == "_put_vllm_to_sleep", order


def test_a_stamp_carries_only_what_a_probe_already_said(server):
    server["replies"][("get", "/metrics")] = _Reply(text=START)
    added = engines.added_by(SPEC, "Qwen/Q")
    assert "tool_calls_probed" not in added
    assert not [c for c in server["calls"] if c[1] == "/v1/chat/completions"], "no request mid-pass"
    server["replies"][("post", "/v1/chat/completions")] = _Reply(
        body={"choices": [{"message": {"tool_calls": [{"id": "1"}]}}]}
    )
    vllm.tool_calls_probed(SPEC, "Qwen/Q")
    assert engines.added_by(SPEC, "Qwen/Q")["tool_calls_probed"] is True


def test_the_generation_role_refuses_a_vllm_that_answers_tools_in_text(monkeypatch):
    from models.registry import Role
    from use_cases import model_acceptance as gate

    cpu = engines.EngineSpec(4, "vllm-cpu", EngineKind.vllm, "VLLM_CPU", Placement.cpu)
    seen = {"engine": SPEC, "quant": "AWQ", "probed": False}
    monkeypatch.setattr(gate, "_engine_of", lambda name: seen["engine"])
    monkeypatch.setattr(vllm, "artifact_of", lambda repo: {"quant": seen["quant"]})
    monkeypatch.setattr(vllm, "tool_calls_probed", lambda spec, model: seen["probed"])
    monkeypatch.setattr(vllm, "card_state", lambda spec: "awake")
    monkeypatch.setattr(vllm, "pools", lambda spec: False)
    monkeypatch.setattr(vllm, "started_at", lambda spec: "t0")

    with pytest.raises(ValueError, match="does not return tool calls"):
        gate.refuse_unfit_model(Role.generation, "Qwen/Q")
    gate.refuse_unfit_model(Role.judging, "Qwen/Q")
    seen["probed"] = None
    gate.refuse_unfit_model(Role.generation, "Qwen/Q")

    # the cpu build has no kernels for AWQ: the role is refused before the handover wastes a card
    seen.update(engine=cpu, probed=True)
    for unreadable in ("AWQ", "GGUF"):
        seen["quant"] = unreadable
        with pytest.raises(ValueError, match=f"{unreadable}, which vllm-cpu on the cpu cannot serve"):
            gate.refuse_unfit_model(Role.judging, "Qwen/Q")
    seen["quant"] = "BF16"
    gate.refuse_unfit_model(Role.judging, "Qwen/Q")


def test_a_refused_connection_holds_no_card_and_a_timeout_may(server, monkeypatch):
    server["replies"][("get", "/is_sleeping")] = _Reply(body={"is_sleeping": False})
    assert vllm.card_state(SPEC) == "awake"
    server["replies"][("get", "/is_sleeping")] = _Reply(body={"is_sleeping": True})
    assert vllm.card_state(SPEC) == "asleep"
    server["replies"][("get", "/is_sleeping")] = vllm.requests.ConnectionError("refused")
    assert vllm.card_state(SPEC) == "down"
    server["replies"][("get", "/is_sleeping")] = vllm.requests.Timeout("10 s")
    assert vllm.card_state(SPEC) == "unknown"
    # a ConnectTimeout is a ConnectionError too, yet the host may still hold the card
    server["replies"][("get", "/is_sleeping")] = vllm.requests.ConnectTimeout("no route")
    assert vllm.card_state(SPEC) == "unknown"
    monkeypatch.delenv("VLLM_BASE_URL")
    assert vllm.card_state(SPEC) == "down", "an engine with no address runs nowhere"


def test_the_window_is_the_one_the_server_started_with_for_that_model(server):
    server["replies"][("get", "/v1/models")] = _Reply(body={"data": [
        {"id": "other", "max_model_len": 4096}, {"id": "Qwen/Q", "max_model_len": 8192}]})
    assert vllm.max_model_len(SPEC, "Qwen/Q") == 8192
    assert vllm.max_model_len(SPEC, "absent") is None
    server["replies"][("get", "/v1/models")] = _Reply(503)
    assert vllm.max_model_len(SPEC, "Qwen/Q") is None, "silence is not a window"


def test_the_bootstrap_reads_every_ollama_and_pulls_only_where_the_server_answered(monkeypatch):
    # rows on `ollama-cpu` got no status and no pull while the seeded ollama was the only one read
    import bootstrap

    seeded = engines.EngineSpec(1, "ollama", EngineKind.ollama, "OLLAMA", Placement.gpu)
    cpu = engines.EngineSpec(5, "ollama-cpu", EngineKind.ollama, "OLLAMA_CPU", Placement.cpu)
    read = []
    for step in ("_put_vllm_to_sleep", "_ensure_models", "_ensure_roles", "_fill_vllm_rows",
                 "_ensure_index", "_ensure_vector_indexes", "_repair_served_vector_index",
                 "_ensure_question_embeddings"):
        monkeypatch.setattr(bootstrap, step, lambda *a, **kw: None)
    monkeypatch.setattr(bootstrap, "_seeded", lambda: seeded)
    monkeypatch.setattr(bootstrap.engines, "registered", lambda: [seeded, SPEC, cpu])
    monkeypatch.setattr(bootstrap, "_reconcile_with_ollama",
                        lambda spec, pull_when_silent=True: read.append((spec.name, pull_when_silent)))
    bootstrap.bootstrap_models()
    assert read == [("ollama", True), ("ollama-cpu", False)]


def test_a_silent_second_ollama_keeps_its_rows_as_they_were(monkeypatch):
    import bootstrap

    cpu = engines.EngineSpec(5, "ollama-cpu", EngineKind.ollama, "OLLAMA_CPU", Placement.cpu)

    def down(spec):
        raise ConnectionError("refused")

    monkeypatch.setattr(bootstrap.ollama, "list_models", down)
    monkeypatch.setattr(bootstrap, "Session", lambda: pytest.fail("rows touched on silence"))
    bootstrap._reconcile_with_ollama(cpu, pull_when_silent=False)


def test_a_run_records_the_window_of_a_vllm_generator_from_the_server(monkeypatch):
    # the first run with the generator on vLLM recorded `context_length: null`
    from use_cases import run_snapshot

    monkeypatch.setattr("engines.vllm.max_model_len", lambda spec, name: 8192)
    monkeypatch.setattr("engines.ollama.context_length",
                        lambda *a, **kw: pytest.fail("ollama asked about a vLLM generator"))
    assert run_snapshot._window(engines.Resolved("Qwen/Q", SPEC)) == 8192


def _asleep_gate(monkeypatch, state):
    from engines import lookup
    from use_cases import model_acceptance as gate

    asked, recorded = [], {}
    monkeypatch.setattr(gate, "_engine_of", lambda name: SPEC)
    monkeypatch.setattr(vllm, "artifact_of", lambda repo: {"quant": "AWQ"})
    monkeypatch.setattr(vllm, "card_state", lambda spec: state[0])
    monkeypatch.setattr(vllm, "pools", lambda spec: state[1] if len(state) > 1 else False)
    monkeypatch.setattr(vllm, "started_at", lambda spec: "t0")
    monkeypatch.setattr(vllm, "_probe", lambda spec, model: asked.append(model) or False)
    monkeypatch.setattr(lookup, "recorded_tool_probe",
                        lambda engine_id, model, started: recorded.get(started))
    monkeypatch.setattr(lookup, "record_tool_probe",
                        lambda engine_id, model, started, probed: recorded.__setitem__(started, probed))
    vllm._probed.clear()
    return gate, asked, recorded


def test_the_gate_does_not_ask_an_asleep_server_and_reads_the_recorded_probe(monkeypatch):
    # the probe is read from the model row, which every process sees
    from models.registry import Role

    state = ["asleep"]
    gate, asked, recorded = _asleep_gate(monkeypatch, state)
    with pytest.raises(gate.NeedsProbe):
        gate.refuse_unfit_model(Role.generation, "Qwen/Q")
    assert asked == [], "an asleep server is not probed"
    recorded["t0"] = False
    with pytest.raises(ValueError, match="does not return tool calls"):
        gate.refuse_unfit_model(Role.generation, "Qwen/Q")
    recorded.clear()
    vllm._probed.clear()
    state[0] = "awake"
    with pytest.raises(ValueError):
        gate.refuse_unfit_model(Role.generation, "Qwen/Q")
    assert asked == ["Qwen/Q"] and recorded == {"t0": False}, "asked awake, and written for the next"
    vllm._probed.clear()
    with pytest.raises(ValueError):
        gate.refuse_unfit_model(Role.generation, "Qwen/Q")
    assert asked == ["Qwen/Q"], "another process reads the record instead of asking again"


def test_a_role_goes_only_where_its_server_runs_and_answers(monkeypatch):
    # the reranker on a stopped `vllm-rerank` was seated as the generator with a 200
    from models.registry import Role

    state = ["down"]
    gate, _, recorded = _asleep_gate(monkeypatch, state)
    with pytest.raises(gate.EngineDown):
        gate.refuse_unfit_model(Role.generation, "BAAI/r")
    # a timeout is refused as a refusal is, not queued as a probe that would fail its handover
    state[:] = ["unknown"]
    with pytest.raises(gate.EngineDown, match="unknown"):
        gate.refuse_unfit_model(Role.generation, "BAAI/r")
    state[:] = ["awake", True]
    with pytest.raises(ValueError, match="pooling runner"):
        gate.refuse_unfit_model(Role.generation, "BAAI/r")
    gate.refuse_unfit_model(Role.reranking, "BAAI/r")
    state[:] = ["asleep", False]
    with pytest.raises(ValueError, match="generating runner"):
        gate.refuse_unfit_model(Role.reranking, "Qwen/Q")
    # a server that cannot say how it runs is not refused on that
    state[:] = ["asleep", None]
    recorded["t0"] = True
    gate.refuse_unfit_model(Role.generation, "Qwen/Q")


def test_a_role_on_a_stopped_ollama_is_refused_like_one_on_a_stopped_vllm(monkeypatch):
    # a stopped ollama is refused a role as a stopped vLLM is
    from models.registry import Role
    from use_cases import model_acceptance as gate

    ollama = engines.EngineSpec(1, "ollama", EngineKind.ollama, "OLLAMA", Placement.gpu)
    monkeypatch.setattr(engines, "spec_of_id", lambda engine_id: ollama)
    state = ["down"]
    monkeypatch.setattr(gate.ollama, "card_reading", lambda spec: (state[0], []))
    monkeypatch.setattr(gate.ollama, "shown", lambda model, spec=None: {})
    with pytest.raises(gate.EngineDown, match="ollama is down"):
        gate.refuse_unfit_model(Role.generation, "llama3.1:8b", engine_id=1)
    # silent past its timeout reads as the handover reads it: nothing is seated there either
    state[0] = "unknown"
    with pytest.raises(gate.EngineDown, match="ollama is unknown"):
        gate.refuse_unfit_model(Role.generation, "llama3.1:8b", engine_id=1)
    state[0] = "free"
    gate.refuse_unfit_model(Role.generation, "llama3.1:8b", engine_id=1)


def test_the_server_says_whether_it_pools(server):
    info = "model='Qwen/Q', quantization=auto_awq, pooler_config={}, compilation_config=..."
    server["replies"][("get", "/server_info")] = _Reply(body={"vllm_config": info.format("None")})
    assert vllm.pools(SPEC) is False
    server["replies"][("get", "/server_info")] = _Reply(
        body={"vllm_config": info.format("PoolerConfig(pooling_type='CLS')")})
    assert vllm.pools(SPEC) is True
    server["replies"][("get", "/server_info")] = _Reply(body={"vllm_config": "model='x'"})
    assert vllm.pools(SPEC) is None, "a config that names no pooler says nothing"
    server["replies"][("get", "/server_info")] = _Reply(404)
    assert vllm.pools(SPEC) is None


def test_a_recorded_probe_speaks_only_for_the_process_start_it_was_asked_at(monkeypatch):
    from types import SimpleNamespace

    from engines import lookup

    class _Session:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def execute(self, stmt):
            return SimpleNamespace(first=lambda: SimpleNamespace(tool_probe=False,
                                                                 tool_probe_start="t0"))

    monkeypatch.setattr(lookup, "Session", _Session)
    assert lookup.recorded_tool_probe(SPEC.id, "Qwen/Q", "t0") is False
    assert lookup.recorded_tool_probe(SPEC.id, "Qwen/Q", "t1") is None, "a restart may change flags"
    vllm._probed.clear()
    assert vllm.known_probe(SPEC, "Qwen/Q", None) is None, "no start, no process to vouch for"


def test_the_role_door_on_an_asleep_server_queues_the_probe_and_a_down_one_is_503(monkeypatch):
    import asyncio
    from types import SimpleNamespace

    from api.v1 import model_role
    from fastapi import HTTPException
    from models.registry import Role

    class _Session:
        async def get(self, cls, ident):
            return SimpleNamespace(id=10, name="Qwen/Q", engine_id=3) if ident == 10 else None

    queued = []
    monkeypatch.setattr(model_role.job_queue, "pending_of_type", lambda *a, **kw: None)
    monkeypatch.setattr(model_role.job_queue, "enqueue",
                        lambda kind, options: queued.append((kind, options)) or 77)
    fault = [None]

    def refuse(role, name, engine_id):
        assert engine_id == 3, "the door knows the engine and says it"
        raise fault[0]

    monkeypatch.setattr(model_role.model_acceptance, "refuse_unfit_model", refuse)
    request = model_role.RoleAssignRequest(model_id=10)

    fault[0] = model_role.model_acceptance.NeedsProbe("vllm is asleep")
    answer = asyncio.run(model_role.assign_role(Role.generation, request, _Session()))
    assert answer.status_code == 202 and b'"job_id":77' in answer.body
    assert queued == [("hand_card", {"engine_id": 3, "model": "Qwen/Q", "seat": "generation",
                                     "seat_over": None})], "no role yet, so nothing to be overtaken"

    # a seat already waiting answers the second ask with its own job
    monkeypatch.setattr(model_role.job_queue, "pending_of_type", lambda *a, **kw: 55)
    again = asyncio.run(model_role.assign_role(Role.generation, request, _Session()))
    assert b'"job_id":55' in again.body and len(queued) == 1
    monkeypatch.setattr(model_role.job_queue, "pending_of_type", lambda *a, **kw: None)

    fault[0] = model_role.model_acceptance.EngineDown("vllm-rerank does not answer")
    with pytest.raises(HTTPException) as down:
        asyncio.run(model_role.assign_role(Role.generation, request, _Session()))
    assert down.value.status_code == 503 and len(queued) == 1, "nothing is queued for a dead server"


def test_an_unexpected_probe_body_reads_unknown_not_a_500(server):
    server["replies"][("get", "/metrics")] = _Reply(text=START)
    server["replies"][("post", "/v1/chat/completions")] = _Reply(body={"object": "error"})
    assert vllm.tool_calls_probed(SPEC, "Qwen/Q") is None


def test_a_handover_that_seats_a_role_asks_the_woken_server_first(monkeypatch):
    from job_handlers import card as handler

    calls = []
    monkeypatch.setattr(handler.engines, "spec_of_id", lambda engine_id: SPEC)
    monkeypatch.setattr(handler, "take", lambda spec, model=None: calls.append("take"))
    monkeypatch.setattr(handler.model_acceptance, "seat",
                        lambda role, engine_id, model, over: calls.append(f"seat {role} {model} {over}"))
    verdict = [None]

    def gate(role, model, engine_id):
        calls.append(f"gate {engine_id}")
        if verdict[0]:
            raise verdict[0]

    monkeypatch.setattr(handler.model_acceptance, "refuse_unfit_model", gate)
    options = {"engine_id": 3, "model": "Qwen/Q", "seat": "generation", "seat_over": 4}
    handler.hand_card(options)
    assert calls == ["take", "gate 3", "seat generation Qwen/Q 4"]
    calls.clear()
    verdict[0] = ValueError("Qwen/Q on vllm does not return tool calls")
    # a retry would only wake the server and take the card again for the same answer
    with pytest.raises(handler.Final, match="tool calls"):
        handler.hand_card(options)
    assert calls == ["take", "gate 3"], "a refused role stays where it was"
    calls.clear()
    handler.hand_card({"engine_id": 3, "model": "Qwen/Q"})
    assert calls == ["take"], "a plain `/load` seats nothing"


def test_the_worker_probes_the_generator_it_just_woke(monkeypatch):
    from job_handlers import card as handler

    probed = []
    monkeypatch.setattr(handler.llm, "resolve", lambda role: engines.Resolved("Qwen/Q", SPEC))
    monkeypatch.setattr(handler.vllm, "tool_calls_probed", lambda spec, model: probed.append(model))
    handler._probe_the_woken_generator(SPEC)
    other = engines.EngineSpec(7, "vllm-rerank", EngineKind.vllm, "VLLM_RERANK", Placement.gpu)
    handler._probe_the_woken_generator(other)
    assert probed == ["Qwen/Q"], "a pooling server asked for tool calls would read as parserless"


def test_a_rerun_bootstrap_leaves_the_card_to_a_running_job(monkeypatch):
    # `compose --profile rerank up` reran the bootstrap and put a judging judge to sleep
    import bootstrap
    from engines import card

    slept = []
    monkeypatch.setattr(card, "sleep_every_vllm", lambda: slept.append(1))
    monkeypatch.setattr(bootstrap.job_queue, "running_in_lane", lambda lane: True)
    bootstrap._put_vllm_to_sleep()
    monkeypatch.setattr(bootstrap.job_queue, "running_in_lane", lambda lane: False)
    bootstrap._put_vllm_to_sleep()
    assert slept == [1]
