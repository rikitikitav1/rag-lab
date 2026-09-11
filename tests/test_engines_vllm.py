import engines
import pytest
from engines import vllm
from models.registry import EngineKind, Placement

SPEC = engines.EngineSpec(3, "vllm", EngineKind.vllm, "VLLM", Placement.gpu)


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
    # without VLLM_SERVER_DEV_MODE the route is absent, which says nothing about the card
    server["replies"][("get", "/is_sleeping")] = _Reply(404)
    assert vllm.is_sleeping(SPEC) is None


def test_a_wake_that_the_server_refused_is_an_error_not_a_quiet_retry(server):
    # 11.09: a wake on a card ollama still held answered 500 and left the server asleep
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

    state, slept = {}, []
    other = engines.EngineSpec(4, "vllm-2", EngineKind.vllm, "VLLM_2", Placement.gpu)
    monkeypatch.setattr(bootstrap.engines, "card_engines", lambda kind=None: [SPEC, other])
    monkeypatch.setattr(bootstrap.vllm, "is_sleeping", lambda spec: state[spec.name])
    monkeypatch.setattr(bootstrap.vllm, "sleep", lambda spec: slept.append(spec.name))

    state.update({"vllm": False, "vllm-2": True})
    bootstrap._put_vllm_to_sleep()
    assert slept == ["vllm"]

    # a server that does not answer holds no card, so the stack still comes up
    state.update({"vllm": None, "vllm-2": None})
    bootstrap._put_vllm_to_sleep()
    assert slept == ["vllm"]

    def refuses(spec):
        raise RuntimeError("did not go to sleep")

    monkeypatch.setattr(bootstrap.vllm, "sleep", refuses)
    state["vllm"] = False
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

    with pytest.raises(ValueError, match="does not return tool calls"):
        gate.refuse_unfit_model(Role.generation, "Qwen/Q")
    gate.refuse_unfit_model(Role.judging, "Qwen/Q")
    seen["probed"] = None
    gate.refuse_unfit_model(Role.generation, "Qwen/Q")

    # the cpu build has no kernels for AWQ: the role is refused before the handover wastes a card
    seen.update(engine=cpu, probed=True)
    with pytest.raises(ValueError, match="cannot serve"):
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
    # 11.09: the first run with the generator on vLLM recorded `context_length: null`
    from use_cases import run_snapshot

    monkeypatch.setattr(run_snapshot.vllm, "max_model_len", lambda spec, name: 8192)
    monkeypatch.setattr(run_snapshot.ollama, "context_length",
                        lambda *a, **kw: pytest.fail("ollama asked about a vLLM generator"))
    assert run_snapshot._window(engines.Resolved("Qwen/Q", SPEC)) == 8192
