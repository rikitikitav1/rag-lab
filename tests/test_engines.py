from contextlib import contextmanager

import engines
import llm
import pytest
from engines import core, lookup
from models.registry import EngineKind, Placement

OLLAMA = engines.EngineSpec(1, "ollama", EngineKind.ollama, "OLLAMA", Placement.gpu)
SECOND = engines.EngineSpec(2, "ollama2", EngineKind.ollama, "OLLAMA2", Placement.gpu)
VLLM = engines.EngineSpec(3, "vllm", EngineKind.vllm, "VLLM", Placement.gpu)
CLOUD = engines.EngineSpec(4, "cloud", EngineKind.openai_compatible, "CLOUD", Placement.remote)


@pytest.fixture(autouse=True)
def _clean_cache():
    engines.forget_clients()
    yield
    engines.forget_clients()


def _rows_are(monkeypatch, rows):
    @contextmanager
    def session():
        class _Result:
            def first(self):
                return rows[0] if rows else None

            def all(self):
                return rows

        class _Session:
            def execute(self, _stmt):
                return _Result()

        yield _Session()

    monkeypatch.setattr(lookup, "Session", session)


def test_a_client_is_built_once_per_engine(monkeypatch):
    monkeypatch.setenv("VLLM_BASE_URL", "http://vllm:8000")
    assert engines.client_for(OLLAMA) is engines.client_for(OLLAMA)
    assert engines.client_for(OLLAMA) is not engines.client_for(VLLM)


def test_the_address_comes_from_the_prefix_not_the_row(monkeypatch):
    monkeypatch.setenv("VLLM_BASE_URL", "http://elsewhere:9000")
    assert engines.base_url(VLLM) == "http://elsewhere:9000"


def test_only_the_seeded_engine_falls_back_to_the_config(monkeypatch):
    # a typo in a second engine's prefix used to address the first one and stamp the second's name
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    monkeypatch.delenv("OLLAMA2_BASE_URL", raising=False)
    assert engines.base_url(OLLAMA) == llm.LLM_BASE
    with pytest.raises(core.Unconfigured, match="address is not configured"):
        engines.base_url(SECOND)


def test_an_engine_without_an_address_refuses(monkeypatch):
    monkeypatch.delenv("VLLM_BASE_URL", raising=False)
    with pytest.raises(core.Unconfigured):
        engines.base_url(VLLM)


def test_an_address_carrying_a_key_refuses_before_the_first_call(monkeypatch):
    monkeypatch.setenv("VLLM_BASE_URL", "https://user:secret@host:8000")
    with pytest.raises(core.Unconfigured, match="carries credentials"):
        engines.base_url(VLLM)


def test_an_address_we_do_not_speak_refuses(monkeypatch):
    monkeypatch.setenv("VLLM_BASE_URL", "file:///etc/passwd")
    with pytest.raises(core.Unconfigured, match="not an http url"):
        engines.base_url(VLLM)


def test_a_paid_engine_without_a_key_refuses_before_the_call(monkeypatch):
    monkeypatch.delenv("CLOUD_API_KEY", raising=False)
    with pytest.raises(core.Unconfigured):
        engines.api_key(CLOUD)


def test_no_refusal_names_the_variable_the_secret_lives_in(monkeypatch):
    # the text reaches `jobs.error` and the open job route, so it says what, not where
    monkeypatch.delenv("CLOUD_API_KEY", raising=False)
    with pytest.raises(core.Unconfigured) as caught:
        engines.api_key(CLOUD)
    assert "CLOUD_API_KEY" not in str(caught.value)


def test_a_local_engine_needs_no_key(monkeypatch):
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
    assert engines.api_key(OLLAMA) == "ollama", "the wire carried this before the layer"


def test_the_stamped_address_carries_no_userinfo(monkeypatch):
    monkeypatch.setenv("VLLM_BASE_URL", "http://vllm:8000")
    assert engines.address_of(VLLM) == "vllm:8000"


def test_what_the_engine_will_not_carry_is_named_rather_than_dropped_silently(monkeypatch):
    monkeypatch.setitem(engines.ACCEPTS, EngineKind.vllm, frozenset({"temperature"}))
    got = engines.translate(VLLM, {"temperature": 0, "seed": 0, "max_tokens": 512})
    assert got.sent == {"temperature": 0}
    assert got.dropped == {"seed": 0, "max_tokens": 512}


def test_todays_engine_carries_the_whole_sampler():
    # wire equality holds only while nothing is dropped, and this is the assertion behind it
    got = engines.translate(OLLAMA, {"temperature": 0, "seed": 0, "max_tokens": 512})
    assert got.dropped == {}


def test_a_name_on_two_engines_refuses_instead_of_picking_one(monkeypatch):
    _rows_are(monkeypatch, [
        (1, "ollama", EngineKind.ollama, "OLLAMA", Placement.gpu),
        (3, "vllm", EngineKind.vllm, "VLLM", Placement.gpu),
    ])
    with pytest.raises(engines.Ambiguous, match="ollama, vllm"):
        engines.find_model("qwen2.5:7b")


def test_a_name_on_one_engine_takes_that_engine(monkeypatch):
    _rows_are(monkeypatch, [(3, "vllm", EngineKind.vllm, "VLLM", Placement.gpu)])
    assert engines.find_model("qwen2.5:7b").engine.name == "vllm"


def test_a_name_nowhere_is_absent_rather_than_an_error(monkeypatch):
    _rows_are(monkeypatch, [])
    assert engines.find_model("never-registered") is None


def test_an_unregistered_name_runs_on_the_engine_of_the_role(monkeypatch):
    _rows_are(monkeypatch, [])
    monkeypatch.setattr(llm, "resolve", lambda role: engines.Resolved("configured", VLLM))
    picked = llm.resolve_for("generation", "never-registered")
    assert (picked.name, picked.engine.name) == ("never-registered", "vllm")


def test_a_second_ollama_is_a_refusal_and_not_a_pick(monkeypatch):
    # the config's models belong to ollama by kind, and two of them is a question, not a default
    _rows_are(monkeypatch, [
        (1, "ollama", EngineKind.ollama, "OLLAMA", Placement.gpu),
        (2, "ollama2", EngineKind.ollama, "OLLAMA2", Placement.gpu),
    ])
    with pytest.raises(engines.Unnamed, match="2 ollama engines"):
        engines.seeded_ollama()


def test_a_table_that_cannot_be_read_says_so_rather_than_saying_empty(monkeypatch):
    def boom():
        raise RuntimeError("no database")

    monkeypatch.setattr(lookup, "Session", boom)
    assert engines.registered_names() is None


def test_a_row_from_before_the_engines_table_reads_as_unnamed():
    # the pair that proves the migration changed nothing is exactly this pair, and it must join
    got = engines.engine_of({"engine": "ollama:11434"}, {"ollama"})
    assert (got.name, got.address, got.state) == (None, "ollama:11434", engines.UNNAMED)


def test_a_named_engine_still_registered_reads_as_named():
    got = engines.engine_of({"engine": "ollama:11434", "engine_name": "ollama"}, {"ollama"})
    assert got.state == engines.NAMED


def test_a_name_whose_engine_was_deleted_does_not_read_as_never_recorded():
    got = engines.engine_of({"engine": "ollama:11434", "engine_name": "ollama"}, set())
    assert (got.name, got.state) == ("ollama", engines.DANGLING)


def test_an_unreadable_table_never_calls_a_live_engine_deleted():
    got = engines.engine_of({"engine": "x:1", "engine_name": "ollama"}, None)
    assert got.state == engines.NAMED


def test_ollama_adds_the_window_it_was_started_with(monkeypatch):
    from engines import ollama

    monkeypatch.setattr(ollama, "context_length", lambda model, spec=None: 8192)
    assert engines.added_by(OLLAMA, "qwen2.5:7b") == {"num_ctx": 8192}


def test_a_server_that_answers_nothing_leaves_the_key_out(monkeypatch):
    # absent and null read the same to a later reader, and only one of them is honest
    from engines import ollama

    monkeypatch.setattr(ollama, "context_length", lambda model, spec=None: None)
    assert engines.added_by(OLLAMA, "qwen2.5:7b") == {}


def test_vllm_adds_its_window_and_its_version(monkeypatch):
    from types import SimpleNamespace

    served = SimpleNamespace(id="m", max_model_len=8192, model_extra={})
    monkeypatch.setattr(core, "client_for", lambda _spec: SimpleNamespace(
        models=SimpleNamespace(list=lambda: SimpleNamespace(data=[served]))
    ))
    monkeypatch.setattr(core, "_asked", lambda spec, path, key: "0.29.1")
    assert engines.added_by(VLLM, "m") == {"max_model_len": 8192, "engine_version": "0.29.1"}


def test_the_stamp_says_whether_the_server_had_batch_invariance_on(monkeypatch):
    # measured on 0.29.1: `max_num_seqs` is nowhere in the server's answers, this flag is
    from types import SimpleNamespace

    monkeypatch.setattr(core, "client_for", lambda _spec: SimpleNamespace(
        models=SimpleNamespace(list=lambda: SimpleNamespace(data=[]))
    ))
    monkeypatch.setattr(core, "_asked", lambda spec, path, key: (
        {"VLLM_BATCH_INVARIANT": True} if path == "/server_info" else None
    ))
    assert engines.added_by(VLLM, "m") == {"batch_invariant": True}


def test_a_server_without_dev_mode_leaves_the_flag_out_rather_than_calling_it_off(monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setattr(core, "client_for", lambda _spec: SimpleNamespace(
        models=SimpleNamespace(list=lambda: SimpleNamespace(data=[]))
    ))
    # `/server_info` answers 404 unless VLLM_SERVER_DEV_MODE is set, and absent is not False
    monkeypatch.setattr(core, "_asked", lambda spec, path, key: None)
    assert "batch_invariant" not in engines.added_by(VLLM, "m")


def test_an_unreachable_engine_adds_nothing_rather_than_failing_the_pass(monkeypatch):
    def boom(_spec):
        raise RuntimeError("down")

    monkeypatch.setattr(core, "client_for", boom)
    monkeypatch.setattr(core, "_asked", lambda spec, path, key: None)
    assert engines.added_by(VLLM, "m") == {}


def test_an_override_naming_a_model_on_two_engines_takes_the_role_own_engine(monkeypatch):
    # `qwen2.5:7b` sits on two ollama rows, and a bench naming it used to fail the whole pass
    _rows_are(monkeypatch, [
        (1, "ollama", EngineKind.ollama, "OLLAMA", Placement.gpu),
        (5, "ollama-cpu", EngineKind.ollama, "OLLAMA_CPU", Placement.cpu),
    ])
    monkeypatch.setattr(llm, "resolve", lambda role: engines.Resolved("qwen2.5:7b", OLLAMA))

    asked = []

    def _one(name, engine_id=None):
        asked.append(engine_id)
        if engine_id is None:
            raise engines.Ambiguous("two engines")
        return engines.Resolved(name, SECOND)

    monkeypatch.setattr(llm.engines, "find_model", _one)
    picked = llm.resolve_for("judging", "qwen2.5:7b")

    # the second ask names the role's engine, and its answer is what the pass must use
    assert asked == [None, OLLAMA.id], "the retry must name the role's engine, not guess"
    assert picked.engine is SECOND, "the row found on that engine wins, not a fabricated pair"
