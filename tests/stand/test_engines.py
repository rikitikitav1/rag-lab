from contextlib import contextmanager

import config
import engines
import llm
import pytest
from engines import core, lookup
from models.registry import EngineKind, Placement
from stand_specs import OLLAMA, VLLM

SECOND = engines.EngineSpec(2, "ollama2", EngineKind.ollama, "OLLAMA2", Placement.gpu)
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
    assert engines.base_url(OLLAMA) == config.settings.llm.base_url
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
        ("none", {}, 1, "ollama", EngineKind.ollama, "OLLAMA", Placement.gpu),
        ("none", {}, 3, "vllm", EngineKind.vllm, "VLLM", Placement.gpu),
    ])
    with pytest.raises(engines.Ambiguous, match="ollama, vllm"):
        engines.find_model("qwen2.5:7b")


def test_a_name_on_one_engine_takes_that_engine(monkeypatch):
    # the row's parser comes first and rides along, so every call cuts the answer the row's way
    _rows_are(monkeypatch, [("think_tags", {}, 3, "vllm", EngineKind.vllm, "VLLM", Placement.gpu)])
    found = engines.find_model("qwen2.5:7b")
    assert (found.engine.name, found.parser) == ("vllm", "think_tags")


def test_a_name_nowhere_is_absent_rather_than_an_error(monkeypatch):
    _rows_are(monkeypatch, [])
    assert engines.find_model("never-registered") is None


def test_an_unregistered_name_runs_on_the_engine_of_the_role(monkeypatch):
    _rows_are(monkeypatch, [])
    monkeypatch.setattr(llm, "resolve", lambda role: engines.Resolved("configured", VLLM))
    picked = llm.resolve_for("generation", "never-registered")
    assert (picked.name, picked.engine.name) == ("never-registered", "vllm")


def test_a_second_ollama_does_not_unseat_the_seeded_one_but_a_guess_is_refused(monkeypatch):
    # with `ollama-cpu` registered the bootstrap skipped the seeded ollama's models
    _rows_are(monkeypatch, [
        (1, "ollama", EngineKind.ollama, "OLLAMA", Placement.gpu),
        (5, "ollama-cpu", EngineKind.ollama, "OLLAMA_CPU", Placement.cpu),
    ])
    assert engines.seeded_ollama().name == "ollama"
    # without the seeded prefix two ollamas are still a question, not a default
    _rows_are(monkeypatch, [
        (1, "ollama1", EngineKind.ollama, "OLLAMA1", Placement.gpu),
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


def test_a_name_on_two_engines_is_taken_from_the_role_s_engine_or_refused(monkeypatch):
    # the fallback registered the name again on the seeded ollama and hit the unique key
    from job_handlers import base
    from models.registry import Status

    def find(name, engine_id=None):
        if engine_id is None:
            raise engines.Ambiguous(f"{name} sits on two engines")
        return engines.Resolved(name, OLLAMA) if engine_id == OLLAMA.id else None

    class _Session:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def scalar(self, _stmt):
            return Status.ready

        def add(self, _row):
            pytest.fail("an ambiguous name is never registered again")

    monkeypatch.setattr(engines, "find_model", find)
    monkeypatch.setattr(engines.lookup, "find_model", find)
    monkeypatch.setattr(llm, "resolve", lambda role: engines.Resolved(
        "m", {"generation": OLLAMA, "judging": VLLM}[role]))
    monkeypatch.setattr(base, "Session", _Session)
    base.require_model_ready("bge-m3", "generation")
    with pytest.raises(engines.Ambiguous):
        base.require_model_ready("bge-m3", "judging")
    with pytest.raises(engines.Ambiguous):
        base.require_model_ready("bge-m3")


def test_a_name_with_a_double_dash_is_refused_at_every_door():
    # `a--b` would share the hub cache directory of `a/b`, and the boot died on such a row
    from models.registry import refuse_unknown_registry

    with pytest.raises(ValueError, match="share"):
        refuse_unknown_registry("org--x/model")


def test_ollama_adds_the_window_it_was_started_with(monkeypatch):
    from engines import ollama

    monkeypatch.setattr(ollama, "context_length", lambda model, spec=None: 8192)
    monkeypatch.setattr(ollama, "repetition_penalty_served", lambda model, spec=None: 1.1)
    monkeypatch.setattr(ollama, "server_version", lambda spec=None: "0.32.0")
    assert engines.added_by(OLLAMA, "qwen2.5:7b") == {"num_ctx": 8192, "repetition_penalty": 1.1,
                                                      "server_version": "0.32.0"}


def test_a_server_that_answers_nothing_leaves_the_key_out(monkeypatch):
    # absent and null read the same to a later reader, and only one of them is honest
    from engines import ollama

    monkeypatch.setattr(ollama, "context_length", lambda model, spec=None: None)
    monkeypatch.setattr(ollama, "repetition_penalty_served", lambda model, spec=None: None)
    monkeypatch.setattr(ollama, "server_version", lambda spec=None: None)
    assert engines.added_by(OLLAMA, "qwen2.5:7b") == {}


def test_vllm_adds_its_window_and_its_version(monkeypatch):
    from engines import vllm

    monkeypatch.setattr(vllm, "max_model_len", lambda spec, model: 8192)
    monkeypatch.setattr(core, "_asked", lambda spec, path, key: "0.29.1")
    assert engines.added_by(VLLM, "m") == {"max_model_len": 8192, "engine_version": "0.29.1"}


def test_the_stamp_says_whether_the_server_had_batch_invariance_on(monkeypatch):
    # measured on 0.29.1: `max_num_seqs` is nowhere in the server's answers, this flag is
    from engines import vllm

    monkeypatch.setattr(vllm, "max_model_len", lambda spec, model: None)
    monkeypatch.setattr(core, "_asked", lambda spec, path, key: (
        {"VLLM_BATCH_INVARIANT": True} if path == "/server_info" else None
    ))
    assert engines.added_by(VLLM, "m") == {"batch_invariant": True}


def test_the_stamp_says_the_dtype_the_server_loaded_rather_than_the_one_the_weights_declare(monkeypatch):
    # a pair "only the engine" could compare bf16 with F16, and nothing in the record said which
    from engines import vllm

    monkeypatch.setattr(vllm, "max_model_len", lambda spec, model: None)
    said = ("ModelConfig(model='Qwen/Q', dtype=torch.float16, quantization=auto_awq, seed=0), "
            "CacheConfig(kv_cache_dtype=auto), SpeculativeConfig(dtype=torch.bfloat16, quantization=None)")
    monkeypatch.setattr(core, "_asked", lambda spec, path, key: said if key == "vllm_config" else None)
    assert engines.added_by(VLLM, "m") == {"dtype": "float16", "quantization": "auto_awq", "kv_cache_dtype": "auto"}


def test_a_server_without_dev_mode_leaves_the_flag_out_rather_than_calling_it_off(monkeypatch):
    from engines import vllm

    monkeypatch.setattr(vllm, "max_model_len", lambda spec, model: None)
    # `/server_info` answers 404 unless VLLM_SERVER_DEV_MODE is set, and absent is not False
    monkeypatch.setattr(core, "_asked", lambda spec, path, key: None)
    assert "batch_invariant" not in engines.added_by(VLLM, "m")


def test_an_unreachable_engine_adds_nothing_rather_than_failing_the_pass(monkeypatch):
    from engines import vllm

    def boom(*a, **kw):
        raise vllm.requests.ConnectionError("down")

    monkeypatch.setattr(vllm.requests, "get", boom)
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

    monkeypatch.setattr(llm.engines.lookup, "find_model", _one)
    picked = llm.resolve_for("judging", "qwen2.5:7b")

    # the second ask names the role's engine, and its answer is what the pass must use
    assert asked == [None, OLLAMA.id], "the retry must name the role's engine, not guess"
    assert picked.engine is SECOND, "the row found on that engine wins, not a fabricated pair"


def test_an_embedder_is_loaded_by_an_empty_embed_and_a_generator_by_an_empty_generate(monkeypatch):
    # `/api/generate` on bge-m3 answers 400, so a handover naming the embedder failed
    from engines import ollama

    sent = []
    caps = {"bge-m3": ["embedding"], "llama3.1:8b": ["completion", "tools"]}
    monkeypatch.setattr(ollama, "shown", lambda model, spec=None: {"capabilities": caps[model]})
    monkeypatch.setattr(ollama, "post", lambda path, body, spec=None: sent.append((path, body)))
    monkeypatch.setattr(ollama, "context_length", lambda model, spec=None: None)
    ollama.load_into_memory("bge-m3")
    ollama.load_into_memory("llama3.1:8b")
    assert sent == [("/api/embed", {"model": "bge-m3", "input": []}),
                    ("/api/generate", {"model": "llama3.1:8b"})]


def test_an_ollama_card_reading_tells_a_stopped_server_from_a_silent_one(monkeypatch):
    # a stopped ollama read as a holder, and a handover to the judge waited 60 s
    import requests
    from engines import ollama

    class _Answer:
        def __init__(self, status, body=None):
            self.status_code, self.ok, self._body = status, status < 400, body
            self.text = "x" if body is not None or status >= 400 else ""

        def json(self):
            return self._body or {"error": "boom"}

    held = {"models": [{"name": "llama3.1:8b", "size": 2**30, "size_vram": 2**30}]}
    spilled = {"models": [{"name": "qwen2.5:7b", "size": 2**30, "size_vram": 0}]}
    for said, state in ((requests.ConnectionError("refused"), "down"),
                        (requests.ConnectTimeout("no route"), "unknown"),
                        (requests.ReadTimeout("slow"), "unknown"),
                        (_Answer(500), "unknown"),
                        (_Answer(200, held), "holds"),
                        (_Answer(200, spilled), "free")):
        def get(url, timeout, said=said):
            if isinstance(said, Exception):
                raise said
            return said

        monkeypatch.setattr(ollama.requests, "get", get)
        assert ollama.card_reading(OLLAMA)[0] == state, said
    # the residency is the card's own read
    assert ollama.residency(OLLAMA) == ollama.card_reading(OLLAMA)[1] != []
    # no address configured means the server runs nowhere
    assert ollama.card_reading(SECOND)[0] == "down"


def test_the_stamp_names_the_rules_the_judge_json_was_decoded_by(monkeypatch):
    # with free whitespace one reply wrote tabs until its limit, and nothing in the record said which rules held
    from engines import vllm

    monkeypatch.setattr(vllm, "max_model_len", lambda spec, model: None)
    said = ("ModelConfig(model='Qwen/Q', dtype=torch.float16, quantization=auto_awq), SchedulerConfig(backend='x'), "
            "StructuredOutputsConfig(backend='xgrammar', disable_any_whitespace=True, reasoning_parser='')")
    monkeypatch.setattr(core, "_asked", lambda spec, path, key: said if key == "vllm_config" else None)
    added = engines.added_by(VLLM, "m")
    assert added["json_backend"] == "xgrammar" and added["json_disable_any_whitespace"] == "True"


def test_the_penalty_ollama_applies_is_the_model_s_else_the_measured_default_of_its_version(monkeypatch):
    # 0.32.0 applied 1.1 where the Modelfile named none, byte for byte; a version never measured is unknown
    from engines import ollama

    monkeypatch.setattr(
        ollama, "shown", lambda model, spec=None: {"parameters": 'num_ctx 16384\nrepeat_penalty 1.2\nstop "<x>"'}
    )
    assert ollama.repetition_penalty_served("m") == 1.2
    monkeypatch.setattr(ollama, "shown", lambda model, spec=None: {"parameters": 'stop "<x>"'})
    monkeypatch.setattr(ollama, "server_version", lambda spec=None: "0.32.0")
    assert ollama.repetition_penalty_served("m") == 1.1
    monkeypatch.setattr(ollama, "server_version", lambda spec=None: "0.33.1")
    assert ollama.repetition_penalty_served("m") == "unknown"

    def silent(model, spec=None):
        raise RuntimeError("down")

    monkeypatch.setattr(ollama, "shown", silent)
    assert ollama.repetition_penalty_served("m") is None, "an unread server is absent in the stamp, not a guess"


def test_the_judge_s_vllm_says_the_penalty_its_model_file_would_apply(monkeypatch, tmp_path):
    from engines import vllm

    (tmp_path / "generation_config.json").write_text('{"repetition_penalty": 1.05, "temperature": 0.7}')
    monkeypatch.setattr(vllm, "_snapshot", lambda repo: tmp_path)
    assert vllm.model_default("Qwen/Q", "repetition_penalty") == 1.05
    monkeypatch.setattr(vllm, "_snapshot", lambda repo: None)
    assert vllm.model_default("Qwen/Q", "repetition_penalty") is None


def test_a_cloud_engine_stamps_which_key_it_spent_on_and_never_the_key(monkeypatch):
    # a guest resumed on a second account's key, and nothing in its rows said the bill had moved
    import hashlib

    monkeypatch.setenv("CLOUD_API_KEY", "fakefakefakefake")
    got = engines.added_by(CLOUD, "m")
    assert got == {"key_fingerprint": hashlib.sha256(b"fakefakefakefake").hexdigest()[:12]}
    assert "fakefakefakefake" not in str(got)
    monkeypatch.delenv("CLOUD_API_KEY")
    assert engines.added_by(CLOUD, "m") == {}, "no key, no fingerprint, and nothing invented"
