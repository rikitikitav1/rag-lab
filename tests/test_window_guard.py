
import engines
import llm
import pytest
from engines import drivers, ollama
from models.registry import EngineKind, Placement

OLLAMA = engines.EngineSpec(1, "ollama", EngineKind.ollama, "OLLAMA", Placement.gpu)


def test_a_windowed_tag_is_made_from_its_base_with_its_own_window(monkeypatch):
    # the OpenAI door takes no num_ctx per call, so the guest's window has to live in the model
    asked = []
    monkeypatch.setattr(ollama, "post", lambda path, payload, spec=None, timeout=None: asked.append((path, payload)))
    ollama.pull_model("qwen2.5:7b-w16384")
    assert asked[0] == ("/api/pull", {"model": "qwen2.5:7b", "stream": False})
    assert asked[1] == ("/api/create", {"model": "qwen2.5:7b-w16384", "from": "qwen2.5:7b",
                                        "parameters": {"num_ctx": 16384}, "stream": False})
    assert ollama.windowed("qwen2.5:7b") is None and ollama.windowed("llama3.1:8b") is None


def test_an_unloaded_windowed_tag_is_guarded_by_its_own_window(monkeypatch):
    monkeypatch.setattr(drivers.ollama, "context_length", lambda model, spec=None: None)
    assert engines.window_or_configured(OLLAMA, "qwen2.5:7b-w16384") == 16384


def test_an_input_past_the_window_never_reaches_ollama(monkeypatch):
    # ollama kept the head and a tail of an 8849-token input, and the guest scored what was left
    monkeypatch.setattr(llm.engines, "window_or_configured", lambda spec, name: 100)
    with pytest.raises(llm.InputOverWindow, match="window"):
        llm._refuse_an_input_over_the_window(OLLAMA, "qwen2.5:7b", [{"role": "user", "content": "слово " * 400}])
    assert llm._refuse_an_input_over_the_window(OLLAMA, "qwen2.5:7b", [{"role": "user", "content": "short"}]) == (100, 0)


# a live answer and a live context chunk from the stand, counted by qwen2.5's own tokenizer
RUSSIAN_ANSWER = (
    "Связь между признаком и предсказуемым результатом (целевой переменной) является **критически важным "
    "фактором** при определении важности признаков и последующем выборе признаков [source]. Для оценки этой "
    "связи используются такие метрики, как коэффициент корреляции Пирсона (для линейной связи между числовыми "
    "переменными), точечно-бисериальная корреляция (для связи между бинарной и непрерывной переменной) и "
    "\\(R^2\\) для непрерывных целевых переменных [source].\n\nОднако при использовании корреляции для выбора "
    "признаков есть распространённые ошибки: корреляционные метрики, особенно \\(r\\), плохо улавливают "
    "нелинейные связи, а также может игнорироваться избыточность — даже если два признака имеют умеренную "
    "корреляцию с целевой переменной, один из них может быть избыточным, если они сильно коррелируют друг с "
    "другом [source]."
)
RUSSIAN_QWEN = 255
ENGLISH_CHUNK = (
    "[redis-doc/commands/cluster-failover.md]\n# CLUSTER FAILOVER\n## Implementation details and notes\n"
    "* `CLUSTER FAILOVER`, unless the **TAKEOVER** option is specified, does not execute a failover synchronously.\n"
    "  It only *schedules* a manual failover, bypassing the failure detection stage.\n"
    "* An `OK` reply is no guarantee that the failover will succeed.\n"
    "* A replica can only be promoted to a master if it is known as a replica by a majority of the masters in the "
    "cluster.\n  If the replica is a new node that has just been added to the cluster (for example after upgrading "
    "it), it may not yet be known to all the masters in the cluster.\n  To check that the masters are aware of a new "
    "replica, you can send `CLUSTER NODES` or `CLUSTER REPLICAS` to each of the master nodes and check that it "
    "appears as a replica, before sending `CLUSTER FAILOVER` to the replica.\n* To check that the failover has "
    "actually happened you can use `ROLE`, `INFO REPLICATION` (which indicates \"role:master\" after successful "
    "failover), or `CLUSTER NODES` to verify that the state of the cluster has changed sometime after the command "
    "was sent."
)
ENGLISH_QWEN = 271


def test_the_estimate_never_passes_qwens_own_count_in_either_language():
    # a flat 1.25 read this answer as 279 tokens, over its 255, and would refuse a fitting input
    assert len(llm._encoding().encode(RUSSIAN_ANSWER)) / 1.25 > RUSSIAN_QWEN
    assert llm._least_tokens([{"content": RUSSIAN_ANSWER}]) <= RUSSIAN_QWEN
    assert 0.95 * ENGLISH_QWEN <= llm._least_tokens([{"content": ENGLISH_CHUNK}]) <= ENGLISH_QWEN


def test_the_share_is_read_over_the_whole_input_not_per_message():
    whole = llm._least_tokens([{"content": RUSSIAN_ANSWER}, {"content": ENGLISH_CHUNK}])
    assert whole <= RUSSIAN_QWEN + ENGLISH_QWEN
    assert llm._cyrillic_share(ENGLISH_CHUNK) == 0 and llm._cyrillic_share(RUSSIAN_ANSWER) > 0.95


def test_a_cut_the_server_made_is_read_off_its_prompt_count():
    # ollama 0.32 answers a cut input with exactly the window less the budget, plus two
    assert llm._cut_by_the_server(8192, {"max_tokens": 4096}, 4098)
    assert not llm._cut_by_the_server(8192, {"max_tokens": 4096}, 4097)
    assert not llm._cut_by_the_server(None, {"max_tokens": 4096}, 4098)


def test_a_guest_axis_over_the_window_is_dropped_however_ragas_wrapped_it():
    from job_handlers import judging

    try:
        try:
            raise llm.InputOverWindow("over")
        except llm.InputOverWindow as inner:
            raise RuntimeError("The output parser failed") from inner
    except RuntimeError as wrapped:
        assert judging._chain_has(wrapped, llm.InputOverWindow)
    assert not judging._chain_has(RuntimeError("other"), llm.InputOverWindow)
