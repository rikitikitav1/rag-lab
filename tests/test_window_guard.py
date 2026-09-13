
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
    assert llm._refuse_an_input_over_the_window(OLLAMA, "qwen2.5:7b", [{"role": "user", "content": "short"}]) == 100


def test_the_count_errs_low_so_a_fitting_input_is_never_refused():
    # cl100k read Russian 1.21 times longer than qwen on the stand's own docs
    text = "Семантический поиск находит куски корпуса по смыслу вопроса. " * 50
    assert llm._least_tokens([{"content": text}]) * llm._OVERCOUNT <= len(llm._encoding().encode(text)) + 1


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
